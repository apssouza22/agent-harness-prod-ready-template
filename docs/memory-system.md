# Long-Term Memory System

This document describes the long-term memory system implemented in this project: how it works, how it is wired into agents, and how to configure and operate it.

## Overview

The project stores **per-user durable facts** in **PostgreSQL with pgvector**. The harness owns the full pipeline — there is no external memory SDK dependency.

At a high level, each agent invocation follows two memory phases:

1. **Retrieval (before invoke)** — embed the latest user message, run hybrid search, and inject relevant facts into the system prompt.
2. **Ingest (after invoke, background)** — extract facts from the conversation, reconcile them against existing memories (ADD / UPDATE / DELETE), and persist changes without blocking the HTTP response.

Memory is scoped by **`user_id`**, not by session. Facts learned in one conversation are available in future sessions for the same user.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent invoke lifecycle                          │
├─────────────────────────────────────────────────────────────────────────┤
│  before_invoke                                                          │
│    └─ MemoryMiddleware.search(user_id, last_message)                     │
│         └─ ctx.metadata["long_term_memory"] = formatted bullet list     │
│                                                                         │
│  core_invoke                                                            │
│    └─ graph.ainvoke({ messages, long_term_memory, ... })                │
│         └─ system prompt includes {long_term_memory}                    │
│                                                                         │
│  after_invoke (background, non-blocking)                                │
│    └─ MemoryService.schedule_add(user_id, messages, metadata)           │
│         └─ FactExtractor → MemoryReconciler → pgvector CRUD             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Package layout

All memory code lives under `src/app/core/memory/`:

| Module | Responsibility |
|--------|----------------|
| `memory.py` | Public façade — `MemoryService` (`search`, `add`, `schedule_add`) |
| `engine.py` | `LongTermMemoryEngine` — orchestrates extract, reconcile, store, and search |
| `vector_store.py` | `PgVectorMemoryStore` — pgvector CRUD, HNSW index, tsvector keyword search |
| `entity_store.py` | `EntityLinkStore` — parallel `{collection}_entity_links` table |
| `embedder.py` | `MemoryEmbedder` — OpenAI or Bedrock embeddings |
| `extractor.py` | `FactExtractor` — LLM structured fact extraction from conversations |
| `reconciler.py` | `MemoryReconciler` — ADD / UPDATE / DELETE / NONE decisions |
| `entity_extractor.py` | `EntityExtractor` — extracts people, orgs, tools from memory text |
| `entities.py` | `MemoryEntity`, normalization, overlap scoring |
| `search_ranker.py` | Hybrid fusion and entity-boost reranking |
| `prompts.py` | Fact extraction, reconciliation, and entity extraction prompts |
| `config_builder.py` | Provider and model resolution for memory LLM and embedder |
| `llm.py` | `make_memory_chat_model()` |
| `factory.py` | `make_memory_service()` |
| `middleware.py` | `MemoryMiddleware` — before/after invoke hooks |
| `models.py` | Pydantic schemas for LLM structured output |

## Initialization

Memory is initialized at application startup in `src/app/main.py`:

```python
memory_service = make_memory_service(settings)
app.state.memory_service = memory_service
```

`MemoryService` uses **lazy initialization**. The underlying `LongTermMemoryEngine` is created on the first `search()` or `add()` call. Initialization:

1. Ensures the pgvector extension exists.
2. Creates the memory table and indexes if they do not exist.
3. Creates the entity links table and indexes if they do not exist.

The memory layer shares the same async PostgreSQL connection pool as LangGraph checkpointing (`src/app/core/db/connection_pool.py`). Docker Compose uses the `pgvector/pgvector:pg16` image.

When `LONG_TERM_MEMORY_ENABLED` is `false`, all memory operations return early and `MemoryMiddleware` becomes a no-op.

## Public API

`MemoryService` exposes three methods:

```python
async def search(user_id: int, query: str) -> str
```

Searches memories for a user and returns a formatted bullet list (`"* fact\n* fact"`). Returns an empty string on error or when memory is disabled.

```python
async def add(user_id: int, messages: list[dict], metadata: dict | None = None) -> None
```

Runs the full extract → reconcile → persist pipeline synchronously.

```python
def schedule_add(user_id: int, messages: list[dict], metadata: dict | None = None) -> None
```

Schedules `add()` via `asyncio.create_task` so the HTTP response is not blocked.

`MemoryServiceDep` is available in `src/app/dependencies.py` for FastAPI dependency injection, but **there are no dedicated REST endpoints for memory CRUD**. Memory is consumed indirectly through agent chat endpoints.

## Retrieval

Retrieval is the process of finding the most relevant stored facts for a user given their latest message. It runs synchronously on every agent invoke (before the LLM call) via `LongTermMemoryEngine.search()` and is orchestrated by `MemoryService.search()`, which formats the results as a bullet list for the system prompt.

The design goal is **recall under varied phrasing**: a user might say "tell me about my job" when the stored fact is "Is a software engineer at Acme Corp". No single search technique handles all cases well, so the engine combines up to three complementary signals and fuses them into a final ranking.

### End-to-end pipeline

```
User message (query string)
        │
        ├──────────────────────────────────────────┐
        │                                          │
        ▼                                          ▼
  MemoryEmbedder.embed(query)              query string (raw text)
        │                                          │
        ▼                                          ▼
  search_vector()                          search_keyword()  [optional]
  (pgvector cosine, HNSW)                  (tsvector + ts_rank_cd)
        │                                          │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
              Candidate pool (up to search_limit × pool_multiplier)
                       │
                       ▼
              Entity resolution  [optional]
                ├─ list_user_entities(user_id)
                ├─ match_query_entities(query, known_entities)
                └─ get_entities_for_memories(user_id, candidate_ids)
                       │
                       ▼
              Rank / fuse (mode depends on config flags)
                       │
                       ▼
              Top N results (LONG_TERM_MEMORY_SEARCH_LIMIT)
                       │
                       ▼
              Formatted bullet list → system prompt
```

**Query input.** Retrieval always uses the **last user message** as the query — either from `MemoryMiddleware.before_invoke()` (`ctx.messages[-1].content`) or from the streaming path in `agent_chatbot.py`. The full conversation history is not embedded at search time.

**User scoping.** Every database query filters on `payload->>'user_id' = <user_id>`. Memories from other users are never returned, regardless of semantic similarity.

**Candidate pool sizing.** Before fusion, the engine fetches a larger candidate pool than the final result count:

```
pool_limit = max(search_limit, search_limit × LONG_TERM_MEMORY_ENTITY_SEARCH_POOL_MULTIPLIER)
```

With defaults (`search_limit=10`, `pool_multiplier=3`), both vector and keyword searches retrieve up to **30 candidates** each. Fusion then trims to the top 10. The larger pool gives keyword-only or entity-boosted hits a chance to surface even when they rank lower in pure vector search.

### Search technique 1: Vector (semantic) search

Vector search is the primary recall mechanism. It finds memories whose **meaning** is closest to the query, even when the exact words differ.

**How it works:**

1. The query string is embedded via `MemoryEmbedder` using the configured provider (OpenAI or Bedrock) and model (`LONG_TERM_MEMORY_EMBEDDER_MODEL`, default `text-embedding-3-small`).
2. The embedding vector is compared against stored memory vectors using pgvector's **cosine distance operator** (`<=>`).
3. Results are ordered by ascending distance (closest first) and limited to `pool_limit`.

**SQL shape** (from `PgVectorMemoryStore.search_vector()`):

```sql
SELECT id, vector <=> $query_vector AS distance, payload
FROM longterm_memory
WHERE payload->>'user_id' = $user_id
ORDER BY distance
LIMIT $pool_limit
```

**Index.** An **HNSW index** (`longterm_memory_hnsw_idx`) accelerates approximate nearest-neighbor lookups using `vector_cosine_ops`. HNSW trades a small amount of recall accuracy for sub-linear query time as the table grows.

**Score interpretation.** The raw score stored in `MemoryRecord.score` is the **cosine distance** (0 = identical, higher = more distant). During fusion, this is converted to a similarity in `[0, 1]`:

```
vector_similarity = max(0.0, 1.0 - cosine_distance)
```

**Strengths:**
- Handles paraphrasing ("my programming language" → "Uses Python daily")
- Works across languages when the embedding model supports them
- Finds related concepts even without shared keywords

**Weaknesses:**
- Can miss exact identifiers (project codes, SKUs, uncommon names) that embeddings treat as distant
- May return semantically similar but factually unrelated memories
- Embedding quality and dimensionality (`LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS`, default 1536) directly affect recall

**When memories are stored**, the same embedder embeds the fact text. On UPDATE, the vector is re-embedded so search stays aligned with the current text.

### Search technique 2: Keyword (full-text) search

Keyword search complements vector search by matching **exact terms** in the memory text and linked entity names. It uses PostgreSQL's built-in full-text search — not a separate search engine.

**Indexed document.** Each memory row has a `search_text TSVECTOR` column. At insert/update time, the engine builds a search document from:

1. The memory text (`payload.data`)
2. All linked entity names (`payload.entities[].name`)

```python
# PgVectorMemoryStore.build_search_document()
"Works at Acme Corp" + "Acme Corp" + "Python"
→ "Works at Acme Corp Acme Corp Python"
```

This document is tokenized with `to_tsvector(config, document)` where `config` is `LONG_TERM_MEMORY_KEYWORD_SEARCH_CONFIG` (default `simple`).

**Query parsing.** The user's message is parsed with `websearch_to_tsquery`, which supports natural web-style syntax (quoted phrases, negation with `-`, implicit AND between terms).

**Ranking.** Matching rows are scored with `ts_rank_cd(search_text, query)`, which implements a **cover density** ranking — similar in spirit to BM25. It rewards documents where query terms appear close together and with high frequency. Results are ordered by descending rank.

**SQL shape** (from `PgVectorMemoryStore.search_keyword()`):

```sql
SELECT id, ts_rank_cd(search_text, websearch_to_tsquery($config, $query)) AS rank, payload
FROM longterm_memory
WHERE search_text @@ websearch_to_tsquery($config, $query)
  AND payload->>'user_id' = $user_id
ORDER BY rank DESC
LIMIT $pool_limit
```

**Index.** A **GIN index** on `search_text` (`longterm_memory_search_text_idx`) makes `@@` containment checks fast.

**Score normalization.** Raw `ts_rank_cd` scores vary across queries. During fusion, keyword scores are **min-max normalized** against the maximum score in the candidate set:

```
normalized_keyword_score = rank / max(rank)   # 0.0 if max ≤ 0
```

**Strengths:**
- Precise matching on proper nouns, product names, project codes, and technical terms
- Can surface a memory that vector search missed entirely (a "keyword-only hit")
- No additional API call — runs entirely in PostgreSQL

**Weaknesses:**
- No semantic understanding ("my employer" won't match "Acme Corp" without shared tokens)
- Sensitive to stemming/tokenization config (`simple` does minimal stemming)
- Empty or very short queries return no keyword results

**Toggle:** `LONG_TERM_MEMORY_KEYWORD_SEARCH_ENABLED` (default `true`).

### Search technique 3: Entity overlap

Entity search adds a structured signal on top of vector and keyword results. Instead of matching raw text, it reasons about **named entities** (people, organizations, tools, projects) that were extracted and linked during ingest.

**Entity storage.** On ADD/UPDATE, `EntityExtractor` identifies entities in the memory text. They are stored in two places:

1. **`longterm_memory_entity_links` table** — one row per `(user_id, memory_id, entity_normalized)` with indexes on `(user_id, entity_normalized)` and `(memory_id)`.
2. **`payload.entities`** — embedded in the JSONB payload for redundancy.

**Query entity resolution.** At search time, the engine does not run an LLM call. Instead it uses **substring matching** against the user's known entity vocabulary:

```python
# entities.py — match_query_entities()
known_entities = await entity_store.list_user_entities(user_id)
query_entities = match_query_entities(query, known_entities)
```

`match_query_entities()` lowercases the query and checks whether each known normalized entity name appears as a substring. Longer entity names are checked first to avoid partial matches (e.g., "Acme" vs "Acme Corp"). If a shorter entity is already covered by a longer match, it is skipped.

**Memory entity lookup.** For each candidate memory (from vector + keyword results), the engine loads linked entities from the entity store and merges them with any entities in the payload:

```python
memory_entities = await entity_store.get_entities_for_memories(user_id, candidate_ids)
memory_entities = _merge_payload_entities(records, memory_entities)
```

**Overlap score.** The entity signal for a memory is the fraction of query-matched entities that also appear in that memory:

```
entity_overlap = |query_entities ∩ memory_entities| / |query_entities|
```

If the query mentions no known entities, or the memory has no linked entities, the overlap is `0.0`.

**Example:**

| Query | Known entities in query | Memory entities | Overlap |
|-------|------------------------|-----------------|---------|
| "How is the Acme Corp project going?" | `{acme corp}` | `{acme corp, python}` | `1.0` |
| "Tell me about my tools" | `{}` | `{python, rust}` | `0.0` |
| "Acme and Atlas updates" | `{acme corp, atlas}` | `{acme corp}` | `0.5` |

**Strengths:**
- Boosts memories about the same person/org/tool even when the query uses different surrounding words
- Cheap at query time (no LLM call, just DB lookups and string matching)
- Helps disambiguate when multiple memories are semantically similar

**Weaknesses:**
- Only works for entities that were previously extracted and stored
- Substring matching can produce false positives on short entity names
- Cannot discover new entities at search time — only matches against known vocabulary

**Toggle:** `LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED` (default `true`).

### Ranking modes

The engine selects one of three ranking strategies based on configuration flags. The decision tree in `LongTermMemoryEngine._fuse_search_results()`:

```
keyword_enabled AND entity_enabled?
  └─ YES → fuse_hybrid_search_results()     [full hybrid fusion]
  └─ NO  → keyword_enabled?
              └─ YES → (unreachable — keyword path requires entity block above)
              └─ NO  → entity_enabled?
                          └─ YES → rank_records_with_entity_boost()  [vector + entity rerank]
                          └─ NO  → vector_records[:limit]             [pure vector]
```

In practice, with default settings, **full hybrid fusion** is always used.

#### Mode A: Full hybrid fusion (default)

Used when `LONG_TERM_MEMORY_KEYWORD_SEARCH_ENABLED=true`.

All candidates from vector and keyword searches are merged by ID (deduplicated). Each candidate receives a fused score:

```
fused_score = (vector_weight × vector_similarity)
            + (keyword_weight × normalized_keyword_score)
            + (entity_weight × entity_overlap)
```

Default weights sum to `1.0`:

| Signal | Weight | Config variable |
|--------|--------|-----------------|
| Vector similarity | 0.55 | `LONG_TERM_MEMORY_HYBRID_VECTOR_WEIGHT` |
| Keyword rank | 0.30 | `LONG_TERM_MEMORY_HYBRID_KEYWORD_WEIGHT` |
| Entity overlap | 0.15 | `LONG_TERM_MEMORY_HYBRID_ENTITY_WEIGHT` |

A memory that appears in only one search channel still competes — missing signals contribute `0.0` for that channel. This is how **keyword-only hits** (memories with strong term match but weak semantic similarity) can outrank pure vector results.

**Worked example** (from unit tests):

| Memory | Vector distance | Keyword rank | Entity overlap | Fused (approx) |
|--------|----------------|--------------|----------------|----------------|
| "Uses Rust daily" | 0.55 → sim 0.45 | — | — | ~0.25 |
| "Project Atlas uses PostgreSQL" | not in vector top | 0.92 → norm 1.0 | — | ~0.30 |

The keyword-only hit wins despite never appearing in vector results.

#### Mode B: Vector + entity rerank

Used when keyword search is disabled but entity boost is enabled.

Vector results are reranked by **reducing cosine distance** when entity overlap exists (lower distance = better rank):

```
boosted_distance = cosine_distance - (entity_overlap × LONG_TERM_MEMORY_ENTITY_BOOST_WEIGHT)
```

Default `LONG_TERM_MEMORY_ENTITY_BOOST_WEIGHT` is `0.15`. A memory with full entity overlap gets its distance reduced by 0.15, potentially jumping ahead of semantically closer but entity-unrelated memories.

#### Mode C: Pure vector

Used when both `LONG_TERM_MEMORY_KEYWORD_SEARCH_ENABLED` and `LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED` are `false`.

The top `search_limit` vector results are returned as-is, ordered by cosine distance. No fusion or reranking occurs.

### Retrieval output

After ranking, `LongTermMemoryEngine.search()` returns a list of `{id, memory, score}` dicts. `MemoryService.search()` formats them for prompt injection:

```
* Is a software engineer
* Prefers vegetarian restaurants
* Planning a trip to Japan in March
```

If no results are found, or search fails, the middleware sets `"No relevant memory found."` as the fallback string.

### Tuning retrieval

| Goal | Suggested change |
|------|-----------------|
| More exact-name matching | Increase `LONG_TERM_MEMORY_HYBRID_KEYWORD_WEIGHT` |
| More semantic/paraphrase recall | Increase `LONG_TERM_MEMORY_HYBRID_VECTOR_WEIGHT` |
| Better person/org/tool disambiguation | Increase `LONG_TERM_MEMORY_HYBRID_ENTITY_WEIGHT` or `LONG_TERM_MEMORY_ENTITY_BOOST_WEIGHT` |
| Larger candidate pool for fusion | Increase `LONG_TERM_MEMORY_ENTITY_SEARCH_POOL_MULTIPLIER` |
| More results in prompt | Increase `LONG_TERM_MEMORY_SEARCH_LIMIT` |
| Disable keyword search entirely | `LONG_TERM_MEMORY_KEYWORD_SEARCH_ENABLED=false` |
| Disable entity signals entirely | `LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED=false` |
| Different tokenization/stemming | Change `LONG_TERM_MEMORY_KEYWORD_SEARCH_CONFIG` (e.g., `english`) |

**Note:** `LONG_TERM_MEMORY_HYBRID_RRF_K` and `LONG_TERM_MEMORY_DEDUP_DISTANCE` are defined in config but not used by the current fusion implementation. Fusion uses weighted normalized scores, not reciprocal rank fusion (RRF).

## Ingest flow

Ingest runs in the background after each successful agent invoke via `schedule_add()`:

```
Conversation messages
        │
        ▼
  FactExtractor (LLM)
        │  {"facts": ["...", "..."]}
        ▼
  For each fact: embed + vector-search similar existing memories
        │
        ▼
  MemoryReconciler (LLM)
        │  ADD / UPDATE / DELETE / NONE per fact
        ▼
  Apply actions
        ├─ ADD    → new UUID, insert vector + payload, sync entity links
        ├─ UPDATE → re-embed, update vector/payload/search_text, replace entity links
        ├─ DELETE → delete row + entity links
        └─ NONE   → skip
```

### Fact extraction

`FactExtractor` uses an LLM with structured JSON output. Facts are extracted **from user messages only** — assistant and system messages are excluded.

The default prompt (`prompts.py`) instructs the model to capture:

- Personal preferences and dislikes
- Important personal details (names, relationships, dates)
- Plans, intentions, and goals
- Activity and service preferences
- Health and wellness preferences
- Professional details
- Miscellaneous durable facts

Facts are recorded in the **same language** as the user input. Custom extraction guidelines can be appended via `LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS`.

### Reconciliation

`MemoryReconciler` compares newly extracted facts against similar existing memories and decides:

| Action | When to use |
|--------|-------------|
| **ADD** | Genuinely new information not present in memory |
| **UPDATE** | New fact supersedes or refines an existing memory (e.g., job title change) |
| **DELETE** | New fact contradicts or explicitly retracts an existing memory |
| **NONE** | Fact is semantically equivalent to an existing memory |

Candidate memories are found by embedding each new fact and searching for similar vectors scoped to the user. Temporary IDs are assigned for the reconciliation LLM prompt and mapped back to real UUIDs before applying actions.

**Fallback:** if reconciliation returns no actions, all extracted facts are ADDed so nothing is silently dropped.

### Entity linking

When `LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED` is `true`, `EntityExtractor` runs on ADD and UPDATE. Extracted entities are:

- Stored in the `{collection}_entity_links` table for search boosting.
- Embedded in the memory payload as `entities: [{name, type}]`.
- Included in the `search_text` tsvector document alongside the memory text.

## Storage schema

Schema is created at runtime — there are no Alembic migration files for memory tables.

### Memory table (`longterm_memory` by default)

```sql
CREATE TABLE longterm_memory (
    id         UUID PRIMARY KEY,
    vector     vector(1536),       -- dimension from LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS
    payload    JSONB NOT NULL,
    search_text TSVECTOR
);

-- HNSW index on vector (cosine)
CREATE INDEX longterm_memory_hnsw_idx
    ON longterm_memory USING hnsw (vector vector_cosine_ops);

-- GIN index on search_text
CREATE INDEX longterm_memory_search_text_idx
    ON longterm_memory USING GIN (search_text);
```

### Payload structure

Each row's `payload` JSONB contains:

```json
{
  "user_id": "42",
  "session_id": "abc-123",
  "agent_name": "chatbot",
  "data": "Is a software engineer",
  "hash": "md5-of-text",
  "created_at": "2026-01-15T10:30:00+00:00",
  "updated_at": "2026-01-15T10:30:00+00:00",
  "entities": [
    {"name": "Acme Corp", "type": "organization"}
  ]
}
```

The `data` field holds the human-readable memory text. The `hash` field is an MD5 digest of the text, used for change detection.

### Entity links table (`longterm_memory_entity_links` by default)

```sql
CREATE TABLE longterm_memory_entity_links (
    id                BIGSERIAL PRIMARY KEY,
    user_id           TEXT NOT NULL,
    memory_id         UUID NOT NULL,
    entity_name       TEXT NOT NULL,
    entity_normalized TEXT NOT NULL,
    entity_type       TEXT NOT NULL DEFAULT 'unknown',
    UNIQUE (user_id, memory_id, entity_normalized)
);
```

## Agent integration

### MemoryMiddleware

`MemoryMiddleware` (`src/app/core/memory/middleware.py`) is registered in agent factories and hooks into the agent pipeline:

- **`before_invoke`** — searches memory using the last user message and stores the result in `ctx.metadata["long_term_memory"]`.
- **`after_invoke`** — schedules a background memory update with the conversation messages and session metadata.

### Chatbot agent

The chatbot is the primary consumer of retrieved memory.

**System prompt injection** — `src/app/agents/chatbot/system.md` includes a `{long_term_memory}` placeholder under "What you know about the user". The chat node loads this via `load_system_prompt(long_term_memory=state.long_term_memory)`.

**Graph state** — `GraphState` (`src/app/core/common/model/graph.py`) carries `long_term_memory: str` alongside `messages` and `dialogue_state`.

**Streaming path** — streaming bypasses middleware for retrieval and update. `agent_invoke_stream()` calls `MemoryService.search()` directly before `astream()` and `schedule_add()` after the stream completes.

| Path | Retrieval | Prompt injection | Background ingest |
|------|-----------|------------------|-------------------|
| Chatbot sync invoke | `MemoryMiddleware` | Yes (`GraphState`) | `MemoryMiddleware` |
| Chatbot stream | Direct `MemoryService` | Yes | Direct `schedule_add` |
| Deep research sync | `MemoryMiddleware` | No | `MemoryMiddleware` |
| Deep research stream | None | No | Direct `schedule_add` |

### Deep research agent

`MemoryMiddleware` runs on non-stream invokes and schedules background ingest, but retrieved memory is **not injected** into deep research prompts. The agent still learns from conversations via background ingest.

### Text-to-SQL agent

The text-to-SQL agent does not use this pgvector memory system. It uses Deep Agents file-based memory (`./AGENTS.md`) instead.

## Long-term memory vs other persistence layers

This project has three distinct persistence concerns on the same PostgreSQL instance:

| Aspect | LangGraph checkpoints | Long-term memory | Dialogue state |
|--------|----------------------|------------------|----------------|
| Scope | Per **session** (`thread_id`) | Per **user** (`user_id`) | Per **session** |
| Purpose | Conversation state, graph replay | Durable user facts across sessions | Structured snapshot of current conversation |
| Storage | LangGraph tables via `AsyncPostgresSaver` | Custom pgvector + entity_links tables | Custom `dialogue_state` table |
| Used in prompt | Via `messages` in graph state | Via `long_term_memory` string | Via `dialogue_state` string |
| Lifecycle | Tied to chat session | Persists beyond session deletion | Tied to chat session |

Checkpoints store raw message history and graph node state. Long-term memory stores **distilled facts** extracted from conversations, not transcripts.

## Configuration

All settings are defined in `src/app/core/common/config.py` and documented in `.env.example`.

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LONG_TERM_MEMORY_ENABLED` | `true` | Master toggle for all memory operations |
| `LONG_TERM_MEMORY_MODEL` | `gpt-5-nano` | LLM for fact extraction, reconciliation, and entity extraction |
| `LONG_TERM_MEMORY_LLM_PROVIDER` | `""` | Override LLM provider (`openai` / `bedrock`); falls back to `DEFAULT_LLM_PROVIDER` |
| `LONG_TERM_MEMORY_EMBEDDER_MODEL` | `text-embedding-3-small` | Embedding model for vector search |
| `LONG_TERM_MEMORY_EMBEDDER_PROVIDER` | `""` | Override embedder provider |
| `LONG_TERM_MEMORY_COLLECTION_NAME` | `longterm_memory` | pgvector table name |
| `LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS` | optional | Extra guidelines appended to the fact extraction prompt |

### Search settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LONG_TERM_MEMORY_SEARCH_LIMIT` | `10` | Final number of memories returned per search |
| `LONG_TERM_MEMORY_RECONCILE_CANDIDATE_LIMIT` | `5` | Similar memories considered per fact during reconciliation |
| `LONG_TERM_MEMORY_ENTITY_SEARCH_POOL_MULTIPLIER` | `3` | Candidate pool size multiplier before fusion |
| `LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS` | `1536` | Vector dimension (falls back to `CACHE_EMBEDDING_DIMENSIONS`) |

### Hybrid search settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LONG_TERM_MEMORY_KEYWORD_SEARCH_ENABLED` | `true` | Enable full-text keyword signal |
| `LONG_TERM_MEMORY_KEYWORD_SEARCH_CONFIG` | `simple` | Postgres text search configuration |
| `LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED` | `true` | Enable entity extraction and entity signal |
| `LONG_TERM_MEMORY_ENTITY_BOOST_WEIGHT` | `0.15` | Entity boost weight in vector-only reranking mode |
| `LONG_TERM_MEMORY_HYBRID_VECTOR_WEIGHT` | `0.55` | Vector fusion weight |
| `LONG_TERM_MEMORY_HYBRID_KEYWORD_WEIGHT` | `0.30` | Keyword fusion weight |
| `LONG_TERM_MEMORY_HYBRID_ENTITY_WEIGHT` | `0.15` | Entity fusion weight |
| `LONG_TERM_MEMORY_HYBRID_RRF_K` | `60` | Retained for config compatibility; fusion uses weighted scores |

### Database prerequisites

Memory requires the standard PostgreSQL connection settings:

```
POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_POOL_SIZE
```

### Bifrost integration

When `BIFROST_ENABLED=true`, `MemoryEmbedder` routes OpenAI embedding calls through the Bifrost gateway's `/v1` endpoint. Fact extraction and reconciliation use the shared LLM factory like other agents.

## Error handling and observability

- **Search errors** — logged via `logger.exception("failed_to_get_relevant_memory")`, returns empty string (agent continues without memory).
- **Ingest errors** — logged via `logger.exception("failed_to_update_long_term_memory")`, silently swallowed in background task.
- **Reconciliation fallback** — if the reconciler returns no actions, facts are ADDed with a warning log.
- **Disabled memory** — all methods return early without error when `LONG_TERM_MEMORY_ENABLED=false`.

Key structured log events:

| Event | When |
|-------|------|
| `long_term_memory_initialized` | Engine first initialized |
| `relevant_memory_retrieved` | Search completed |
| `memory_added` / `memory_updated` / `memory_deleted` | Reconciled action applied |
| `memory_reconcile_fallback_to_add` | Reconciler returned empty, falling back to ADD |
| `memory_hybrid_search_applied` | Hybrid fusion completed |

## Testing

Unit tests live under `tests/unit/`:

| Test file | Coverage |
|-----------|----------|
| `test_memory_service.py` | Service API, disabled flag, background task scheduling |
| `test_memory_engine.py` | ADD / UPDATE / DELETE flows, hybrid search |
| `test_memory_reconciler.py` | Action parsing, unknown ID handling |
| `test_memory_entity_extractor.py` | Structured output conversion |
| `test_memory_embedder.py` | Provider and model resolution |
| `test_memory_hybrid_search.py` | Fusion ranking, search document building |
| `test_memory_entities.py` | Entity normalization and overlap scoring |
| `test_di_factories.py` | `make_memory_service` factory |

Tests mock store and engine dependencies — there are no integration tests against a live PostgreSQL instance for memory.

Run memory tests:

```bash
pytest tests/unit/test_memory_*.py -v
```

## Operations

### Disable memory

Set `LONG_TERM_MEMORY_ENABLED=false` in your environment. No code changes are required.

### Inspect stored memories

Connect to PostgreSQL and query the memory table directly:

```sql
SELECT id, payload->>'data' AS memory, payload->>'user_id' AS user_id
FROM longterm_memory
WHERE payload->>'user_id' = '42'
ORDER BY payload->>'updated_at' DESC;
```

### Change the table name

Set `LONG_TERM_MEMORY_COLLECTION_NAME` to a custom name. The entity links table will be named `{collection}_entity_links`. Tables are created automatically on first use.

### Custom extraction behavior

Set `LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS` to natural-language guidelines. These are appended to the fact extraction prompt. Example from `.env.example`:

```
LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS="Extract user preferences, goals, and constraints. Exclude personal identifiers."
```

## Related documentation

- [ARTICLE.md — Section 5](./ARTICLE.md) — architecture overview within the broader agent harness article
- [middleware-for-agent-harness.md](./middleware-for-agent-harness.md) — `MemoryMiddleware` hook behavior in the agent pipeline
- [AGENTS.md](../AGENTS.md) — AI agent development guidelines for this project
