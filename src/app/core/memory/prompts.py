"""Prompt templates for long-term memory fact extraction."""

from datetime import datetime

DEFAULT_FACT_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences.
Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.
This allows for easy retrieval and personalization in future interactions.

Types of Information to Remember:

1. Store Personal Preferences: likes, dislikes, and specific preferences.
2. Maintain Important Personal Details: names, relationships, and important dates.
3. Track Plans and Intentions: upcoming events, trips, goals, and plans.
4. Remember Activity and Service Preferences: dining, travel, hobbies, and services.
5. Monitor Health and Wellness Preferences: dietary restrictions and fitness routines.
6. Store Professional Details: job titles, work habits, and career goals.
7. Miscellaneous Information: favorite books, movies, brands, and similar details.

Guidelines:
- Generate facts solely based on the user's messages. Do not include assistant or system messages.
- If nothing relevant is found, return an empty list.
- Detect the language of the user input and record facts in the same language.
- Return JSON with a single key "facts" whose value is a list of strings.
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.

Examples:

Input:
User: Hi, I am looking for a restaurant in San Francisco.
Assistant: Sure, I can help with that.
Output: {{"facts": ["Looking for a restaurant in San Francisco"]}}

Input:
User: Hi, my name is John. I am a software engineer.
Assistant: Nice to meet you, John!
Output: {{"facts": ["Name is John", "Is a software engineer"]}}

Input:
User: Hi.
Assistant: Hello!
Output: {{"facts": []}}
"""


def build_fact_extraction_prompt(custom_instructions: str | None = None) -> str:
    """Build the system prompt for fact extraction, optionally with custom guidelines."""
    if not custom_instructions:
        return DEFAULT_FACT_EXTRACTION_PROMPT

    return (
        f"{DEFAULT_FACT_EXTRACTION_PROMPT}\n\n"
        "Additional extraction guidelines from the application:\n"
        f"{custom_instructions.strip()}\n"
    )


DEFAULT_MEMORY_RECONCILE_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Compare newly retrieved facts with the existing memory. For each relevant item, decide whether to:
- ADD: Add new information not present in memory
- UPDATE: Replace an existing memory when the new fact supersedes or refines it
- DELETE: Remove an existing memory when the new fact contradicts it or explicitly retracts it
- NONE: Make no change when the information is already represented

Guidelines:
- UPDATE keeps the same memory id and replaces outdated text (for example job title or city changes).
- DELETE removes contradicted facts (for example "dislikes pizza" vs "loves pizza").
- NONE when the fact is semantically equivalent to an existing memory.
- ADD only for genuinely new information.
- Return JSON only.
"""


def build_memory_reconcile_prompt(
    existing_memories: list[dict[str, str]],
    new_facts: list[str],
) -> str:
    """Build the user prompt for reconciling new facts against existing memories."""
    if existing_memories:
        current_memory_part = (
            "Below is the current content of memory which has been collected so far:\n\n"
            f"{existing_memories}\n"
        )
    else:
        current_memory_part = "Current memory is empty.\n"

    facts_block = "\n".join(f"- {fact}" for fact in new_facts)
    return f"""{DEFAULT_MEMORY_RECONCILE_PROMPT}

{current_memory_part}

The new retrieved facts are:
{facts_block}

Return your response in this JSON structure only:
{{
  "memory": [
    {{
      "id": "<existing id for UPDATE/DELETE/NONE, or a new id string for ADD>",
      "text": "<content>",
      "event": "ADD|UPDATE|DELETE|NONE",
      "old_memory": "<previous content, required for UPDATE>"
    }}
  ]
}}

Rules:
- Use only ids from the existing memory list for UPDATE, DELETE, and NONE.
- Generate a new id string only for ADD events.
- If current memory is empty, ADD all relevant new facts.
"""


DEFAULT_ENTITY_EXTRACTION_PROMPT = """You extract named entities from user memory text.

Focus on durable entities such as:
- people
- organizations and companies
- projects and products
- tools and technologies
- places and locations

Return JSON only:
{
  "entities": [
    {"name": "John", "type": "person"},
    {"name": "Acme Corp", "type": "organization"}
  ]
}

Rules:
- Include only entities explicitly mentioned or clearly implied by the text.
- Do not invent entities.
- Use concise canonical names.
- If no entities are present, return {"entities": []}.
"""
