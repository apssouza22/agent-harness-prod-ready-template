# AI Agent Development Guide

This document provides essential guidelines for AI agents working on this LangGraph FastAPI Agent project.

## Project Overview

This is a production-ready AI agent application built with:
- **LangGraph** for stateful, multi-step AI agent workflows
- **FastAPI** for high-performance async REST API endpoints
- **Langfuse** for LLM observability and tracing
- **PostgreSQL + pgvector** for long-term memory storage (mem0ai)
- **JWT authentication** with session management
- **Prometheus + Grafana** for monitoring

## Quick Reference: Critical Rules

### Import Rules
- **All imports MUST be at the top of the file** - never add imports inside functions or classes

### Logging Rules
- Use **structlog** for all logging
- Log messages must be **lowercase_with_underscores** (e.g., `"user_login_successful"`)
- **NO f-strings in structlog events** - pass variables as kwargs
- Use `logger.exception()` instead of `logger.error()` to preserve tracebacks
- Example: `logger.info("chat_request_received", session_id=session.id, message_count=len(messages))`


### Caching Rules
- **Only cache successful responses**, never cache errors
- Use appropriate cache TTL based on data volatility

### FastAPI Rules
- All routes must have rate limiting decorators
- Use dependency injection for services, database connections, and auth
- All database operations must be async

## Code Style Conventions

### Python/FastAPI
- Use `async def` for asynchronous operations
- Use type hints for all function signatures
- Prefer Pydantic models over raw dictionaries
- Use functional, declarative programming; avoid classes except for services and agents
- File naming: lowercase with underscores (e.g., `user_routes.py`)
- Use the RORO pattern (Receive an Object, Return an Object)

### Error Handling
- Handle errors at the beginning of functions
- Use early returns for error conditions
- Place the happy path last in the function
- Use guard clauses for preconditions
- Use `HTTPException` for expected errors with appropriate status codes

## LangGraph & LangChain Patterns

### Graph Structure
- Use `StateGraph` for building AI agent workflows
- Define clear state schemas using Pydantic models
- Use `CompiledStateGraph` for production workflows
- Implement `AsyncPostgresSaver` for checkpointing and persistence
- Use `Command` for controlling graph flow between nodes

## Database Operations
- Use SQLModel for ORM models (combines SQLAlchemy + Pydantic)
- Use async database operations with asyncpg
- Use LangGraph's AsyncPostgresSaver for agent checkpointing

## Performance Guidelines

- Minimize blocking I/O operations
- Use async for all database and external API calls
- Implement caching for frequently accessed data
- Use connection pooling for database connections
- Optimize LLM calls with streaming responses

## Observability

- Integrate Langfuse for LLM tracing on all agent operations
- Export Prometheus metrics for API performance
- Use structured logging with context binding (request_id, session_id, user_id)
- Track LLM inference duration, token usage, and costs

## Testing & Evaluation

- Implement metric-based evaluations for LLM outputs (see `src/evals/` directory)
- Create custom evaluation metrics as markdown files in `src/evals/metrics/prompts/`
- Use Langfuse traces for evaluation data sources
- Generate JSON reports with success rates

## Configuration Management

- Use environment-specific configuration files (`.env.development`, `.env.staging`, `.env.production`)
- Use Pydantic Settings for type-safe configuration (see `app/core/config.py`)
- Never hardcode secrets or API keys

## Key Dependencies

- **FastAPI** - Web framework
- **LangGraph** - Agent workflow orchestration
- **LangChain** - LLM abstraction and tools
- **Langfuse** - LLM observability and tracing
- **Pydantic v2** - Data validation and settings
- **structlog** - Structured logging
- **mem0ai** - Long-term memory management
- **PostgreSQL + pgvector** - Database and vector storage
- **SQLModel** - ORM for database models
- **tenacity** - Retry logic
- **rich** - Terminal formatting
- **slowapi** - Rate limiting
- **prometheus-client** - Metrics collection

## 10 Commandments for This Project

1. All routes must have rate limiting decorators
2. All LLM operations must have Langfuse tracing
3. All async operations must have proper error handling
4. All logs must follow structured logging format with lowercase_underscore event names
5. All retries must use tenacity library
6. All console outputs should use rich formatting
7. All caching should only store successful responses
8. All imports must be at the top of files
9. All database operations must be async
10. All endpoints must have proper type hints and Pydantic models

## Common Pitfalls to Avoid

- ❌ Using f-strings in structlog events
- ❌ Adding imports inside functions
- ❌ Forgetting rate limiting decorators on routes
- ❌ Missing Langfuse tracing on LLM calls
- ❌ Caching error responses
- ❌ Using `logger.error()` instead of `logger.exception()` for exceptions
- ❌ Blocking I/O operations without async
- ❌ Hardcoding secrets or API keys
- ❌ Missing type hints on function signatures

## When Making Changes

Before modifying code:
1. Read the existing implementation first
2. Check for related patterns in the codebase
3. Ensure consistency with existing code style
4. Add appropriate logging with structured format
5. Include error handling with early returns
6. Add type hints and Pydantic models
7. Verify Langfuse tracing is enabled for LLM calls

## References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- LangChain Documentation: https://python.langchain.com/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Langfuse Documentation: https://langfuse.com/docs


# Task execution plan
Important: Always plan the task step by step before writing code. Ask for permission to proceed with the plan.
Important: Before proceed with the plan, create a new file named `.agents/plans/name-of-the-task.md`. Based on the approved plan, list all necessary implementation steps as GitHub-style checkboxes (`- [ ] Step Description`). Use sub-bullets for granular details within each main step.

- Plans should be detailed enough to execute without ambiguity
- Each task in the plan must include at least one validation test to verify it works
- Assess complexity and single-pass feasibility - can an agent realistically complete this in one go?
- Include a complexity indicator at the top of each plan:
  ✅ Simple - Single-pass executable, low risk
  ⚠️ Medium - May need iteration, some complexity
  🔴 Complex - Break into sub-plans before executing

**CRITICAL: After you successfully complete each step, you MUST update the `.agent/plans/name-of-the-task.md` file by changing the corresponding checkbox from `- [ ]` to `- [x]`.**
Only proceed to the *next* unchecked item after confirming the previous one is checked off in the file. Announce which step you are starting.

