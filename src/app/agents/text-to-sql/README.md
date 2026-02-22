# Text-to-SQL Deep Agent

A natural language to SQL query agent powered by LangChain's **Deep Agents** framework.  This is an advanced version of a text-to-SQL agent with planning, filesystem, and subagent capabilities.

## What is Deep Agents?

Deep Agents is a sophisticated agent framework built on LangGraph that provides:

- **Planning capabilities** - Break down complex tasks with `write_todos` tool
- **Filesystem backend** - Save and retrieve context with file operations
- **Subagent spawning** - Delegate specialized tasks to focused agents
- **Context management** - Prevent context window overflow on complex tasks

## Demo Database

Uses the [Chinook database](https://github.com/lerocha/chinook-database) - a sample database representing a digital media store.

## Quick Start


### Installation

1. Clone the deepagents repository and navigate to this example:

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents/examples/text-to-sql-agent
```

1. Download the Chinook database:

```bash
# Download the SQLite database file
curl -L -o chinook.db https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite
```

## How the Deep Agent Works

### Architecture

```
User Question
     ↓
Deep Agent (with planning)
     ├─ write_todos (plan the approach)
     ├─ SQL Tools
     │  ├─ list_tables
     │  ├─ get_schema
     │  ├─ query_checker
     │  └─ execute_query
     ├─ Filesystem Tools (optional)
     │  ├─ ls
     │  ├─ read_file
     │  ├─ write_file
     │  └─ edit_file
     └─ Subagent Spawning (optional)
     ↓
SQLite Database (Chinook)
     ↓
Formatted Answer
```

### Configuration

Deep Agents uses **progressive disclosure** with memory files and skills:

**AGENTS.md** (always loaded) - Contains:

- Agent identity and role
- Core principles and safety rules
- General guidelines
- Communication style

**skills/** (loaded on-demand) - Specialized workflows:

- **query-writing** - How to write and execute SQL queries (simple and complex)
- **schema-exploration** - How to discover database structure and relationships

The agent sees skill descriptions in its context but only loads the full SKILL.md instructions when it determines which skill is needed for the current task. This **progressive disclosure** pattern keeps context efficient while providing deep expertise when needed.

## Example Queries

### Simple Query

```
"How many customers are from Canada?"
```

The agent will directly query and return the count.

### Complex Query with Planning

```
"Which employee generated the most revenue and from which countries?"
```

The agent will:

1. Use `write_todos` to plan the approach
2. Identify required tables (Employee, Invoice, Customer)
3. Plan the JOIN structure
4. Execute the query
5. Format results with analysis

## Deep Agent Output Example

The Deep Agent shows its reasoning process:

```
Question: Which employee generated the most revenue by country?

[Planning Step]
Using write_todos:
- [ ] List tables in database
- [ ] Examine Employee and Invoice schemas
- [ ] Plan multi-table JOIN query
- [ ] Execute and aggregate by employee and country
- [ ] Format results

[Execution Steps]
1. Listing tables...
2. Getting schema for: Employee, Invoice, InvoiceLine, Customer
3. Generating SQL query...
4. Executing query...
5. Formatting results...

[Final Answer]
Employee Jane Peacock (ID: 3) generated the most revenue...
Top countries: USA ($1000), Canada ($500)...
```
