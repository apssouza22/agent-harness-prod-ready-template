import os
from typing import Any, Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from src.app.core.guardrails import (
    PIIStrategy,
    PIIType,
    apply_pii_strategy,
    check_content_filter,
    detect_pii,
    evaluate_safety,
    get_safe_replacement_message,
)
from src.app.core.common.config import settings
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.logging import logger
from src.app.core.common.model.message import Message
from src.app.init import langfuse_callback_handler


class TextSQLDeepAgent:
    """SQL Deep Agent that can interact with a SQL database using natural language instructions."""

    def __init__(self, name: str):
        self.name = name
        self.agent = create_sql_deep_agent()

    async def agent_invoke(
        self,
        agent_input: dict[str, Any],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        """Invoke the SQL Deep Agent with the given input and return its response."""
        query = agent_input.get("query", "")

        # Input guardrails: content filter + PII block
        filter_result = check_content_filter(query)
        if filter_result.is_blocked:
            logger.info("text_sql_input_guardrail_blocked", reason=filter_result.reason, session_id=session_id)
            return [Message(role="assistant", content="I cannot process this request. Please rephrase your message.")]

        pii_findings = detect_pii(query, pii_types=[PIIType.API_KEY, PIIType.SSN, PIIType.CREDIT_CARD])
        if pii_findings:
            detected_types = list({f["type"].value for f in pii_findings})
            logger.info("text_sql_input_guardrail_pii_blocked", pii_types=detected_types, session_id=session_id)
            return [Message(
                role="assistant",
                content="Your message contains sensitive information. Please remove it and try again.",
            )]

        config = {
            "callbacks": [langfuse_callback_handler],
            "configurable": {"thread_id": session_id},
            "metadata": {
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
                "user_id": user_id,
                "session_id": session_id,
            },
        }

        response = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        }, config=config)
        messages = process_messages(response["messages"])

        await self.process_safe_output(messages)

        return messages

    async def process_safe_output(self, messages):
        # Output guardrails: PII redaction + safety check
        if messages:
            last = messages[-1]
            if last.role == "assistant":
                modified_content = last.content

                output_pii = detect_pii(modified_content, pii_types=[
                    PIIType.EMAIL, PIIType.CREDIT_CARD, PIIType.SSN,
                    PIIType.PHONE, PIIType.API_KEY, PIIType.IP,
                ])
                if output_pii:
                    redacted = apply_pii_strategy(modified_content, output_pii, PIIStrategy.REDACT)
                    if redacted is not None:
                        modified_content = redacted

                is_safe = await evaluate_safety(modified_content)
                if not is_safe:
                    modified_content = get_safe_replacement_message()

                if modified_content != last.content:
                    messages[-1] = Message(role="assistant", content=modified_content)


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Connect to Chinook database
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)

    model = ChatOpenAI(model="gpt-5-mini", reasoning={"effort": "medium"}, temperature=0)

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    agent = create_deep_agent(
        model=model,
        memory=["./AGENTS.md"],  # Agent identity and general instructions
        skills=[
            "./skills/"
        ],  # Specialized workflows (query-writing, schema-exploration)
        tools=sql_tools,  # SQL database tools
        subagents=[],  # No subagents needed
        backend=FilesystemBackend(root_dir=base_dir),  # Persistent file storage
    )

    return agent
