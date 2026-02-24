"""Main graph node functions for the Deep Research agent.

This module provides the top-level node functions used by the main deep research
graph: clarification, research brief generation, and final report generation.
Subgraph logic (supervisor, researcher) lives in dedicated standalone classes.
"""

from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.types import Command

from src.app.agents.open_deep_research.config import (
    DEEP_RESEARCH_AGENT_NAME,
    FINAL_REPORT_MODEL,
    MAX_CONCURRENT_RESEARCH_UNITS,
    MAX_RESEARCHER_ITERATIONS,
    RESEARCH_MODEL,
    research_brief_model, clarification_model,
)
from src.app.agents.open_deep_research.prompts import (
    clarify_with_user_instructions,
    final_report_generation_prompt,
    lead_researcher_prompt,
    transform_messages_into_research_topic_prompt,
)
from src.app.agents.open_deep_research.state import (
    AgentState,
)
from src.app.core.common.token_limit import is_token_limit_exceeded, get_model_token_limit
from src.app.core.common.utils import (
    get_today_str,
)
from src.app.core.llm.llm_utils import record_token_usage, record_llm_error
from src.app.core.metrics.metrics import llm_inference_duration_seconds


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.

    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences

    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    messages = state["messages"]


    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    with llm_inference_duration_seconds.labels(model=RESEARCH_MODEL, agent_name=DEEP_RESEARCH_AGENT_NAME).time():
        response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.

    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to research supervisor with initialized context
    """


    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    with llm_inference_duration_seconds.labels(model=RESEARCH_MODEL, agent_name=DEEP_RESEARCH_AGENT_NAME).time():
        response = await research_brief_model.ainvoke([HumanMessage(content=prompt_content)])

    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.

    This function takes all collected research findings and synthesizes them into a
    well-structured, comprehensive final report using the configured report generation model.

    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys

    Returns:
        Dictionary containing the final report and cleared state
    """
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)


    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            with llm_inference_duration_seconds.labels(model=FINAL_REPORT_MODEL, agent_name=DEEP_RESEARCH_AGENT_NAME).time():
                final_report = await final_report_model.ainvoke([
                    HumanMessage(content=final_report_prompt)
                ])

            record_token_usage(final_report, FINAL_REPORT_MODEL, DEEP_RESEARCH_AGENT_NAME)
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            record_llm_error(FINAL_REPORT_MODEL, DEEP_RESEARCH_AGENT_NAME)
            if is_token_limit_exceeded(e, FINAL_REPORT_MODEL):
                current_retry += 1

                if current_retry == 1:
                    model_token_limit = get_model_token_limit(FINAL_REPORT_MODEL)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)

                findings = findings[:findings_token_limit]
                continue
            else:
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }
