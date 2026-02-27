from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from src.app.core.common.logging import logger
from src.app.core.metrics.metrics import llm_inference_duration_seconds, tokens_in_counter, tokens_out_counter


async def model_invoke_with_metrics(
    model,
    model_input: LanguageModelInput,
    model_name:str,
    agent_name:str,
    config: RunnableConfig | None = None
):
    with llm_inference_duration_seconds.labels(model=model_name, agent_name=agent_name).time():
        response = await model.ainvoke(model_input, config)
    record_token_usage(response, model_name, agent_name)
    return response

def record_token_usage(response: BaseMessage, model: str, agent_name: str) -> None:
    """Extract token usage from an LLM response and increment Prometheus counters.

    Uses LangChain's standardised ``usage_metadata`` attribute which is
    populated by all major providers (OpenAI, Anthropic, Google, etc.).
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        logger.debug("no_usage_metadata_in_response", model=model, agent_name=agent_name)
        return

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    tokens_in_counter.labels(agent_name=agent_name).inc(input_tokens)
    tokens_out_counter.labels(agent_name=agent_name).inc(output_tokens)

    logger.debug(
        "llm_token_usage_recorded",
        model=model,
        agent_name=agent_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
