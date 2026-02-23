"""LLM service for managing LLM calls with retries and fallback mechanisms."""

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import (
    APIError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.app.core.common.config import (
    Environment,
    settings,
)
from src.app.core.common.logging import logger


def _create_model(name: str, **kwargs) -> Dict[str, Any]:
    """Create a model registry entry with shared defaults.

    Args:
        name: Model name (e.g., "gpt-5-mini")
        **kwargs: Additional ChatOpenAI parameters that override defaults

    Returns:
        Dict with "name" and "llm" keys
    """
    defaults = {
        "api_key": settings.OPENAI_API_KEY,
        "max_tokens": settings.MAX_TOKENS,
    }
    defaults.update(kwargs)
    return {"name": name, "llm": ChatOpenAI(model=name, **defaults)}


class LLMRegistry:
    """Registry of available LLM models organized by agent name.

    Models are grouped under agent names, allowing each agent to define
    its own model chain with specific configurations and fallback order.
    The "default" agent serves as a fallback when a requested agent is not found.
    """

    AGENTS: Dict[str, List[Dict[str, Any]]] = {
        "default": [
            _create_model("gpt-5-mini", reasoning={"effort": "low"}),
            _create_model("gpt-5", reasoning={"effort": "medium"}),
            _create_model("gpt-5-nano", reasoning={"effort": "minimal"}),
            _create_model(
                "gpt-4o",
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                top_p=0.95 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
                presence_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
                frequency_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
            ),
            _create_model(
                "gpt-4o-mini",
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                top_p=0.9 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
            ),
        ],
        "deep_research": [
            _create_model("gpt-5", reasoning={"effort": "high"}),
            _create_model("gpt-5-mini", reasoning={"effort": "medium"}),
        ],
    }

    @classmethod
    def get_agent_models(cls, agent_name: str) -> List[Dict[str, Any]]:
        """Get models for a specific agent, falling back to default.

        Args:
            agent_name: Name of the agent

        Returns:
            List of model entries for the agent
        """
        if agent_name in cls.AGENTS:
            return cls.AGENTS[agent_name]
        logger.warning(
            "agent_not_found_using_default",
            agent_name=agent_name,
            available_agents=list(cls.AGENTS.keys()),
        )
        return cls.AGENTS["default"]

    @classmethod
    def get(cls, model_name: str, agent_name: str = "default", **kwargs) -> BaseChatModel:
        """Get an LLM by name for a specific agent with optional argument overrides.

        Args:
            model_name: Name of the model to retrieve
            agent_name: Name of the agent whose model chain to search
            **kwargs: Optional arguments to override default model configuration

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If model_name is not found for the given agent
        """
        models = cls.get_agent_models(agent_name)

        model_entry = None
        for entry in models:
            if entry["name"] == model_name:
                model_entry = entry
                break

        if not model_entry:
            available_models = [entry["name"] for entry in models]
            raise ValueError(
                f"model '{model_name}' not found for agent '{agent_name}'. "
                f"available models: {', '.join(available_models)}"
            )

        if kwargs:
            logger.debug(
                "creating_llm_with_custom_args",
                model_name=model_name,
                agent_name=agent_name,
                custom_args=list(kwargs.keys()),
            )
            return ChatOpenAI(model=model_name, api_key=settings.OPENAI_API_KEY, **kwargs)

        logger.debug("using_default_llm_instance", model_name=model_name, agent_name=agent_name)
        return model_entry["llm"]

    @classmethod
    def get_all_names(cls, agent_name: str = "default") -> List[str]:
        """Get all registered LLM names for a specific agent in order.

        Args:
            agent_name: Name of the agent

        Returns:
            List of LLM names
        """
        models = cls.get_agent_models(agent_name)
        return [entry["name"] for entry in models]

    @classmethod
    def get_model_at_index(cls, index: int, agent_name: str = "default") -> Dict[str, Any]:
        """Get model entry at specific index for a specific agent.

        Args:
            index: Index of the model in the agent's model list
            agent_name: Name of the agent

        Returns:
            Model entry dict
        """
        models = cls.get_agent_models(agent_name)
        if 0 <= index < len(models):
            return models[index]
        return models[0]

    @classmethod
    def get_agent_names(cls) -> List[str]:
        """Get all registered agent names.

        Returns:
            List of agent names
        """
        return list(cls.AGENTS.keys())

