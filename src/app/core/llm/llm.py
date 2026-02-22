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


class LLMService:
    """Service for managing LLM calls with retries and circular fallback.

    This service handles all LLM interactions with automatic retry logic,
    rate limit handling, and circular fallback through the agent's available models.
    """

    def __init__(self, agent_name: str = "default"):
        """Initialize the LLM service for a specific agent.

        Args:
            agent_name: Name of the agent to load models for
        """
        self._agent_name = agent_name
        self._llm: Optional[BaseChatModel] = None
        self._current_model_index: int = 0

        all_names = LLMRegistry.get_all_names(agent_name)
        try:
            self._current_model_index = all_names.index(settings.DEFAULT_LLM_MODEL)
            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL, agent_name)
            logger.info(
                "llm_service_initialized",
                agent_name=agent_name,
                default_model=settings.DEFAULT_LLM_MODEL,
                model_index=self._current_model_index,
                total_models=len(all_names),
                environment=settings.ENVIRONMENT.value,
            )
        except (ValueError, Exception) as e:
            self._current_model_index = 0
            models = LLMRegistry.get_agent_models(agent_name)
            self._llm = models[0]["llm"]
            logger.warning(
                "default_model_not_found_using_first",
                agent_name=agent_name,
                requested=settings.DEFAULT_LLM_MODEL,
                using=all_names[0] if all_names else "none",
                error=str(e),
            )

    def _get_next_model_index(self) -> int:
        """Get the next model index in circular fashion.

        Returns:
            Next model index (wraps around to 0 if at end)
        """
        total_models = len(LLMRegistry.get_agent_models(self._agent_name))
        return (self._current_model_index + 1) % total_models

    def _switch_to_next_model(self) -> bool:
        """Switch to the next model in the agent's chain (circular).

        Returns:
            True if successfully switched, False otherwise
        """
        try:
            next_index = self._get_next_model_index()
            next_model_entry = LLMRegistry.get_model_at_index(next_index, self._agent_name)

            logger.warning(
                "switching_to_next_model",
                agent_name=self._agent_name,
                from_index=self._current_model_index,
                to_index=next_index,
                to_model=next_model_entry["name"],
            )

            self._current_model_index = next_index
            self._llm = next_model_entry["llm"]

            logger.info("model_switched", new_model=next_model_entry["name"], new_index=next_index)
            return True
        except Exception as e:
            logger.error("model_switch_failed", error=str(e))
            return False

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, 3),
        reraise=True,
    )
    async def _call_llm_with_retry(self, messages: List[BaseMessage]) -> BaseMessage:
        """Call the LLM with automatic retry logic.

        Args:
            messages: List of messages to send to the LLM

        Returns:
            BaseMessage response from the LLM

        Raises:
            OpenAIError: If all retries fail
        """
        if not self._llm:
            raise RuntimeError("llm not initialized")

        try:
            response = await self._llm.ainvoke(messages)
            logger.debug("llm_call_successful", message_count=len(messages))
            return response
        except (RateLimitError, APITimeoutError, APIError) as e:
            logger.warning(
                "llm_call_failed_retrying",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            raise
        except OpenAIError as e:
            logger.error(
                "llm_call_failed",
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

    async def call(
        self,
        messages: List[BaseMessage],
        model_name: Optional[str] = None,
        **model_kwargs,
    ) -> BaseMessage:
        """Call the LLM with the specified messages and circular fallback.

        Args:
            messages: List of messages to send to the LLM
            model_name: Optional specific model to use. If None, uses current model.
            **model_kwargs: Optional kwargs to override default model configuration

        Returns:
            BaseMessage response from the LLM

        Raises:
            RuntimeError: If all models fail after retries
        """
        if model_name:
            try:
                self._llm = LLMRegistry.get(model_name, self._agent_name, **model_kwargs)
                all_names = LLMRegistry.get_all_names(self._agent_name)
                try:
                    self._current_model_index = all_names.index(model_name)
                except ValueError:
                    pass
                logger.info("using_requested_model", model_name=model_name, has_custom_kwargs=bool(model_kwargs))
            except ValueError as e:
                logger.error("requested_model_not_found", model_name=model_name, error=str(e))
                raise

        agent_models = LLMRegistry.get_agent_models(self._agent_name)
        total_models = len(agent_models)
        models_tried = 0
        starting_index = self._current_model_index
        last_error = None

        while models_tried < total_models:
            try:
                response = await self._call_llm_with_retry(messages)
                return response
            except OpenAIError as e:
                last_error = e
                models_tried += 1

                current_model_name = LLMRegistry.get_model_at_index(
                    self._current_model_index, self._agent_name
                )["name"]
                logger.error(
                    "llm_call_failed_after_retries",
                    agent_name=self._agent_name,
                    model=current_model_name,
                    models_tried=models_tried,
                    total_models=total_models,
                    error=str(e),
                )

                if models_tried >= total_models:
                    starting_model = LLMRegistry.get_model_at_index(
                        starting_index, self._agent_name
                    )["name"]
                    logger.error(
                        "all_models_failed",
                        agent_name=self._agent_name,
                        models_tried=models_tried,
                        starting_model=starting_model,
                    )
                    break

                if not self._switch_to_next_model():
                    logger.error("failed_to_switch_to_next_model")
                    break

        raise RuntimeError(
            f"failed to get response from llm after trying {models_tried} models "
            f"for agent '{self._agent_name}'. last error: {str(last_error)}"
        )

    def get_llm(self) -> Optional[BaseChatModel]:
        """Get the current LLM instance.

        Returns:
            Current BaseChatModel instance or None if not initialized
        """
        return self._llm

    def bind_tools(self, tools: List) -> "LLMService":
        """Bind tools to the current LLM.

        Args:
            tools: List of tools to bind

        Returns:
            Self for method chaining
        """
        if self._llm:
            self._llm = self._llm.bind_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self
