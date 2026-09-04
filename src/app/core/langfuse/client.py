"""Langfuse v4 tracer wrapper.

Agents and routes must use ``LangfuseTracer`` instead of importing the Langfuse SDK directly.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger


class LangfuseTracer:
    """Wraps Langfuse SDK calls with logging and safe no-ops when disabled."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.client: Optional[Langfuse] = None

        public_key = settings.LANGFUSE_PUBLIC_KEY
        secret_key = settings.LANGFUSE_SECRET_KEY
        if not public_key or not secret_key:
            logger.info("langfuse_disabled_missing_credentials")
            return

        try:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=settings.LANGFUSE_HOST,
            )
        except Exception:
            logger.warning("langfuse_client_initialization_failed", exc_info=True)
            self.client = None

    @contextmanager
    def _propagate_attributes(
        self,
        *,
        trace_name: str,
        user_id: Optional[Any] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        environment: Optional[str] = None,
    ) -> Generator[None, None, None]:
        if not self.client:
            yield
            return

        metadata_str = {key: str(value) for key, value in (metadata or {}).items()}
        propagate_kwargs: dict[str, Any] = {
            "trace_name": trace_name,
            "metadata": metadata_str,
            "tags": tags or [],
        }
        if user_id is not None:
            propagate_kwargs["user_id"] = str(user_id)
        if session_id is not None:
            propagate_kwargs["session_id"] = session_id
        if environment is not None:
            propagate_kwargs["environment"] = environment

        try:
            with propagate_attributes(**propagate_kwargs):
                yield
        except Exception:
            logger.warning("langfuse_propagate_attributes_failed", exc_info=True)
            yield

    @contextmanager
    def trace_agent_request(
        self,
        name: str,
        *,
        input_data: Optional[Any] = None,
        user_id: Optional[Any] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        environment: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """Open a root request observation with propagated trace attributes."""
        if not self.client:
            yield None
            return

        try:
            with self._propagate_attributes(
                trace_name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                tags=tags,
                environment=environment,
            ):
                with self.client.start_as_current_observation(
                    as_type="span",
                    name=name,
                    input=input_data,
                ) as observation:
                    yield observation
        except Exception:
            logger.warning("langfuse_trace_agent_request_failed", exc_info=True)
            yield None

    @contextmanager
    def trace_langgraph_agent(
        self,
        name: str,
        *,
        input_data: Optional[Any] = None,
        user_id: Optional[Any] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        environment: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """Alternate LangGraph wrapper with the same root observation semantics."""
        with self.trace_agent_request(
            name,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
            environment=environment,
        ) as observation:
            yield observation

    def get_callback_handler(self) -> Optional[CallbackHandler]:
        """Return a LangChain callback handler (must be used inside propagate scope)."""
        if not self.client:
            return None

        try:
            return CallbackHandler()
        except Exception:
            logger.warning("langfuse_callback_handler_creation_failed", exc_info=True)
            return None

    def create_span(
        self,
        trace: Any,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        if not self.client or trace is None:
            return None

        try:
            return trace.start_observation(
                as_type="span",
                name=name,
                input=input_data,
                metadata=metadata,
            )
        except Exception:
            logger.warning("langfuse_create_span_failed", exc_info=True)
            return None

    def end_span(
        self,
        span: Any,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if span is None:
            return

        try:
            span.update(output=output, metadata=metadata)
            span.end()
        except Exception:
            logger.warning("langfuse_end_span_failed", exc_info=True)

    def update_span(
        self,
        span: Any,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ) -> None:
        if span is None:
            return

        update_kwargs: dict[str, Any] = {}
        if output is not None:
            update_kwargs["output"] = output
        if metadata is not None:
            update_kwargs["metadata"] = metadata
        if level is not None:
            update_kwargs["level"] = level
        if status_message is not None:
            update_kwargs["status_message"] = status_message

        if not update_kwargs:
            return

        try:
            span.update(**update_kwargs)
        except Exception:
            logger.warning("langfuse_update_span_failed", exc_info=True)

    def get_trace_id(self, trace: Any = None) -> Optional[str]:
        if trace is not None:
            trace_id = getattr(trace, "trace_id", None)
            if trace_id:
                return str(trace_id)

        if not self.client:
            return None

        try:
            current_trace_id = self.client.get_current_trace_id()
            return str(current_trace_id) if current_trace_id else None
        except Exception:
            logger.warning("langfuse_get_trace_id_failed", exc_info=True)
            return None

    def submit_feedback(
        self,
        trace_id: str,
        score: float,
        name: str = "user-feedback",
        comment: Optional[str] = None,
    ) -> bool:
        if not self.client:
            return False

        try:
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment,
            )
            return True
        except Exception:
            logger.warning("langfuse_submit_feedback_failed", exc_info=True)
            return False

    def flush(self) -> None:
        if not self.client:
            return

        try:
            self.client.flush()
        except Exception:
            logger.warning("langfuse_flush_failed", exc_info=True)

    def shutdown(self) -> None:
        if not self.client:
            return

        try:
            self.client.flush()
            self.client.shutdown()
            logger.info("langfuse_shutdown_complete")
        except Exception:
            logger.warning("langfuse_shutdown_failed", exc_info=True)

    def start_generation(
        self,
        name: str,
        model: str,
        input_data: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        if not self.client:
            return None

        try:
            return self.client.start_observation(
                as_type="generation",
                name=name,
                model=model,
                input=input_data,
                metadata=metadata,
            )
        except Exception:
            logger.warning("langfuse_start_generation_failed", exc_info=True)
            return None

    def start_span(
        self,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        if not self.client:
            return None

        try:
            return self.client.start_observation(
                as_type="span",
                name=name,
                input=input_data,
                metadata=metadata,
            )
        except Exception:
            logger.warning("langfuse_start_span_failed", exc_info=True)
            return None
