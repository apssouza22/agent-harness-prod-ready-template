"""LangGraph checkpoint persistence and session cleanup."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.app.core.checkpoint.models import (
    CheckpointDetail,
    CheckpointSummary,
    StateHistoryEntry,
    serialize_checkpoint_detail,
    serialize_checkpoint_summary,
    serialize_state_history_entry,
)
from src.app.core.common.config import Environment, Settings, settings as default_settings
from src.app.core.common.logging import logger
from src.app.core.graph.compiled import StateGraphCompiled


class CheckpointService:
    """Service for LangGraph checkpoint persistence and session cleanup.

    Encapsulates AsyncPostgresSaver initialization and checkpoint read/write
    operations for a given thread/session id.
    """

    def __init__(
        self,
        app_settings: Settings,
        connection_pool: AsyncConnectionPool | None,
    ) -> None:
        self._settings = app_settings
        self._connection_pool = connection_pool
        self._checkpointer: AsyncPostgresSaver | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def connection_pool(self) -> AsyncConnectionPool | None:
        return self._connection_pool

    @staticmethod
    def _thread_config(session_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
        """Build a RunnableConfig for a session/thread id."""
        configurable: dict[str, str] = {"thread_id": session_id}
        if checkpoint_id is not None:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    async def get_checkpointer(self) -> AsyncPostgresSaver | None:
        """Return the LangGraph checkpointer, initializing it on first access."""
        if self._checkpointer is not None:
            return self._checkpointer

        if self._connection_pool is None:
            if self._settings.ENVIRONMENT != Environment.PRODUCTION:
                raise RuntimeError("connection pool initialization failed")
            return None

        self._checkpointer = AsyncPostgresSaver(self._connection_pool)
        await self._checkpointer.setup()
        logger.info(
            "checkpointer_initialized",
            environment=self._settings.ENVIRONMENT.value,
        )
        return self._checkpointer

    async def _require_checkpointer(self) -> AsyncPostgresSaver:
        """Return an initialized checkpointer or raise when unavailable."""
        checkpointer = await self.get_checkpointer()
        if checkpointer is None:
            logger.error("checkpointer_unavailable")
            raise RuntimeError("checkpointer unavailable")
        return checkpointer

    async def list_checkpoints(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        before: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[CheckpointSummary]:
        """List checkpoint summaries for a session/thread id.

        Args:
            session_id: LangGraph thread id tied to the user session.
            limit: Maximum number of checkpoints to return.
            before: Return checkpoints created before this checkpoint id.
            filter: Optional metadata filter passed to the checkpointer.

        Returns:
            list[CheckpointSummary]: Checkpoint summaries ordered newest first.
        """
        checkpointer = await self._require_checkpointer()
        config = self._thread_config(session_id)
        before_config = self._thread_config(session_id, before) if before else None

        summaries: list[CheckpointSummary] = []
        async for checkpoint_tuple in checkpointer.alist(
            config,
            filter=filter,
            before=before_config,
            limit=limit,
        ):
            summaries.append(serialize_checkpoint_summary(checkpoint_tuple))

        logger.info(
            "checkpoints_listed",
            session_id=session_id,
            checkpoint_count=len(summaries),
        )
        return summaries

    async def get_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> CheckpointDetail | None:
        """Get detailed checkpoint metadata for a session/thread id.

        Args:
            session_id: LangGraph thread id tied to the user session.
            checkpoint_id: Specific checkpoint id. Uses latest when omitted.

        Returns:
            CheckpointDetail | None: Checkpoint detail when found.
        """
        checkpointer = await self._require_checkpointer()
        config = self._thread_config(session_id, checkpoint_id)
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            logger.info(
                "checkpoint_not_found",
                session_id=session_id,
                checkpoint_id=checkpoint_id,
            )
            return None

        detail = serialize_checkpoint_detail(checkpoint_tuple)
        logger.info(
            "checkpoint_retrieved",
            session_id=session_id,
            checkpoint_id=detail.checkpoint_id,
        )
        return detail

    async def get_state_history(
        self,
        session_id: str,
        graph: StateGraphCompiled,
        *,
        limit: int | None = None,
        before: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[StateHistoryEntry]:
        """Get graph state history for a session/thread id.

        Args:
            session_id: LangGraph thread id tied to the user session.
            graph: Compiled graph used to interpret checkpoint state.
            limit: Maximum number of snapshots to return.
            before: Return snapshots created before this checkpoint id.
            filter: Optional metadata filter passed to the graph.

        Returns:
            list[StateHistoryEntry]: State snapshots ordered newest first.
        """
        config = self._thread_config(session_id)
        before_config = self._thread_config(session_id, before) if before else None
        snapshots = await graph.aget_state_history(
            config,
            filter=filter,
            before=before_config,
            limit=limit,
        )
        history = [serialize_state_history_entry(snapshot) for snapshot in snapshots]
        logger.info(
            "state_history_retrieved",
            session_id=session_id,
            snapshot_count=len(history),
        )
        return history

    async def clear_session(self, session_id: str) -> None:
        """Clear all checkpoint rows for a session/thread id.

        Args:
            session_id: LangGraph thread id tied to the user session.

        Raises:
            RuntimeError: When the checkpointer is unavailable.
            Exception: When checkpoint deletion fails.
        """
        checkpointer = await self._require_checkpointer()
        try:
            await checkpointer.adelete_thread(session_id)
            logger.info("session_checkpoints_cleared", session_id=session_id)
        except Exception:
            logger.exception("failed_to_clear_chat_history", session_id=session_id)
            raise

    async def copy_session(self, source_session_id: str, target_session_id: str) -> None:
        """Copy checkpoint data from one session/thread id to another.

        Args:
            source_session_id: Source LangGraph thread id.
            target_session_id: Target LangGraph thread id.

        Raises:
            RuntimeError: When the checkpointer is unavailable.
            Exception: When the copy operation fails.
        """
        checkpointer = await self._require_checkpointer()
        try:
            await checkpointer.acopy_thread(source_session_id, target_session_id)
            logger.info(
                "session_checkpoints_copied",
                source_session_id=source_session_id,
                target_session_id=target_session_id,
            )
        except Exception:
            logger.exception(
                "session_checkpoints_copy_failed",
                source_session_id=source_session_id,
                target_session_id=target_session_id,
            )
            raise

    async def prune_sessions(self, session_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Prune checkpoint data for one or more session/thread ids.

        Args:
            session_ids: LangGraph thread ids to prune.
            strategy: Prune strategy supported by LangGraph (default keep_latest).

        Raises:
            RuntimeError: When the checkpointer is unavailable.
            Exception: When pruning fails.
        """
        checkpointer = await self._require_checkpointer()
        try:
            await checkpointer.aprune(list(session_ids), strategy=strategy)
            logger.info(
                "session_checkpoints_pruned",
                session_count=len(session_ids),
                strategy=strategy,
            )
        except Exception:
            logger.exception(
                "session_checkpoints_prune_failed",
                session_count=len(session_ids),
                strategy=strategy,
            )
            raise
