"""Optional Langfuse tracing helpers.

Provides `observe`, `update_current_generation`, `update_current_trace`, and
`flush`, which become no-ops when Langfuse credentials are not configured.

Credentials are read from environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (default: https://cloud.langfuse.com)

This module only reads `os.environ`; loading `.env` is the responsibility of
the application entry points (`core.app_config`, `services.llm_client`, and
the pipeline scripts).
"""

import logging
import os
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_ENABLED: bool | None = None
_CLIENT: Any = None


def is_enabled() -> bool:
    """Return True when Langfuse public+secret keys are configured."""
    global _ENABLED
    if _ENABLED is None:
        public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        _ENABLED = bool(public and secret)
    return _ENABLED


def get_client() -> Any:
    """Return a cached Langfuse client or None when disabled/unavailable."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        _CLIENT = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("Langfuse tracing enabled host=%s", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
        return _CLIENT
    except Exception as err:
        logger.warning("Langfuse initialization failed: %s", err)
        return None


def observe(
    *,
    name: str | None = None,
    as_type: str | None = None,
) -> Callable[[F], F]:
    """Return the Langfuse `@observe` decorator, or a no-op when disabled."""
    if not is_enabled():
        def _passthrough(func: F) -> F:
            return func
        return _passthrough
    try:
        from langfuse import observe as _langfuse_observe  # type: ignore[import-not-found]
    except Exception as err:
        logger.warning("Langfuse observe import failed: %s", err)
        def _passthrough(func: F) -> F:
            return func
        return _passthrough

    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if as_type is not None:
        kwargs["as_type"] = as_type
    return _langfuse_observe(**kwargs)


def update_current_generation(**kwargs: Any) -> None:
    """Update the active generation span with model/usage/input/output info."""
    client = get_client()
    if client is None:
        return
    try:
        client.update_current_generation(**kwargs)
    except Exception as err:
        logger.debug("update_current_generation failed: %s", err)


def update_current_trace(**kwargs: Any) -> None:
    """Attach metadata (user_id, session_id, tags, ...) to the active trace."""
    client = get_client()
    if client is None:
        return
    try:
        client.update_current_trace(**kwargs)
    except Exception as err:
        logger.debug("update_current_trace failed: %s", err)


def flush() -> None:
    """Flush pending Langfuse events; safe when disabled."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as err:
        logger.debug("Langfuse flush failed: %s", err)


__all__ = [
    "flush",
    "get_client",
    "is_enabled",
    "observe",
    "update_current_generation",
    "update_current_trace",
]
