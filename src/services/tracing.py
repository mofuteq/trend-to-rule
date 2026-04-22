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
from contextlib import contextmanager
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
    """Return a cached Langfuse client or None when disabled/unavailable.

    Used only for flush(); trace/observation updates go through langfuse_context.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        _CLIENT = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=host,
        )
        logger.info("Langfuse tracing enabled host=%s", host)
        return _CLIENT
    except Exception as err:
        logger.warning("Langfuse initialization failed: %s", err)
        return None


def _get_context() -> Any:
    """Return a Langfuse client or None when disabled/unavailable."""
    if not is_enabled():
        return None
    return get_client()


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
    """Attach trace attributes and root-span details to the active trace."""
    client = get_client()
    if client is None:
        return

    # Langfuse v4 replaced langfuse_context.update_current_trace with a split model:
    # trace-level attributes are propagated via propagate_attributes(), while structured
    # input/output/metadata can still be attached to the current active span.
    propagation_kwargs = {
        "user_id": kwargs.get("user_id"),
        "session_id": kwargs.get("session_id"),
        "tags": kwargs.get("tags"),
        "trace_name": kwargs.get("name"),
    }
    propagation_kwargs = {
        key: value for key, value in propagation_kwargs.items() if value is not None
    }

    try:
        if propagation_kwargs:
            with propagate_attributes(**propagation_kwargs):
                pass
    except Exception as err:
        logger.debug("propagate_attributes failed: %s", err)

    span_kwargs = {
        "name": kwargs.get("name"),
        "input": kwargs.get("input"),
        "output": kwargs.get("output"),
        "metadata": kwargs.get("metadata"),
    }
    span_kwargs = {key: value for key, value in span_kwargs.items() if value is not None}
    if not span_kwargs:
        return
    try:
        client.update_current_span(**span_kwargs)
    except Exception as err:
        logger.debug("update_current_span failed: %s", err)


@contextmanager
def propagate_attributes(**kwargs: Any):
    """Propagate trace-level attributes with a no-op fallback when disabled."""
    if not is_enabled():
        yield
        return
    try:
        from langfuse import propagate_attributes as _propagate_attributes  # type: ignore[import-not-found]
    except Exception as err:
        logger.warning("Langfuse propagate_attributes import failed: %s", err)
        yield
        return

    with _propagate_attributes(**kwargs):
        yield


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
    "propagate_attributes",
    "update_current_generation",
    "update_current_trace",
]
