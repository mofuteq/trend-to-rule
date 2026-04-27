"""Low-level LLM client utilities for Gemini and OpenAI-compatible backends."""

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence, TypeVar

from dotenv import load_dotenv
from google.genai.types import Content, ThinkingConfigDict
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from services import tracing

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
ModelName = Literal[
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
ENV_CANDIDATES = (
    SRC_ROOT / ".env",
    Path(".env"),
    PROJECT_ROOT / ".env",
)
for env_path in ENV_CANDIDATES:
    if env_path.is_file():
        load_dotenv(env_path)

DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TOP_P: float = 0.6
DEFAULT_SEED: int = 42

DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemma4:e4b")
DEFAULT_OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
REASONING_EFFORT_UNSUPPORTED_MODELS: set[str] = set()
GEMINI_THINKING_BUDGET_BY_REASONING_EFFORT: dict[str, int] = {
    "low": 512,
    "medium": 1024,
    "high": 2048,
}


def _is_resource_exhausted_error(err: Exception) -> bool:
    """Return True when the backend reports quota/resource exhaustion."""
    msg = str(err).lower()
    return (
        "resource_exhausted" in msg
        or "429" in msg
        or "quota" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
        or "too many requests" in msg
    )


def _is_reasoning_effort_unsupported_error(err: Exception) -> bool:
    """Return True when server rejects reasoning_effort/thinking options."""
    msg = str(err).lower()
    return (
        "reasoning_effort" in msg
        or "think value" in msg
        or "does not support thinking" in msg
        or "not support thinking" in msg
        or "not supported for this model" in msg
    )


def _content_to_pai_messages(history: list[Content]) -> list[ModelMessage]:
    """Convert Google Content history to Pydantic AI ModelMessage list."""
    messages: list[ModelMessage] = []
    for item in history:
        text_parts = [getattr(p, "text", None) for p in (item.parts or [])]
        text = "\n".join(t for t in text_parts if t).strip()
        if not text:
            continue
        if item.role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=text)]))
    return messages


def create(
    user_prompt: str,
    api_key: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    reasoning_effort: Literal["low", "medium",
                              "high"] = DEFAULT_OPENAI_REASONING_EFFORT,
    system_prompt: str | None = None,
    response_model: type[ModelT] | None = None,
    history: list[Content] | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
    fallback_models: Sequence[ModelName] = (
        "gemini-3-flash-preview",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite-preview",
    ),
) -> ModelT | SimpleNamespace | Any:
    """Send a message via Gemini or OpenAI-compatible backend using Pydantic AI.

    Args:
        user_prompt: User prompt text.
        api_key: API key for Gemini backend.
        openai_api_key: API key for OpenAI-compatible backend.
        openai_base_url: Base URL for OpenAI-compatible backend.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning level for backend-specific thinking controls.
        system_prompt: Optional system instruction.
        response_model: Pydantic model for structured output.
        history: Prior conversation contents.
        model: Model name.
        fallback_models: Retry models after the primary Gemini model fails.

    Returns:
        Parsed Pydantic model, or SimpleNamespace with `.text` for plain text.
    """
    if not user_prompt:
        raise ValueError("user_prompt is required")
    if "gemini" in model.lower():
        resolved_api_key = api_key or DEFAULT_API_KEY
        if not resolved_api_key:
            raise ValueError("api_key is required for gemini models")
        return _create_with_gemini(
            user_prompt=user_prompt,
            api_key=resolved_api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reasoning_effort=reasoning_effort,
            response_model=response_model,
            history=history,
            model=model,
            fallback_models=fallback_models,
        )
    return _create_with_openai(
        user_prompt=user_prompt,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
        response_model=response_model,
        history=history,
        model=model,
    )


@tracing.observe(as_type="generation", name="gemini_generation")
def _create_with_gemini(
    *,
    user_prompt: str,
    api_key: str,
    system_prompt: str | None,
    temperature: float,
    top_p: float,
    seed: int,
    reasoning_effort: Literal["low", "medium", "high"],
    response_model: type[ModelT] | None,
    history: list[Content] | None,
    model: str,
    fallback_models: Sequence[ModelName],
) -> ModelT | SimpleNamespace:
    """Create response using Pydantic AI Google backend."""
    thinking_budget = GEMINI_THINKING_BUDGET_BY_REASONING_EFFORT[reasoning_effort]
    model_settings: GoogleModelSettings = {
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "google_thinking_config": ThinkingConfigDict(
            thinking_budget=thinking_budget
        ),
    }

    pai_history = _content_to_pai_messages(history or [])
    model_candidates: list[str] = list(dict.fromkeys([model, *fallback_models]))

    run_result: AgentRunResult[Any] | None = None
    selected_model = model
    last_error: Exception | None = None

    for idx, candidate_model in enumerate(model_candidates):
        selected_model = candidate_model
        try:
            pai_model = GoogleModel(
                candidate_model,
                provider=GoogleProvider(api_key=api_key),
            )
            agent: Agent[None, Any] = Agent(
                model=pai_model,
                output_type=response_model or str,
                system_prompt=system_prompt or "",
            )
            run_result = agent.run_sync(
                user_prompt,
                message_history=pai_history or None,
                model_settings=model_settings,
            )
            break
        except Exception as err:
            last_error = err
            if idx >= len(model_candidates) - 1:
                raise
            logger.warning(
                "gemini_fallback from=%s to=%s reason=%s",
                selected_model,
                model_candidates[idx + 1],
                err,
            )
            if not _is_resource_exhausted_error(err):
                time.sleep(1)

    if run_result is None and last_error is not None:
        raise last_error

    output = run_result.output  # type: ignore[union-attr]
    logger.info(
        "llm_response backend=gemini model=%s response_model=%s",
        selected_model,
        response_model.__name__ if response_model is not None else None,
    )

    _update_gemini_generation(
        model=selected_model,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        run_result=run_result,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        reasoning_effort=reasoning_effort,
        response_model=response_model,
    )

    if response_model:
        return output
    return SimpleNamespace(text=str(output))


def _update_gemini_generation(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str | None,
    run_result: AgentRunResult[Any],
    temperature: float,
    top_p: float,
    seed: int,
    reasoning_effort: str,
    response_model: type[ModelT] | None,
) -> None:
    """Push input/output/usage info into the active Langfuse generation."""
    if not tracing.is_enabled():
        return
    output = run_result.output
    if response_model is not None and hasattr(output, "model_dump_json"):
        output_value: Any = output.model_dump_json()
    else:
        output_value = str(output)

    update_kwargs: dict[str, Any] = {
        "model": model,
        "input": {"user_prompt": user_prompt, "system_prompt": system_prompt},
        "output": output_value,
        "metadata": {
            "backend": "gemini",
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "response_model": response_model.__name__ if response_model else None,
        },
    }
    usage = run_result.usage()
    if usage.input_tokens or usage.output_tokens:
        update_kwargs["usage_details"] = {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.total_tokens,
        }
    tracing.update_current_generation(**update_kwargs)


@tracing.observe(as_type="generation", name="openai_generation")
def _create_with_openai(
    *,
    user_prompt: str,
    openai_api_key: str | None,
    openai_base_url: str | None,
    temperature: float,
    top_p: float,
    seed: int,
    reasoning_effort: str,
    system_prompt: str | None,
    response_model: type[ModelT] | None,
    history: list[Content] | None,
    model: str,
) -> ModelT | SimpleNamespace:
    """Create response using Pydantic AI OpenAI-compatible backend."""
    base_url = (openai_base_url or DEFAULT_OPENAI_BASE_URL).strip()
    if not base_url:
        raise ValueError("openai_base_url is required for non-gemini models")

    resolved_key = openai_api_key or DEFAULT_OPENAI_API_KEY or "dummy"
    provider = OpenAIProvider(base_url=base_url, api_key=resolved_key)
    pai_model = OpenAIModel(model, provider=provider)
    pai_history = _content_to_pai_messages(history or [])

    agent: Agent[None, Any] = Agent(
        model=pai_model,
        output_type=response_model or str,
        system_prompt=system_prompt or "",
    )

    base_settings: OpenAIModelSettings = {
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }
    settings_with_effort: OpenAIModelSettings = {
        **base_settings,
        "openai_reasoning_effort": reasoning_effort,
    }

    run_result: AgentRunResult[Any]
    if model in REASONING_EFFORT_UNSUPPORTED_MODELS:
        run_result = agent.run_sync(
            user_prompt,
            message_history=pai_history or None,
            model_settings=base_settings,
        )
    else:
        try:
            run_result = agent.run_sync(
                user_prompt,
                message_history=pai_history or None,
                model_settings=settings_with_effort,
            )
        except Exception as err:
            if not _is_reasoning_effort_unsupported_error(err):
                raise
            REASONING_EFFORT_UNSUPPORTED_MODELS.add(model)
            run_result = agent.run_sync(
                user_prompt,
                message_history=pai_history or None,
                model_settings=base_settings,
            )

    output = run_result.output
    logger.info(
        "llm_response backend=openai model=%s response_model=%s",
        model,
        response_model.__name__ if response_model else None,
    )

    trace_metadata: dict[str, Any] = {
        "backend": "openai",
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "response_model": response_model.__name__ if response_model else None,
    }
    _update_openai_generation(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        run_result=run_result,
        output=output,
        metadata=trace_metadata,
    )

    if response_model:
        return output
    return SimpleNamespace(text=str(output))


def _update_openai_generation(
    *,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    run_result: AgentRunResult[Any],
    output: Any,
    metadata: dict[str, Any],
) -> None:
    """Push input/output/usage info into the active Langfuse generation."""
    if not tracing.is_enabled():
        return
    if hasattr(output, "model_dump"):
        try:
            output_value: Any = output.model_dump()
        except Exception:
            output_value = str(output)
    else:
        output_value = str(output)

    update_kwargs: dict[str, Any] = {
        "model": model,
        "input": {"system_prompt": system_prompt, "user_prompt": user_prompt},
        "output": output_value,
        "metadata": metadata,
    }
    usage = run_result.usage()
    if usage.input_tokens or usage.output_tokens:
        update_kwargs["usage_details"] = {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.total_tokens,
        }
    tracing.update_current_generation(**update_kwargs)
