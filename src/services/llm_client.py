"""Low-level LLM client utilities for the OpenRouter backend."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TypeVar, cast

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from services import tracing

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

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

DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-3-flash-preview",
)
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


def _get_reasoning_effort() -> ReasoningEffort:
    value = os.getenv("OPENROUTER_REASONING_EFFORT", "low")
    if value not in {"minimal", "low", "medium", "high"}:
        raise ValueError(f"Invalid OPENROUTER_REASONING_EFFORT: {value}")
    return cast(ReasoningEffort, value)


DEFAULT_OPENROUTER_REASONING_EFFORT = _get_reasoning_effort()


@tracing.observe(as_type="generation", name="openrouter_generation")
def create(
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    reasoning_effort: ReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT,
    system_prompt: str | None = None,
    response_model: type[ModelT] | None = None,
    history: list[ModelMessage] | None = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
) -> ModelT | SimpleNamespace | Any:
    """Send a message via the Pydantic AI OpenRouter provider.

    Args:
        user_prompt: User prompt text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning level for backend-specific thinking controls.
        system_prompt: Optional system instruction.
        response_model: Pydantic model for structured output.
        history: Prior conversation messages in Pydantic AI format.
        model: Model name.

    Returns:
        Parsed Pydantic model, or SimpleNamespace with `.text` for plain text.
    """
    if not user_prompt:
        raise ValueError("user_prompt is required")

    resolved_key = DEFAULT_OPENROUTER_API_KEY
    if not resolved_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    provider = OpenRouterProvider(api_key=resolved_key)
    pai_model = OpenRouterModel(model, provider=provider)

    agent: Agent[None, Any] = Agent(
        model=pai_model,
        output_type=response_model or str,
        system_prompt=system_prompt or "",
    )

    model_settings: OpenRouterModelSettings = {
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "openrouter_reasoning": {"effort": reasoning_effort},
    }

    run_result = agent.run_sync(
        user_prompt,
        message_history=history or None,
        model_settings=model_settings,
    )

    output = run_result.output
    logger.info(
        "llm_response backend=openrouter model=%s response_model=%s",
        model,
        response_model.__name__ if response_model else None,
    )

    trace_metadata: dict[str, Any] = {
        "backend": "openrouter",
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "response_model": response_model.__name__ if response_model else None,
    }
    _update_openrouter_generation(
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


def _update_openrouter_generation(
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
