"""Low-level LLM client utilities for a provider-neutral backend.

The runtime uses Pydantic AI's `LiteLLMProvider` so any LiteLLM-supported
backend can be selected via a single `LLM_MODEL` string (e.g.
`openrouter/google/gemini-3-flash-preview`). LiteLLM is used in-process
through the provider layer; no LiteLLM Proxy/server is involved.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.litellm import LiteLLMProvider

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

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openrouter/google/gemini-3-flash-preview",
)


@tracing.observe(as_type="generation", name="llm_generation")
def create(
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    system_prompt: str | None = None,
    response_model: type[ModelT] | None = None,
    history: list[ModelMessage] | None = None,
    model: str = DEFAULT_LLM_MODEL,
) -> ModelT | SimpleNamespace | Any:
    """Send a message via Pydantic AI's LiteLLM provider.

    Args:
        user_prompt: User prompt text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        seed: Random seed for reproducibility.
        system_prompt: Optional system instruction.
        response_model: Pydantic model for structured output.
        history: Prior conversation messages in Pydantic AI format.
        model: LiteLLM-style model identifier (e.g. `openrouter/google/gemini-3-flash-preview`).

    Returns:
        Parsed Pydantic model, or SimpleNamespace with `.text` for plain text.
    """
    if not user_prompt:
        raise ValueError("user_prompt is required")

    resolved_key = DEFAULT_LLM_API_KEY
    if not resolved_key:
        raise ValueError("LLM_API_KEY is required")

    provider_kwargs: dict[str, Any] = {"api_key": resolved_key}
    if DEFAULT_LLM_BASE_URL:
        provider_kwargs["api_base"] = DEFAULT_LLM_BASE_URL
    provider = LiteLLMProvider(**provider_kwargs)
    pai_model = OpenAIChatModel(model, provider=provider)

    agent: Agent[None, Any] = Agent(
        model=pai_model,
        output_type=response_model or str,
        system_prompt=system_prompt or "",
    )

    model_settings: OpenAIChatModelSettings = {
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }

    run_result = agent.run_sync(
        user_prompt,
        message_history=history or None,
        model_settings=model_settings,
    )

    output = run_result.output
    logger.info(
        "llm_response model=%s base_url=%s response_model=%s",
        model,
        DEFAULT_LLM_BASE_URL or "<litellm-default>",
        response_model.__name__ if response_model else None,
    )

    trace_metadata: dict[str, Any] = {
        "model": model,
        "base_url": DEFAULT_LLM_BASE_URL,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "response_model": response_model.__name__ if response_model else None,
    }
    _update_llm_generation(
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


def _update_llm_generation(
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
