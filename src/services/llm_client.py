"""Low-level LLM client utilities for Gemini and OpenAI-compatible backends."""

import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    GenerateContentConfig,
    GenerateContentResponse,
    Part,
    ThinkingConfig,
)
from openai import OpenAI
from pydantic import BaseModel
from tenacity import Retrying, stop_after_attempt, wait_fixed

from core.text_utils import normalize_text_nfkc

try:
    import json5
except ImportError:
    json5 = None  # type: ignore[assignment]

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

DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemma3:12b")
DEFAULT_OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
REASONING_EFFORT_UNSUPPORTED_MODELS: set[str] = set()
GEMINI_THINKING_BUDGET_BY_REASONING_EFFORT: dict[str, int] = {
    "low": 512,
    "medium": 1024,
    "high": 2048,
}


def create(
    user_prompt: str,
    api_key: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    temperature: float = 0.2,
    top_p: float = 0.6,
    seed: int = 42,
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
) -> ModelT | GenerateContentResponse | Any:
    """Send a message via Gemini or OpenAI-compatible backend.

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
        Parsed Pydantic model, backend response object, or namespace with `.text`.
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
) -> ModelT | GenerateContentResponse:
    """Create response using Gemini SDK backend."""
    client = genai.Client(api_key=api_key)
    thinking_config = ThinkingConfig(
        thinking_budget=GEMINI_THINKING_BUDGET_BY_REASONING_EFFORT[reasoning_effort]
    )
    contents = [Content(role="user", parts=[Part(text=user_prompt)])]
    if response_model:
        config = GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            thinking_config=thinking_config,
        )
    else:
        config = GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            thinking_config=thinking_config,
        )

    model_candidates: list[str] = []
    for model_name in (model, *fallback_models):
        if model_name not in model_candidates:
            model_candidates.append(model_name)

    res: GenerateContentResponse
    selected_model = model
    for attempt in Retrying(
        stop=stop_after_attempt(len(model_candidates)),
        wait=wait_fixed(1),
        reraise=True,
    ):
        with attempt:
            idx = min(attempt.retry_state.attempt_number -
                      1, len(model_candidates) - 1)
            selected_model = model_candidates[idx]
            res = client.models.generate_content(
                model=selected_model,
                contents=(history + contents) if history else contents,
                config=config,
            )
    logger.info(
        "llm_response backend=gemini model=%s response_model=%s",
        selected_model,
        response_model.__name__ if response_model is not None else None,
    )

    if response_model:
        res = response_model.model_validate(res.parsed)
    return res


def _content_to_text(content: Content) -> str:
    """Flatten Google Content parts into plain text."""
    text_parts: list[str] = []
    for part in content.parts or []:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(str(text))
    return "\n".join(text_parts).strip()


def _validate_response_model_from_text(response_model: type[ModelT], text: str) -> ModelT:
    """Validate response model via JSON normalization pipeline."""
    normalized_text = normalize_text_nfkc(text or "").strip()
    fence = re.match(
        r"^\s*```(?:json)?\s*\n(?P<body>[\s\S]*?)\n```\s*$",
        normalized_text,
        flags=re.IGNORECASE,
    )
    if fence:
        normalized_text = fence.group("body").strip()

    if json5 is not None:
        normalized_obj = json5.loads(normalized_text)
    else:
        normalized_obj = json.loads(normalized_text)
    strict_json = json.dumps(normalized_obj, ensure_ascii=False)
    strict_obj = json.loads(strict_json)
    return response_model.model_validate(strict_obj)


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


def _chat_parse_with_reasoning_fallback(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[ModelT],
    temperature: float,
    top_p: float,
    seed: int,
    reasoning_effort: str,
) -> Any:
    """Call parse with reasoning_effort, retrying without it if unsupported."""
    if model in REASONING_EFFORT_UNSUPPORTED_MODELS:
        return client.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            response_format=response_model,
        )
    try:
        return client.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reasoning_effort=reasoning_effort,
            response_format=response_model,
        )
    except Exception as err:
        if not _is_reasoning_effort_unsupported_error(err):
            raise
        REASONING_EFFORT_UNSUPPORTED_MODELS.add(model)
        return client.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            response_format=response_model,
        )


def _chat_create_with_reasoning_fallback(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int,
    reasoning_effort: str,
) -> Any:
    """Call create with reasoning_effort, retrying without it if unsupported."""
    if model in REASONING_EFFORT_UNSUPPORTED_MODELS:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reasoning_effort=reasoning_effort,
        )
    except Exception as err:
        if not _is_reasoning_effort_unsupported_error(err):
            raise
        REASONING_EFFORT_UNSUPPORTED_MODELS.add(model)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )


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
) -> ModelT | Any:
    """Create response using OpenAI-compatible SDK backend."""
    base_url = (openai_base_url or DEFAULT_OPENAI_BASE_URL).strip()
    if not base_url:
        raise ValueError("openai_base_url is required for non-gemini models")

    resolved_key = openai_api_key or DEFAULT_OPENAI_API_KEY or "dummy"
    client = OpenAI(base_url=base_url, api_key=resolved_key)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for item in history or []:
        role = "assistant" if item.role == "model" else "user"
        text = _content_to_text(item)
        if text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_prompt})

    if response_model:
        json_retry_messages = list(messages)
        json_retry_messages.append(
            {
                "role": "user",
                "content": (
                    f"Return ONLY valid JSON that strictly matches the following JSON schema: {response_model.model_json_schema()}. "
                    "Do not include markdown, explanations, or code fences."
                ),
            }
        )
        try:
            parsed_resp = _chat_parse_with_reasoning_fallback(
                client,
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                reasoning_effort=reasoning_effort,
            )
            parsed = parsed_resp.choices[0].message.parsed
            if parsed is not None:
                logger.info(
                    "llm_response backend=openai model=%s response_model=%s parse_mode=parse",
                    model,
                    response_model.__name__,
                )
                return parsed
            retry_resp = _chat_create_with_reasoning_fallback(
                client,
                model=model,
                messages=json_retry_messages,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                reasoning_effort=reasoning_effort,
            )
            retry_text = str(
                retry_resp.choices[0].message.content or "").strip()
            if retry_text:
                logger.info(
                    "llm_response backend=openai model=%s response_model=%s parse_mode=json_retry",
                    model,
                    response_model.__name__,
                )
                return _validate_response_model_from_text(response_model, retry_text)
        except Exception:
            pass

    resp = _chat_create_with_reasoning_fallback(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )
    text = str((resp.choices[0].message.content or "")).strip()

    if response_model:
        try:
            logger.info(
                "llm_response backend=openai model=%s response_model=%s parse_mode=text_validate",
                model,
                response_model.__name__,
            )
            return _validate_response_model_from_text(response_model, text)
        except Exception:
            logger.info(
                "llm_response backend=openai model=%s response_model=%s parse_mode=json_loads",
                model,
                response_model.__name__,
            )
            return response_model.model_validate(json.loads(text))
    logger.info(
        "llm_response backend=openai model=%s response_model=%s",
        model,
        None,
    )
    return SimpleNamespace(text=text, raw=resp)
