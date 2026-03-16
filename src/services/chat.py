import os
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    GenerateContentResponse,
    Part,
    GenerateContentConfig,
    ThinkingConfig,
)
from openai import OpenAI
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_fixed
)
from pydantic import BaseModel
from core.models import (
    UserNeeds,
    UserGoal,
    ArticleAttribute,
    StructuredDraft,
    SearchQuery,
    StructuredClaims,
    Claim
)
from core.template_utils import get_j2_template
from core.text_utils import normalize_text_nfkc

try:
    import json5
except ImportError:
    json5 = None  # type: ignore[assignment]


ModelT = TypeVar("ModelT", bound=BaseModel)
ModelName = Literal[
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
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
DEFAULT_OPENAI_REASONING_EFFORT = os.getenv(
    "OPENAI_REASONING_EFFORT", "low")
REASONING_EFFORT_UNSUPPORTED_MODELS: set[str] = set()
GEMINI_THINKING_BUDGET_BY_REASONING_EFFORT: dict[str, int] = {
    "low": 128,
    "medium": 512,
    "high": 1024,
}

# Template
TEMPLATE_SEARCH_PATH = SRC_ROOT / "prompt_template"
TEMPLATE_USER_NEEDS = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="user_needs.j2"
)
TEMPLATE_INFER_ARTICLE = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="infer_attribute.j2"
)
TEMPLATE_STRUCTURED_CLIAMS = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="extract_claims.j2"
)
TEMPLATE_STRUCTURED_DRAFT = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="extract_structured_draft.j2"
)
TEMPLATE_SEARCH_QUERY = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="generate_search_query.j2"
)
TEMPLATE_DECISION_SUPPORT = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="generate_decision_support.j2"
)


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
        "gemini-2.5-flash-lite",
    ),
) -> ModelT | GenerateContentResponse | Any:
    """Send a message via Gemini or OpenAI-compatible backend.

    Args:
        user_prompt (str): User prompt text.
        api_key (str | None): API key for Gemini backend. If omitted, reads
            `GEMINI_API_KEY` from environment.
        openai_api_key (str | None): API key for OpenAI-compatible backend.
        openai_base_url (str | None): Base URL for OpenAI-compatible backend.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        seed (int): Random seed for reproducibility.
        reasoning_effort (str): OpenAI reasoning effort (`low|medium|high`).
        system_prompt (str | None): Optional system instruction.
        response_model (type[ModelT] | None): Pydantic model for structured output.
            If omitted, free-form response is returned.
        history (list[Content] | None): Prior conversation contents to prepend before the new user
            message.
        model (str): Model name. If it contains `gemini`, Gemini SDK is used.
            Otherwise OpenAI SDK is used.
        fallback_models (Sequence[ModelName]): Models to use for retry attempts after
            the primary model fails.

    Returns:
        ModelT | GenerateContentResponse | Any:
            Parsed model instance when `response_model` is provided and parsing succeeds.
            Otherwise backend raw response object (`GenerateContentResponse` for Gemini,
            OpenAI response wrapper for OpenAI-compatible backend).

    Raises:
        ValueError: If required key/prompt is empty for the selected backend.
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
    contents = [
        Content(
            role="user",
            parts=[Part(text=user_prompt)]
        )
    ]
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
    for attempt in Retrying(
        stop=stop_after_attempt(len(model_candidates)),
        wait=wait_fixed(1),
        reraise=True
    ):
        with attempt:
            idx = min(
                attempt.retry_state.attempt_number - 1,
                len(model_candidates) - 1
            )
            selected_model = model_candidates[idx]
            res = client.models.generate_content(
                model=selected_model,
                contents=(history + contents) if history else contents,
                config=config,
            )

    if response_model:
        res = response_model.model_validate(
            res.parsed
        )
    return res


def _content_to_text(content: Content) -> str:
    """Flatten google Content parts into plain text."""
    text_parts: list[str] = []
    for part in content.parts or []:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(str(text))
    return "\n".join(text_parts).strip()


def _validate_response_model_from_text(response_model: type[ModelT], text: str) -> ModelT:
    """Validate response model via robust JSON normalization pipeline.

    Steps:
        1. NFKC normalize raw text.
        2. Remove fenced JSON code block wrapper if present.
        3. Parse with JSON5 (or JSON fallback).
        4. Re-serialize to strict JSON and parse again.
        5. Validate with Pydantic model.
    """
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
    """Return True when server rejects reasoning_effort/think option."""
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
            return _validate_response_model_from_text(response_model, text)
        except Exception:
            return response_model.model_validate(json.loads(text))
    return SimpleNamespace(text=text, raw=resp)


def generate_search_query(
    user_prompt: str,
    user_goal: str,
    last_user_goal: str | None = None
) -> SearchQuery:
    """Generate canonical/emerging search queries from user input.

    Args:
        user_prompt (str): Raw user prompt.
        user_goal (str): Inferred current user goal.
        last_user_goal (str | None): Previous user goal for continuity.

    Returns:
        SearchQuery: Structured query set including canonical and emerging queries.
    """
    search_query = create(
        user_prompt=TEMPLATE_SEARCH_QUERY
        .module.user(
            user_prompt=user_prompt
        ),
        system_prompt=TEMPLATE_SEARCH_QUERY
        .module.system(
            user_goal=user_goal,
            last_user_goal=last_user_goal,
            now=datetime.now()
        ),
        model="gemini-2.5-flash",
        response_model=SearchQuery,
        reasoning_effort="low"
    )
    return search_query


def analyze_user_needs(
    user_prompt: str,
    api_key: str | None = None,
    last_user_goal: str | None = None,
) -> UserNeeds:
    """Analyze user intent and return normalized user needs.

    Args:
        user_prompt (str): Raw user input text.
        api_key (str | None): Gemini API key. Ignored when non-Gemini model is explicitly used.
            If omitted, environment value is used.
        last_user_goal (str | None): Previous inferred user goal for context.

    Returns:
        UserNeeds: Structured user-needs object inferred by the model.
    """
    user_goal = create(
        user_prompt=TEMPLATE_USER_NEEDS
        .module.user(
            user_prompt=user_prompt
        ),
        system_prompt=TEMPLATE_USER_NEEDS
        .module.system(
            last_user_goal=last_user_goal
        ),
        response_model=UserGoal,
        api_key=api_key,
        model="gemini-2.5-flash",
        reasoning_effort="medium"
    )
    search_query = generate_search_query(
        user_prompt=user_prompt,
        user_goal=user_goal.user_goal,
        last_user_goal=last_user_goal
    )
    return UserNeeds(
        user_goal=user_goal.user_goal,
        canditate_queries=search_query,
        reason=user_goal.reason
    )


def infer_article_attribute(
    article_text: str,
    api_key: str | None = None,
    model: str = DEFAULT_OPENAI_MODEL
) -> ArticleAttribute:
    """Infer structured attributes from article text.

    Args:
        article_text (str): Article text used for attribute inference.
        api_key (str | None): Gemini API key when Gemini model is selected.
            If omitted, environment value is used.
        model (str): Model name for inference. Defaults to `DEFAULT_OPENAI_MODEL`.

    Returns:
        ArticleAttribute: Structured attributes inferred from the article.
    """
    res = create(
        user_prompt=TEMPLATE_INFER_ARTICLE
        .module.user(
            article_text=article_text
        ),
        response_model=ArticleAttribute,
        api_key=api_key,
        model=model,
        temperature=0.0,
        top_p=0.6,
        seed=42,
        reasoning_effort="low"
    )
    return res


def extract_claims(
    canonical_context: str,
    emerging_context: str,
    user_goal: str,
    last_user_goal: str | None = None,
    history: list[Content] | None = None,
) -> StructuredClaims:
    """Extract structured claims from canonical and emerging contexts.

    Args:
        canonical_context (str): Retrieved canonical context text.
        emerging_context (str): Retrieved emerging context text.
        user_goal (str): Current inferred user goal.
        last_user_goal (str | None): Previous inferred user goal for continuity.
        history (list[Content] | None): Prior chat history.

    Returns:
        StructuredClaims: Structured claims generated from the provided contexts.
    """
    res = create(
        user_prompt=TEMPLATE_STRUCTURED_CLIAMS
        .module.system(
            user_goal=user_goal,
            last_user_goal=last_user_goal,
            now=datetime.now()
        ),
        system_prompt=TEMPLATE_STRUCTURED_CLIAMS
        .module.user(
            canonical_context=canonical_context,
            emerging_context=emerging_context
        ),
        history=history,
        response_model=StructuredClaims,
        model="gemini-2.5-flash",
        reasoning_effort="low"
    )
    return res


def extract_structured_draft(
    canonical_claims: list[Claim],
    emerging_claims: list[Claim],
    user_goal: str,
    api_key: str | None = None,
    history: list[Content] | None = None,
    last_user_goal: str | None = None
) -> StructuredDraft:
    """Generate structured draft from user prompt and retrieved context.

    Args:
        canonical_context (str): Retrieved canonical context text.
        emerging_context (str): Retrieved emerging context text.
        user_goal (str): Current inferred user goal.
        api_key (str | None): Gemini API key.
        history (list[Content] | None): Prior chat history.
        last_user_goal (str | None): Previous inferred user goal.

    Returns:
        StructuredDraft: Structured draft extracted from prompt and context.
    """
    res = create(
        user_prompt=TEMPLATE_STRUCTURED_DRAFT
        .module.user(
            canonical_claims="\n".join(
                claim.model_dump_json()
                for claim in canonical_claims
            ),
            emerging_claims="\n".join(
                claim.model_dump_json()
                for claim in emerging_claims
            )
        ),
        system_prompt=TEMPLATE_STRUCTURED_DRAFT
        .module.system(
            user_goal=user_goal,
            last_user_goal=last_user_goal,
            now=datetime.now()
        ),
        history=history,
        api_key=api_key,
        model="gemini-2.5-flash",
        response_model=StructuredDraft,
        reasoning_effort="medium"
    )
    return res


def generate_decision_support(
    user_prompt: str,
    user_goal: str,
    structured_draft: StructuredDraft,
    history: list[dict] | None = None,
    last_user_goal: str | None = None,
    api_key: str | None = None,
) -> str:
    """Generate decision-support output from the structured draft.

    Args:
        user_prompt (str): Original user prompt.
        user_goal (str): Current inferred user goal.
        structured_draft (StructuredDraft): Structured draft generated from retrieval context.
        history (list[dict] | None): Prior chat history to preserve conversational context.
        last_user_goal (str | None): Previous inferred user goal.
        api_key (str | None): Gemini API key when Gemini backend is used.

    Returns:
        str: Decision-support result returned by the model.
    """
    res = create(
        user_prompt=TEMPLATE_DECISION_SUPPORT
        .module.user(
            user_prompt=user_prompt,
            structured_draft=structured_draft.model_dump_json()
        ),
        system_prompt=TEMPLATE_DECISION_SUPPORT
        .module.system(
            user_goal=user_goal,
            last_user_goal=last_user_goal,
            now=datetime.now()
        ),
        history=history,
        api_key=api_key,
        model="gemini-2.5-flash",
    )
    return res.text
