"""Domain-level chat service functions built on top of the LLM client."""

from datetime import datetime
from typing import Literal

from pydantic_ai.messages import ModelMessage

from core.models import (
    RequestAnalysis,
    RequestGoal,
    ArticleAttribute,
    StructuredDraft,
    SearchQuery,
    StructuredClaims,
    Claim,
    ExampleQuerySpec
)
from services import tracing
from services.llm_client import create
from services.prompt_service import (
    TEMPLATE_CHAT_TITLE,
    TEMPLATE_DECISION_SUPPORT,
    TEMPLATE_IMAGE_QUERY,
    TEMPLATE_INFER_ARTICLE,
    TEMPLATE_REQUEST_ANALYSIS,
    TEMPLATE_SEARCH_QUERY,
    TEMPLATE_STRUCTURED_CLAIMS,
    TEMPLATE_STRUCTURED_DRAFT,
)

ATTRIBUTE_INFERENCE_MODEL = "google/gemini-2.5-flash-lite"


@tracing.observe(name="generate_search_query")
def generate_search_query(
    user_prompt: str,
    request_goal: str,
    last_request_goal: str | None = None
) -> SearchQuery:
    """Generate canonical/emerging search queries from user input.

    Args:
        user_prompt (str): Raw user prompt.
        request_goal (str): Inferred current request goal.
        last_request_goal (str | None): Previous request goal for continuity.

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
            request_goal=request_goal,
            last_request_goal=last_request_goal,
            now=datetime.now()
        ),
        response_model=SearchQuery,
    )
    return search_query


@tracing.observe(name="infer_attribute")
def infer_attribute(
    input_text: str,
    task: Literal["article", "user_prompt"] = "article",
    model: str = ATTRIBUTE_INFERENCE_MODEL
) -> ArticleAttribute:
    """Infer structured attributes from article text.

    Args:
        input_text (str): Source text used for attribute inference.
        task (Literal["article", "user_prompt"]): Prompt mode that controls whether
            the input text should be interpreted as article content or a user prompt.
        model (str): Model name for inference. Defaults to `ATTRIBUTE_INFERENCE_MODEL`.

    Returns:
        ArticleAttribute: Structured attributes inferred from the supplied text.
    """
    res = create(
        user_prompt=TEMPLATE_INFER_ARTICLE
        .module.user(
            input_text=input_text,
            task=task
        ),
        response_model=ArticleAttribute,
        model=model,
        temperature=0.0,
        top_p=0.6,
        seed=42,
    )
    return res


@tracing.observe(name="analyze_request")
def analyze_request(
    user_prompt: str,
    last_request_goal: str | None = None,
    history: list[ModelMessage] | None = None,
) -> RequestAnalysis:
    """Analyze the user request for intent, retrieval planning, and scope gating.

    Args:
        user_prompt (str): Raw user input text.
        last_request_goal (str | None): Previous inferred request goal for context.
        history (list[ModelMessage] | None): Prior conversation history for multi-turn context.

    Returns:
        RequestAnalysis: Structured request analysis inferred by the model.
    """
    request_goal = create(
        user_prompt=TEMPLATE_REQUEST_ANALYSIS
        .module.user(
            user_prompt=user_prompt
        ),
        system_prompt=TEMPLATE_REQUEST_ANALYSIS
        .module.system(
            last_request_goal=last_request_goal
        ),
        response_model=RequestGoal,
        history=history,
    )
    search_query = generate_search_query(
        user_prompt=user_prompt,
        request_goal=request_goal.request_goal,
        last_request_goal=last_request_goal,
    )
    vertical = infer_attribute(
        input_text=user_prompt,
        task="user_prompt",
    ).vertical
    return RequestAnalysis(
        request_goal=request_goal.request_goal,
        candidate_queries=search_query,
        vertical=vertical,
        is_in_scope=request_goal.is_in_scope,
    )


@tracing.observe(name="extract_claims")
def extract_claims(
    canonical_context: str,
    emerging_context: str,
    request_goal: str,
) -> StructuredClaims:
    """Extract structured claims from canonical and emerging contexts.

    Args:
        canonical_context (str): Retrieved canonical context text.
        emerging_context (str): Retrieved emerging context text.
        request_goal (str): Current inferred request goal.

    Returns:
        StructuredClaims: Structured claims generated from the provided contexts.
    """
    res = create(
        user_prompt=TEMPLATE_STRUCTURED_CLAIMS
        .module.system(
            request_goal=request_goal,
            now=datetime.now()
        ),
        system_prompt=TEMPLATE_STRUCTURED_CLAIMS
        .module.user(
            canonical_context=canonical_context,
            emerging_context=emerging_context
        ),
        response_model=StructuredClaims,
    )
    return res


__all__ = [
    "analyze_request",
    "extract_claims",
    "extract_structured_draft",
    "generate_chat_title",
    "generate_decision_support",
    "generate_query",
    "generate_search_query",
    "infer_attribute",
]


@tracing.observe(name="extract_structured_draft")
def extract_structured_draft(
    canonical_claims: list[Claim],
    emerging_claims: list[Claim],
    request_goal: str,
) -> StructuredDraft:
    """Generate structured draft from user prompt and retrieved context.

    Args:
        canonical_claims (list[Claim]): Canonical claims extracted from retrieved context.
        emerging_claims (list[Claim]): Emerging claims extracted from retrieved context.
        request_goal (str): Current inferred request goal.

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
            request_goal=request_goal,
            now=datetime.now()
        ),
        response_model=StructuredDraft,
    )
    return res


@tracing.observe(name="generate_decision_support")
def generate_decision_support(
    user_prompt: str,
    request_goal: str,
    structured_draft: StructuredDraft,
    history: list[ModelMessage] | None = None,
    last_request_goal: str | None = None,
) -> str:
    """Generate decision-support output from the structured draft.

    Args:
        user_prompt (str): Original user prompt.
        request_goal (str): Current inferred request goal.
        structured_draft (StructuredDraft): Structured draft generated from retrieval context.
        history (list[ModelMessage] | None): Prior chat history to preserve conversational context.
        last_request_goal (str | None): Previous inferred request goal.

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
            request_goal=request_goal,
            last_request_goal=last_request_goal,
            now=datetime.now()
        ),
        history=history,
    )
    return res.text


@tracing.observe(name="generate_query")
def generate_query(
    request_goal: str,
    rule: str,
) -> ExampleQuerySpec:
    """Generate example query specifications from a synthesized rule.

    Args:
        request_goal (str): Current inferred request goal.
        rule (str): Synthesized rule text used as the query-generation source.

    Returns:
        ExampleQuerySpec: Example query specification generated from the rule.
    """
    res = create(
        user_prompt=TEMPLATE_IMAGE_QUERY
        .module.user(
            rule=rule
        ),
        system_prompt=TEMPLATE_IMAGE_QUERY
        .module.system(
            request_goal=request_goal
        ),
        response_model=ExampleQuerySpec,
    )
    return res


@tracing.observe(name="generate_chat_title")
def generate_chat_title(
    messages: list[dict[str, str]],
) -> str:
    """Generate a short chat title from chat history.

    Args:
        messages: Chat messages in chronological order.

    Returns:
        str: Short chat title suitable for sidebar history display.
    """
    conversation_lines: list[str] = []
    for message in messages[-8:]:
        role = str(message.get("role") or "").strip() or "user"
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        conversation_lines.append(f"{role}: {content}")

    prompt = "\n".join(conversation_lines).strip()
    if not prompt:
        return "Untitled chat"

    res = create(
        user_prompt=TEMPLATE_CHAT_TITLE
        .module.user(
            conversation_history=prompt
        ),
        system_prompt=TEMPLATE_CHAT_TITLE
        .module.system(),
        temperature=0.2,
        top_p=0.8,
    )
    title = str(res.text or "").strip()
    return title or "Untitled chat"
