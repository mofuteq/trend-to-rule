"""High-level chat workflow orchestration shared by the Streamlit UI."""

import logging
from dataclasses import dataclass

from core.app_config import AppConfig
from core.models import StructuredClaims, StructuredDraft, UserNeeds
from core.query_utils import render_example_query
from core.text_utils import normalize_text_nfkc
from retrieval.app_retrieval import (
    points_to_prompt_context,
    points_to_table_rows,
    retrieve_vector_results_by_queries,
)
from services.chat import (
    extract_claims,
    extract_structured_draft,
    generate_decision_support,
    generate_query,
)
from services.image_search import ImageSearchResult, search_images

logger = logging.getLogger(__name__)


def _build_fallback_image_query(user_needs: UserNeeds, rule: str) -> str:
    """Build a safe fallback image query when structured rendering fails.

    Args:
        user_needs: Structured user-needs object.
        rule: Final synthesized assistant rule.

    Returns:
        str: A conservative image-search query string.
    """
    vertical = str(getattr(user_needs, "vertical", "") or "").strip().lower()
    if vertical == "mens":
        prefix = "menswear"
    elif vertical == "womens":
        prefix = "womenswear"
    elif vertical == "unisex":
        prefix = "unisex fashion"
    else:
        prefix = "fashion outfit"

    rule_hint = normalize_text_nfkc(rule or "").strip()
    if rule_hint:
        rule_hint = " ".join(rule_hint.split())
        return f"{prefix} {rule_hint}"
    return prefix


@dataclass(slots=True)
class RetrievalBundle:
    """Retrieved vector-search outputs used downstream by the chat workflow."""

    canonical_context: str
    emerging_context: str
    canonical_rows: list[dict[str, str]]
    emerging_rows: list[dict[str, str]]


@dataclass(slots=True)
class AssistantResponseBundle:
    """All derived assistant artifacts for a single user prompt."""

    structured_claims: StructuredClaims
    structured_draft: StructuredDraft
    rule: str
    image_query: str
    image_results: list[ImageSearchResult]


def retrieve_supporting_context(
    user_needs: UserNeeds,
    *,
    config: AppConfig,
) -> RetrievalBundle:
    """Retrieve canonical and emerging context for the current user needs.

    Args:
        user_needs: Structured user-needs object containing candidate queries.
        config: App runtime config.

    Returns:
        RetrievalBundle: Retrieved contexts and table rows for display.
    """
    candidate_queries = user_needs.candidate_queries
    retrieved = retrieve_vector_results_by_queries(
        canonical_query=str(candidate_queries.canonical_query or ""),
        emerging_query=str(candidate_queries.emerging_query or ""),
        user_vertical=user_needs.vertical,
        vector_candidate_k=config.vector_candidate_k,
        mmr_diversity=config.vector_mmr_diversity,
        per_query_top_k=config.vector_per_query_top_k,
    )
    canonical_points = retrieved.get("canonical", [])
    emerging_points = retrieved.get("emerging", [])
    return RetrievalBundle(
        canonical_context=points_to_prompt_context(canonical_points, label="Canonical"),
        emerging_context=points_to_prompt_context(emerging_points, label="Emerging"),
        canonical_rows=points_to_table_rows(canonical_points),
        emerging_rows=points_to_table_rows(emerging_points),
    )


def generate_assistant_response(
    user_prompt: str,
    *,
    user_needs: UserNeeds,
    retrieval: RetrievalBundle,
    config: AppConfig,
    last_user_goal: str | None,
    history: list[dict] | None,
) -> AssistantResponseBundle:
    """Generate assistant-side outputs from retrieval and user intent.

    Args:
        user_prompt: Raw user prompt.
        user_needs: Structured user-needs object.
        retrieval: Retrieved context bundle.
        config: App runtime config.
        last_user_goal: Previous inferred user goal.
        history: Prior chat history.

    Returns:
        AssistantResponseBundle: Claims, draft, final rule, and related images.
    """
    structured_claims = extract_claims(
        canonical_context=retrieval.canonical_context,
        emerging_context=retrieval.emerging_context,
        user_goal=user_needs.user_goal,
    )
    structured_draft = extract_structured_draft(
        canonical_claims=structured_claims.canonical_claims,
        emerging_claims=structured_claims.emerging_claims,
        user_goal=user_needs.user_goal,
    )
    decision_support = generate_decision_support(
        user_prompt=user_prompt,
        user_goal=user_needs.user_goal,
        last_user_goal=last_user_goal,
        history=history,
        structured_draft=structured_draft,
    )
    rule = normalize_text_nfkc(decision_support or "")
    query_spec = generate_query(
        user_goal=user_needs.user_goal,
        rule=rule,
    )
    try:
        image_query = render_example_query(query_spec)
    except ValueError as err:
        image_query = _build_fallback_image_query(
            user_needs=user_needs,
            rule=rule,
        )
        logger.warning(
            "render_example_query failed: %s; fallback_image_query=%s query_spec=%s",
            err,
            image_query,
            query_spec.model_dump(),
        )

    image_results: list[ImageSearchResult] = []
    try:
        image_results = search_images(
            image_query,
            limit=config.searxng_image_limit,
            base_url=config.searxng_base_url,
        )
    except Exception as err:
        logger.warning("Image search failed: %s", err)

    return AssistantResponseBundle(
        structured_claims=structured_claims,
        structured_draft=structured_draft,
        rule=rule,
        image_query=image_query,
        image_results=image_results,
    )
