"""Shared prompt template loading for service-layer LLM workflows."""

from pathlib import Path

from core.template_utils import get_j2_template

SRC_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SEARCH_PATH = SRC_ROOT / "prompt_template"

TEMPLATE_USER_NEEDS = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="user_needs.j2",
)
TEMPLATE_INFER_ARTICLE = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="infer_attribute.j2",
)
TEMPLATE_STRUCTURED_CLAIMS = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="extract_claims.j2",
)
TEMPLATE_STRUCTURED_DRAFT = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="extract_structured_draft.j2",
)
TEMPLATE_SEARCH_QUERY = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="generate_search_query.j2",
)
TEMPLATE_IMAGE_QUERY = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="generate_image_query.j2",
)
TEMPLATE_DECISION_SUPPORT = get_j2_template(
    searchpath=TEMPLATE_SEARCH_PATH,
    name="generate_decision_support.j2",
)

__all__ = [
    "TEMPLATE_DECISION_SUPPORT",
    "TEMPLATE_IMAGE_QUERY",
    "TEMPLATE_INFER_ARTICLE",
    "TEMPLATE_SEARCH_QUERY",
    "TEMPLATE_STRUCTURED_CLAIMS",
    "TEMPLATE_STRUCTURED_DRAFT",
    "TEMPLATE_USER_NEEDS",
]
