"""High-level chat workflow orchestration shared by the Streamlit UI."""

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import ConnectionPool

from core.app_config import AppConfig
from core.models import ExampleQuerySpec, StructuredClaims, StructuredDraft, UserNeeds
from core.query_utils import render_example_query
from core.text_utils import normalize_text_nfkc
from retrieval.app_retrieval import (
    points_to_prompt_context,
    points_to_table_rows,
    retrieve_vector_results_by_queries,
)
from services import tracing
from services.chat import (
    extract_claims,
    extract_structured_draft,
    generate_decision_support,
    generate_query,
)
from services.image_search import ImageSearchResult, search_and_rerank_images

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


@tracing.observe(name="retrieve_supporting_context")
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


class AssistantResponseState(TypedDict, total=False):
    """LangGraph state for the linear assistant-response pipeline.

    `total=False` is intentional: each node progressively adds its own fields
    to the state as the graph advances, so most keys are absent at earlier
    steps. The required input fields (`user_prompt`, `user_needs`,
    `retrieval`, `config`, `last_user_goal`, `history`) are always supplied
    in the initial state passed to `invoke()`.
    """

    user_prompt: str
    user_needs: UserNeeds
    retrieval: RetrievalBundle
    config: AppConfig
    last_user_goal: str | None
    history: list[dict] | None
    structured_claims: StructuredClaims
    structured_draft: StructuredDraft
    rule: str
    query_spec: ExampleQuerySpec
    image_query: str
    image_results: list[ImageSearchResult]


def _node_extract_claims(state: AssistantResponseState) -> dict:
    structured_claims = extract_claims(
        canonical_context=state["retrieval"].canonical_context,
        emerging_context=state["retrieval"].emerging_context,
        user_goal=state["user_needs"].user_goal,
    )
    return {"structured_claims": structured_claims}


def _node_extract_structured_draft(state: AssistantResponseState) -> dict:
    structured_draft = extract_structured_draft(
        canonical_claims=state["structured_claims"].canonical_claims,
        emerging_claims=state["structured_claims"].emerging_claims,
        user_goal=state["user_needs"].user_goal,
    )
    return {"structured_draft": structured_draft}


def _node_generate_decision_support(state: AssistantResponseState) -> dict:
    decision_support = generate_decision_support(
        user_prompt=state["user_prompt"],
        user_goal=state["user_needs"].user_goal,
        last_user_goal=state.get("last_user_goal"),
        history=state.get("history"),
        structured_draft=state["structured_draft"],
    )
    rule = normalize_text_nfkc(decision_support or "")
    return {"rule": rule}


def _node_generate_query(state: AssistantResponseState) -> dict:
    query_spec = generate_query(
        user_goal=state["user_needs"].user_goal,
        rule=state["rule"],
    )
    return {"query_spec": query_spec}


def _node_render_image_query(state: AssistantResponseState) -> dict:
    query_spec = state["query_spec"]
    try:
        image_query = render_example_query(query_spec)
    except ValueError as err:
        image_query = _build_fallback_image_query(
            user_needs=state["user_needs"],
            rule=state["rule"],
        )
        logger.warning(
            "render_example_query failed: %s; fallback_image_query=%s query_spec=%s",
            err,
            image_query,
            query_spec.model_dump(),
        )
    return {"image_query": image_query}


def _node_search_images(state: AssistantResponseState) -> dict:
    config = state["config"]
    image_query = state["image_query"]
    image_results: list[ImageSearchResult] = []
    try:
        logger.info(
            "image_search_start query=%r fetch_limit=%s final_limit=%s",
            image_query,
            config.searxng_image_fetch_limit,
            config.searxng_image_limit,
        )
        image_results = search_and_rerank_images(
            image_query,
            base_url=config.searxng_base_url,
            fetch_limit=config.searxng_image_fetch_limit,
            rerank_limit=config.searxng_image_limit,
        )
    except Exception as err:
        logger.warning("Image search failed: %s", err)
    return {"image_results": image_results}


def _build_assistant_response_graph_builder() -> StateGraph:
    """Build a linear-transition state machine for the assistant response.

    This graph is intentionally linear and preserves the existing Fixed RAR
    behavior: every turn runs claims -> draft -> decision support -> query ->
    image query -> image search in a fixed order. Conditional routing,
    sufficiency checks, and re-retrieval loops are deliberately out of scope
    here and are planned as future Agentic RAR work on top of this graph.
    """
    graph = StateGraph(AssistantResponseState)
    graph.add_node("extract_claims", _node_extract_claims)
    graph.add_node("extract_structured_draft", _node_extract_structured_draft)
    graph.add_node("generate_decision_support", _node_generate_decision_support)
    graph.add_node("generate_query", _node_generate_query)
    graph.add_node("render_image_query", _node_render_image_query)
    graph.add_node("search_images", _node_search_images)

    graph.add_edge(START, "extract_claims")
    graph.add_edge("extract_claims", "extract_structured_draft")
    graph.add_edge("extract_structured_draft", "generate_decision_support")
    graph.add_edge("generate_decision_support", "generate_query")
    graph.add_edge("generate_query", "render_image_query")
    graph.add_edge("render_image_query", "search_images")
    graph.add_edge("search_images", END)
    return graph


_ASSISTANT_RESPONSE_GRAPH_BUILDER = _build_assistant_response_graph_builder()
_GRAPH_LOCK = threading.Lock()
_COMPILED_GRAPH = None
_CHECKPOINTER_POOL: ConnectionPool | None = None


def _build_checkpointer() -> PostgresSaver | None:
    """Create a PostgresSaver from ``LANGGRAPH_POSTGRES_URL`` if configured.

    Returns None when the env var is unset so local runs without the postgres
    container still work; in that case the graph compiles without persistence.
    """
    global _CHECKPOINTER_POOL
    url = os.getenv("LANGGRAPH_POSTGRES_URL")
    if not url:
        return None
    _CHECKPOINTER_POOL = ConnectionPool(
        conninfo=url,
        max_size=int(os.getenv("LANGGRAPH_POSTGRES_POOL_MAX", "10")),
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=True,
    )
    saver = PostgresSaver(_CHECKPOINTER_POOL)
    saver.setup()
    return saver


def _get_compiled_graph():
    """Lazily compile the assistant-response graph with a Postgres checkpointer."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH
    with _GRAPH_LOCK:
        if _COMPILED_GRAPH is None:
            checkpointer = _build_checkpointer()
            _COMPILED_GRAPH = _ASSISTANT_RESPONSE_GRAPH_BUILDER.compile(
                checkpointer=checkpointer,
            )
    return _COMPILED_GRAPH


@tracing.observe(name="generate_assistant_response")
def generate_assistant_response(
    user_prompt: str,
    *,
    user_needs: UserNeeds,
    retrieval: RetrievalBundle,
    config: AppConfig,
    last_user_goal: str | None,
    history: list[dict] | None,
    thread_id: str | None = None,
) -> AssistantResponseBundle:
    """Generate assistant-side outputs from retrieval and user intent.

    Args:
        user_prompt: Raw user prompt.
        user_needs: Structured user-needs object.
        retrieval: Retrieved context bundle.
        config: App runtime config.
        last_user_goal: Previous inferred user goal.
        history: Prior chat history.
        thread_id: LangGraph checkpoint thread id; one is generated when omitted
            so each invocation starts from a clean state.

    Returns:
        AssistantResponseBundle: Claims, draft, final rule, and related images.
    """
    initial_state: AssistantResponseState = {
        "user_prompt": user_prompt,
        "user_needs": user_needs,
        "retrieval": retrieval,
        "config": config,
        "last_user_goal": last_user_goal,
        "history": history,
    }
    invoke_config = {
        "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
    }
    final_state = _get_compiled_graph().invoke(initial_state, invoke_config)
    return AssistantResponseBundle(
        structured_claims=final_state["structured_claims"],
        structured_draft=final_state["structured_draft"],
        rule=final_state["rule"],
        image_query=final_state["image_query"],
        image_results=final_state["image_results"],
    )
