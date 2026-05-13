"""High-level chat workflow orchestration shared by the Streamlit UI."""

import logging
import os
import threading
import uuid

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessage
from psycopg_pool import ConnectionPool

from core.app_config import AppConfig
from core.models import (
    ExampleQuerySpec,
    RequestAnalysis,
    StructuredClaims,
    StructuredDraft,
    WebSource,
)
from core.query_utils import render_example_query
from core.text_utils import normalize_text_nfkc
from services import tracing
from services.chat import (
    analyze_request,
    extract_claims,
    extract_structured_draft,
    generate_decision_support,
    generate_query,
)
from services.image_search import (
    ImageSearchResult,
    search_images,
    select_top_k,
)
from services.web_search import (
    dedupe_sources_by_url,
    search_text_sources,
    sources_to_prompt_context,
    sources_to_table_rows,
)

logger = logging.getLogger(__name__)
OUT_OF_SCOPE_MESSAGE = (
    "This app is focused on fashion, styling, and trend analysis.\n"
    "This request appears to be outside that scope, so I did not run retrieval or "
    "reasoning.\n"
    "Please reframe the question in a fashion, styling, outfit, apparel, visual "
    "reference, or trend-related context."
)
TAVILY_NOT_CONFIGURED_MESSAGE = (
    "I cannot run the in-scope evidence workflow because Tavily text retrieval is "
    "not configured. Set TAVILY_API_KEY and try again; I did not generate an "
    "evidence-based answer without sources."
)
TAVILY_NO_EVIDENCE_MESSAGE = (
    "I could not retrieve Tavily text evidence for this in-scope request, so I am "
    "abstaining instead of generating an evidence-based trend rule. Try again "
    "later or broaden the fashion/style query."
)


def _build_fallback_image_query(request_analysis: RequestAnalysis, rule: str) -> str:
    """Build a safe fallback image query when structured rendering fails.

    Args:
        request_analysis: Structured request-analysis object.
        rule: Final synthesized assistant rule.

    Returns:
        str: A conservative image-search query string.
    """
    vertical = str(getattr(request_analysis, "vertical", "") or "").strip().lower()
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


class RetrievalBundle(BaseModel):
    """Retrieved web evidence used downstream by the chat workflow."""

    canonical_context: str
    emerging_context: str
    canonical_sources: list[WebSource] = Field(default_factory=list)
    emerging_sources: list[WebSource] = Field(default_factory=list)
    canonical_rows: list[dict[str, str]] = Field(default_factory=list)
    emerging_rows: list[dict[str, str]] = Field(default_factory=list)
    error_message: str = ""

    def has_text_evidence(self) -> bool:
        """Return True when at least one normalized source is available."""
        return bool(self.canonical_sources or self.emerging_sources)

    def total_source_count(self) -> int:
        """Return total normalized source count."""
        return len(self.canonical_sources) + len(self.emerging_sources)


def _empty_retrieval_bundle(error_message: str = "") -> RetrievalBundle:
    return RetrievalBundle(
        canonical_context="",
        emerging_context="",
        canonical_sources=[],
        emerging_sources=[],
        canonical_rows=[],
        emerging_rows=[],
        error_message=error_message,
    )


class AssistantResponseBundle(BaseModel):
    """All derived assistant artifacts for a single user prompt."""

    request_analysis: RequestAnalysis
    retrieval: RetrievalBundle = Field(default_factory=_empty_retrieval_bundle)
    structured_claims: StructuredClaims | None = None
    structured_draft: StructuredDraft | None = None
    rule: str
    image_query: str = ""
    image_results: list[ImageSearchResult] = Field(default_factory=list)


@tracing.observe(name="retrieve_supporting_context")
def retrieve_supporting_context(
    request_analysis: RequestAnalysis,
    *,
    config: AppConfig,
) -> RetrievalBundle:
    """Retrieve canonical and emerging context for the current request.

    Args:
        request_analysis: Structured request-analysis object containing candidate queries.
        config: App runtime config.

    Returns:
        RetrievalBundle: Retrieved contexts and table rows for display.
    """
    if not str(config.tavily_api_key or "").strip():
        _log_text_retrieval_counts(
            canonical_source_count=0,
            emerging_source_count=0,
            total_source_count=0,
            reason="missing_tavily_api_key",
        )
        return _empty_retrieval_bundle(error_message=TAVILY_NOT_CONFIGURED_MESSAGE)

    candidate_queries = request_analysis.candidate_queries
    canonical_query = str(candidate_queries.canonical_query or "")
    emerging_query = str(candidate_queries.emerging_query or "")
    canonical_sources = search_text_sources(
        canonical_query,
        query_kind="canonical",
        max_results=config.tavily_text_max_results,
        api_key=config.tavily_api_key,
        search_depth=config.tavily_search_depth,
        include_raw_content=config.tavily_include_raw_content,
    )
    emerging_sources = search_text_sources(
        emerging_query,
        query_kind="emerging",
        max_results=config.tavily_text_max_results,
        api_key=config.tavily_api_key,
        search_depth=config.tavily_search_depth,
        include_raw_content=config.tavily_include_raw_content,
    )
    canonical_sources, emerging_sources = dedupe_sources_by_url(
        canonical_sources=canonical_sources,
        emerging_sources=emerging_sources,
    )
    bundle = RetrievalBundle(
        canonical_context=sources_to_prompt_context(
            canonical_sources,
            label="Canonical",
        ),
        emerging_context=sources_to_prompt_context(
            emerging_sources,
            label="Emerging",
        ),
        canonical_sources=canonical_sources,
        emerging_sources=emerging_sources,
        canonical_rows=sources_to_table_rows(canonical_sources),
        emerging_rows=sources_to_table_rows(emerging_sources),
    )
    if not bundle.has_text_evidence():
        bundle.error_message = TAVILY_NO_EVIDENCE_MESSAGE
    _log_text_retrieval_counts(
        canonical_source_count=len(canonical_sources),
        emerging_source_count=len(emerging_sources),
        total_source_count=bundle.total_source_count(),
        reason=(bundle.error_message or None),
    )
    return bundle


def _log_text_retrieval_counts(
    *,
    canonical_source_count: int,
    emerging_source_count: int,
    total_source_count: int,
    reason: str | None = None,
) -> None:
    logger.info(
        "text_retrieval_complete backend=tavily canonical_source_count=%s "
        "emerging_source_count=%s total_source_count=%s reason=%s",
        canonical_source_count,
        emerging_source_count,
        total_source_count,
        reason or "",
    )
    metadata: dict[str, object] = {
        "text_retrieval_backend": "tavily",
        "canonical_source_count": canonical_source_count,
        "emerging_source_count": emerging_source_count,
        "total_source_count": total_source_count,
    }
    if reason:
        metadata["text_retrieval_reason"] = reason
    tracing.update_current_trace(metadata=metadata)


class AssistantResponseState(BaseModel):
    """LangGraph state for the linear assistant-response pipeline.

    Each node progressively populates its own fields on the state as the graph
    advances, so most fields are absent at earlier steps and default to
    ``None`` (or empty containers). The required input fields (``user_prompt``,
    ``config``, ``last_request_goal``, and ``history``)
    are always supplied in the initial state passed to ``invoke()``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_prompt: str | None = None
    request_analysis: RequestAnalysis | None = None
    retrieval: RetrievalBundle | None = None
    config: AppConfig | None = None
    last_request_goal: str | None = None
    history: list[ModelMessage] | None = None
    structured_claims: StructuredClaims | None = None
    structured_draft: StructuredDraft | None = None
    rule: str | None = None
    query_spec: ExampleQuerySpec | None = None
    image_query: str | None = None
    image_results: list[ImageSearchResult] = Field(default_factory=list)


def _node_analyze_request(state: AssistantResponseState) -> dict:
    request_analysis = analyze_request(
        user_prompt=state.user_prompt,
        last_request_goal=state.last_request_goal,
        history=state.history,
    )
    return {"request_analysis": request_analysis}


def _node_route_by_scope(state: AssistantResponseState) -> dict:
    return {}


def _route_by_scope(state: AssistantResponseState) -> str:
    if state.request_analysis and state.request_analysis.is_in_scope:
        return "in_scope"
    return "out_of_scope"


def _node_out_of_scope_response(state: AssistantResponseState) -> dict:
    return {
        "retrieval": _empty_retrieval_bundle(),
        "rule": OUT_OF_SCOPE_MESSAGE,
        "image_query": "",
        "image_results": [],
    }


def _node_retrieve_supporting_context(state: AssistantResponseState) -> dict:
    retrieval = _empty_retrieval_bundle()
    try:
        retrieval = retrieve_supporting_context(
            state.request_analysis,
            config=state.config,
        )
    except Exception as err:
        logger.warning("Tavily text retrieval failed: %s", err)
        retrieval = _empty_retrieval_bundle(error_message=TAVILY_NO_EVIDENCE_MESSAGE)
        _log_text_retrieval_counts(
            canonical_source_count=0,
            emerging_source_count=0,
            total_source_count=0,
            reason="text_retrieval_exception",
        )
    return {"retrieval": retrieval}


def _route_by_evidence(state: AssistantResponseState) -> str:
    retrieval = state.retrieval or _empty_retrieval_bundle()
    if retrieval.has_text_evidence():
        return "has_evidence"
    return "missing_evidence"


def _node_evidence_unavailable_response(state: AssistantResponseState) -> dict:
    retrieval = state.retrieval or _empty_retrieval_bundle()
    return {
        "rule": retrieval.error_message or TAVILY_NO_EVIDENCE_MESSAGE,
        "image_query": "",
        "image_results": [],
    }


def _node_extract_claims(state: AssistantResponseState) -> dict:
    structured_claims = extract_claims(
        canonical_context=state.retrieval.canonical_context,
        emerging_context=state.retrieval.emerging_context,
        request_goal=state.request_analysis.request_goal,
    )
    return {"structured_claims": structured_claims}


def _node_extract_structured_draft(state: AssistantResponseState) -> dict:
    structured_draft = extract_structured_draft(
        canonical_claims=state.structured_claims.canonical_claims,
        emerging_claims=state.structured_claims.emerging_claims,
        request_goal=state.request_analysis.request_goal,
    )
    return {"structured_draft": structured_draft}


def _node_generate_decision_support(state: AssistantResponseState) -> dict:
    decision_support = generate_decision_support(
        user_prompt=state.user_prompt,
        request_goal=state.request_analysis.request_goal,
        last_request_goal=state.last_request_goal,
        history=state.history,
        structured_draft=state.structured_draft,
    )
    rule = normalize_text_nfkc(decision_support or "")
    return {"rule": rule}


def _node_generate_query(state: AssistantResponseState) -> dict:
    query_spec = generate_query(
        request_goal=state.request_analysis.request_goal,
        rule=state.rule,
    )
    return {"query_spec": query_spec}


def _node_render_image_query(state: AssistantResponseState) -> dict:
    query_spec = state.query_spec
    try:
        image_query = render_example_query(query_spec)
    except ValueError as err:
        image_query = _build_fallback_image_query(
            request_analysis=state.request_analysis,
            rule=state.rule,
        )
        logger.warning(
            "render_example_query failed: %s; fallback_image_query=%s query_spec=%s",
            err,
            image_query,
            query_spec.model_dump(),
        )
    return {"image_query": image_query}


def _node_search_images(state: AssistantResponseState) -> dict:
    config = state.config
    image_query = state.image_query
    image_results: list[ImageSearchResult] = []
    raw_candidate_count = 0
    try:
        logger.info(
            "image_search_start backend=tavily query=%r fetch_limit=%s final_limit=%s",
            image_query,
            config.tavily_image_fetch_limit,
            config.tavily_image_limit,
        )
        candidates = search_images(
            image_query,
            limit=config.tavily_image_fetch_limit,
            api_key=config.tavily_api_key,
            include_image_descriptions=config.tavily_include_image_descriptions,
        )
        raw_candidate_count = len(candidates)
        image_results = select_top_k(candidates, k=config.tavily_image_limit)
    except Exception as err:
        logger.warning("Tavily image search failed: %s", err)
    final_count = len(image_results)
    logger.info(
        "image_search_complete backend=tavily raw_candidate_count=%s final_count=%s",
        raw_candidate_count,
        final_count,
    )
    tracing.update_current_trace(
        metadata={
            "visual_search_backend": "tavily",
            "raw_image_candidate_count": raw_candidate_count,
            "final_image_result_count": final_count,
        }
    )
    return {"image_results": image_results}


def _build_assistant_response_graph_builder() -> StateGraph:
    """Build the state machine for the assistant response.

    In-scope requests enter the deterministic RAR path after Tavily text
    evidence is available. Missing evidence routes to a safe abstention rather
    than claims extraction. Query refinement loops are deliberately out of scope
    here and are planned as future Agentic RAR work on top of this graph.
    """
    graph = StateGraph(AssistantResponseState)
    graph.add_node("analyze_request", _node_analyze_request)
    graph.add_node("route_by_scope", _node_route_by_scope)
    graph.add_node("out_of_scope_response", _node_out_of_scope_response)
    graph.add_node("retrieve_supporting_context", _node_retrieve_supporting_context)
    graph.add_node("evidence_unavailable_response", _node_evidence_unavailable_response)
    graph.add_node("extract_claims", _node_extract_claims)
    graph.add_node("extract_structured_draft", _node_extract_structured_draft)
    graph.add_node("generate_decision_support", _node_generate_decision_support)
    graph.add_node("generate_query", _node_generate_query)
    graph.add_node("render_image_query", _node_render_image_query)
    graph.add_node("search_images", _node_search_images)

    graph.add_edge(START, "analyze_request")
    graph.add_edge("analyze_request", "route_by_scope")
    graph.add_conditional_edges(
        "route_by_scope",
        _route_by_scope,
        {
            "out_of_scope": "out_of_scope_response",
            "in_scope": "retrieve_supporting_context",
        },
    )
    graph.add_edge("out_of_scope_response", END)
    graph.add_conditional_edges(
        "retrieve_supporting_context",
        _route_by_evidence,
        {
            "has_evidence": "extract_claims",
            "missing_evidence": "evidence_unavailable_response",
        },
    )
    graph.add_edge("evidence_unavailable_response", END)
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
    config: AppConfig,
    last_request_goal: str | None,
    history: list[ModelMessage] | None,
    thread_id: str | None = None,
    langfuse_session_id: str | None = None,
    langfuse_user_id: str | None = None,
) -> AssistantResponseBundle:
    """Generate assistant-side outputs from an end-to-end LangGraph workflow.

    Args:
        user_prompt: Raw user prompt.
        config: App runtime config.
        last_request_goal: Previous inferred request goal.
        history: Prior chat history.
        thread_id: LangGraph checkpoint thread id; one is generated when omitted
            so each invocation starts from a clean state.
        langfuse_session_id: Chat session id used only for Langfuse grouping.
        langfuse_user_id: User id used only for Langfuse grouping.

    Returns:
        AssistantResponseBundle: Claims, draft, final rule, and related images.
    """
    initial_state = AssistantResponseState(
        user_prompt=user_prompt,
        config=config,
        last_request_goal=last_request_goal,
        history=history,
    )
    resolved_thread_id = thread_id or str(uuid.uuid4())
    tags = [*tracing.get_repoa_trace_tags(), "chat_workflow", "langgraph"]
    metadata = {
        **tracing.get_repoa_trace_metadata(),
        "chat_id": langfuse_session_id,
        "langfuse_session_id": langfuse_session_id,
        "langfuse_user_id": langfuse_user_id,
        "thread_id": resolved_thread_id,
        "text_retrieval_backend": "tavily",
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    invoke_config = {
        "configurable": {"thread_id": resolved_thread_id},
        **tracing.get_langchain_invoke_config(
            run_name="chat_workflow",
            tags=tags,
            metadata=metadata,
        ),
    }
    with tracing.propagate_attributes(
        session_id=langfuse_session_id,
        user_id=langfuse_user_id,
        tags=tags,
    ):
        final_state = _get_compiled_graph().invoke(initial_state, invoke_config)
    return AssistantResponseBundle(
        request_analysis=final_state["request_analysis"],
        retrieval=final_state.get("retrieval") or _empty_retrieval_bundle(),
        structured_claims=final_state.get("structured_claims"),
        structured_draft=final_state.get("structured_draft"),
        rule=final_state.get("rule") or "",
        image_query=final_state.get("image_query") or "",
        image_results=final_state.get("image_results") or [],
    )
