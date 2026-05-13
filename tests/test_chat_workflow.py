from pathlib import Path

import pytest

from core.app_config import AppConfig
from core.models import RequestAnalysis, SearchQuery
from services import chat_workflow as workflow


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        db_path=tmp_path / "chat_db",
        chat_db_name="chat_db",
        user_db_name="user_db",
        chat_meta_db_name="chat_meta_db",
        app_log_level="INFO",
        tavily_api_key="",
        tavily_text_max_results=5,
        tavily_search_depth="basic",
        tavily_include_raw_content=False,
        tavily_image_fetch_limit=10,
        tavily_image_limit=3,
        tavily_include_image_descriptions=True,
        langgraph_sqlite_path=tmp_path / "checkpoints.sqlite",
        workspace_query_key="workspace",
        default_workspace_key="demo",
    )


def _request_analysis(*, in_scope: bool) -> RequestAnalysis:
    return RequestAnalysis(
        request_goal="compare denim silhouettes",
        candidate_queries=SearchQuery(
            canonical_query="classic denim silhouettes",
            emerging_query="emerging denim silhouettes",
        ),
        vertical="womens",
        is_in_scope=in_scope,
    )


def test_assistant_response_graph_topology_includes_expected_nodes():
    builder = workflow._build_assistant_response_graph_builder()

    assert {
        "analyze_request",
        "route_by_scope",
        "out_of_scope_response",
        "retrieve_supporting_context",
        "evidence_unavailable_response",
        "extract_claims",
        "extract_structured_draft",
        "generate_decision_support",
        "generate_query",
        "render_image_query",
        "search_images",
    }.issubset(builder.nodes)

    assert (
        builder.branches["route_by_scope"]["_route_by_scope"].ends["out_of_scope"]
        == "out_of_scope_response"
    )
    assert (
        builder.branches["retrieve_supporting_context"]["_route_by_evidence"].ends[
            "missing_evidence"
        ]
        == "evidence_unavailable_response"
    )


def test_missing_evidence_routes_to_evidence_unavailable_response(
    monkeypatch,
    tmp_path,
):
    calls = {"retrieval": 0}

    def fake_analyze_request(*, user_prompt, last_request_goal, history):
        return _request_analysis(in_scope=True)

    def fake_retrieve_supporting_context(request_analysis, *, config):
        calls["retrieval"] += 1
        return workflow._empty_retrieval_bundle(error_message="No evidence available")

    def fail_extract_claims(*args, **kwargs):
        pytest.fail("extract_claims should not run when evidence is missing")

    monkeypatch.setattr(workflow, "analyze_request", fake_analyze_request)
    monkeypatch.setattr(
        workflow,
        "retrieve_supporting_context",
        fake_retrieve_supporting_context,
    )
    monkeypatch.setattr(workflow, "extract_claims", fail_extract_claims)

    graph = workflow._build_assistant_response_graph_builder().compile(
        checkpointer=None
    )
    final_state = graph.invoke(
        workflow.AssistantResponseState(
            user_prompt="What denim shapes are trending?",
            config=_make_config(tmp_path),
            last_request_goal=None,
            history=[],
        )
    )

    assert calls["retrieval"] == 1
    assert final_state["rule"] == "No evidence available"
    assert final_state["image_query"] == ""
    assert final_state["image_results"] == []


def test_out_of_scope_path_does_not_call_retrieval(monkeypatch, tmp_path):
    def fake_analyze_request(*, user_prompt, last_request_goal, history):
        return _request_analysis(in_scope=False)

    def fail_retrieve_supporting_context(*args, **kwargs):
        pytest.fail("retrieve_supporting_context should not run out of scope")

    monkeypatch.setattr(workflow, "analyze_request", fake_analyze_request)
    monkeypatch.setattr(
        workflow,
        "retrieve_supporting_context",
        fail_retrieve_supporting_context,
    )

    graph = workflow._build_assistant_response_graph_builder().compile(
        checkpointer=None
    )
    final_state = graph.invoke(
        workflow.AssistantResponseState(
            user_prompt="Explain database indexing.",
            config=_make_config(tmp_path),
            last_request_goal=None,
            history=[],
        )
    )

    assert final_state["rule"] == workflow.OUT_OF_SCOPE_MESSAGE
    assert final_state["retrieval"].has_text_evidence() is False
    assert final_state["image_query"] == ""
    assert final_state["image_results"] == []
