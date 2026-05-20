from pathlib import Path
import sqlite3

import pytest

from core.app_config import AppConfig
from core.models import (
    ExampleQuerySpec,
    FinalAnswerRubric,
    RequestAnalysis,
    SearchQuery,
    StructuredClaims,
    StructuredDraft,
    WebSource,
)
from services.prompt_service import (
    TEMPLATE_DECISION_SUPPORT,
    TEMPLATE_FINAL_ANSWER_REFLECTION,
)
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
        api_base_url="http://localhost:8000",
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


def _structured_draft() -> StructuredDraft:
    return StructuredDraft(
        theme="denim silhouettes",
        canonical=["Straight-leg denim reads as a durable baseline."],
        emerging=["Barrel and puddle shapes are appearing in styling references."],
        conflicts=["Volume can read directional or difficult depending on context."],
        gaps=["Hem length and footwear context are unresolved."],
        common_rule=[
            "Volume around the leg shifts denim from basic utility toward styling intent."
        ],
    )


def _rubric(**overrides: bool) -> FinalAnswerRubric:
    values = {
        "lends_reference_frame": True,
        "avoids_prescription": True,
        "avoids_user_judgment": True,
        "includes_interpreted_rules": True,
        "rules_are_observation_grounded": True,
        "hides_intermediate_structure": True,
        "flows_as_continuous_prose": True,
        "avoids_listicle_style": True,
        "preserves_logical_completeness": True,
    }
    values.update(overrides)
    return FinalAnswerRubric(
        **values,
        rationale="Rubric rationale.",
        revision_instruction="Revise toward a frame.",
    )


def _retrieval_with_evidence() -> workflow.RetrievalBundle:
    return workflow.RetrievalBundle(
        canonical_context="Canonical source context",
        emerging_context="Emerging source context",
        canonical_sources=[
            WebSource(
                source_id="c1",
                query_kind="canonical",
                title="Canonical denim",
                url="https://example.com/canonical",
                snippet="Straight denim baseline.",
            )
        ],
        emerging_sources=[],
    )


def test_build_checkpointer_allows_repoa_checkpoint_models(tmp_path):
    checkpointer = workflow._build_checkpointer(_make_config(tmp_path))

    try:
        assert checkpointer is not None
        allowed_modules = checkpointer.serde._allowed_msgpack_modules
        assert ("core.models", "RequestAnalysis") in allowed_modules
        assert ("services.chat_workflow", "RetrievalBundle") in allowed_modules
        assert ("core.app_config", "AppConfig") in allowed_modules
    finally:
        if workflow._CHECKPOINTER_CONN is not None:
            workflow._CHECKPOINTER_CONN.close()
            workflow._CHECKPOINTER_CONN = None


def test_generate_assistant_response_resume_invokes_checkpoint_without_new_input(
    monkeypatch,
    tmp_path,
):
    config = _make_config(tmp_path)
    captured = {}

    class FakeGraph:
        def invoke(self, graph_input, invoke_config):
            captured["graph_input"] = graph_input
            captured["invoke_config"] = invoke_config
            return {
                "request_analysis": _request_analysis(in_scope=True),
                "retrieval": workflow._empty_retrieval_bundle(),
                "rule": "Resumed response.",
            }

    monkeypatch.setattr(workflow, "_get_compiled_graph", lambda config: FakeGraph())

    result = workflow.generate_assistant_response(
        user_prompt="Resume this turn",
        config=config,
        last_request_goal="previous goal",
        history=[],
        thread_id="chat-123:2",
        resume_from_checkpoint=True,
    )

    assert result.rule == "Resumed response."
    assert captured["graph_input"] is None
    assert captured["invoke_config"]["configurable"]["thread_id"] == "chat-123:2"


def test_stream_assistant_response_normalizes_tasks_without_raw_payloads(
    monkeypatch,
    tmp_path,
):
    config = _make_config(tmp_path)
    captured = {}
    raw_prompt = "SECRET_RAW_PROMPT"
    raw_article = "SECRET_RETRIEVED_ARTICLE_CONTENT"
    raw_output = "SECRET_FULL_TASK_OUTPUT"
    raw_claim = "SECRET_CLAIM_TEXT"
    raw_draft = "SECRET_STRUCTURED_DRAFT_TEXT"
    raw_reasoning = "SECRET_REFLECTION_RATIONALE"
    raw_image_url = "https://example.test/SECRET_IMAGE_URL.jpg"

    class FakeGraph:
        def stream(self, graph_input, invoke_config, *, stream_mode, version):
            captured["graph_input"] = graph_input
            captured["invoke_config"] = invoke_config
            captured["stream_mode"] = stream_mode
            captured["version"] = version
            yield {
                "type": "tasks",
                "data": {
                    "name": "analyze_request",
                    "input": {
                        "user_prompt": raw_prompt,
                        "graph_state": {"article": raw_article},
                    },
                    "triggers": ("branch:to:analyze_request",),
                },
            }
            yield {
                "type": "tasks",
                "data": {
                    "name": "retrieve_supporting_context",
                    "error": None,
                    "result": {
                        "retrieval": workflow.RetrievalBundle(
                            canonical_context=raw_article,
                            emerging_context=raw_article,
                            canonical_sources=[
                                WebSource(
                                    source_id="C01",
                                    query_kind="canonical",
                                    title="Canonical",
                                    url="https://example.test/canonical-1",
                                    snippet=raw_article,
                                ),
                                WebSource(
                                    source_id="C02",
                                    query_kind="canonical",
                                    title="Canonical 2",
                                    url="https://example.test/canonical-2",
                                    snippet=raw_article,
                                ),
                            ],
                            emerging_sources=[
                                WebSource(
                                    source_id="E01",
                                    query_kind="emerging",
                                    title="Emerging",
                                    url="https://example.test/emerging-1",
                                    snippet=raw_article,
                                ),
                            ],
                        ),
                        "full_output": raw_output,
                    },
                    "interrupts": [],
                },
            }
            yield {
                "type": "tasks",
                "data": {
                    "name": "extract_claims",
                    "error": None,
                    "result": {
                        "structured_claims": StructuredClaims(
                            canonical_claims=[
                                {
                                    "claim": raw_claim,
                                    "claim_type": "observation",
                                    "source_id": "C01",
                                },
                                {
                                    "claim": raw_claim,
                                    "claim_type": "interpretation",
                                    "source_id": "C02",
                                },
                            ],
                            emerging_claims=[
                                {
                                    "claim": raw_claim,
                                    "claim_type": "signal",
                                    "source_id": "E01",
                                }
                            ],
                        )
                    },
                    "interrupts": [],
                },
            }
            yield {
                "type": "tasks",
                "data": {
                    "name": "extract_structured_draft",
                    "error": None,
                    "result": {
                        "structured_draft": StructuredDraft(
                            theme=raw_draft,
                            canonical=[raw_draft],
                            emerging=[raw_draft],
                            conflicts=[raw_draft],
                            gaps=[raw_draft, raw_draft],
                            common_rule=[raw_draft],
                        )
                    },
                    "interrupts": [],
                },
            }
            yield {
                "type": "tasks",
                "data": {
                    "name": "reflect_on_final_answer",
                    "error": None,
                    "result": {
                        "final_answer_reflection": FinalAnswerRubric(
                            lends_reference_frame=True,
                            avoids_prescription=True,
                            avoids_user_judgment=True,
                            includes_interpreted_rules=True,
                            rules_are_observation_grounded=True,
                            hides_intermediate_structure=True,
                            flows_as_continuous_prose=True,
                            avoids_listicle_style=True,
                            preserves_logical_completeness=True,
                            rationale=raw_reasoning,
                            revision_instruction=raw_reasoning,
                        ),
                        "final_answer_reflection_passed": True,
                    },
                    "interrupts": [],
                },
            }
            yield {
                "type": "tasks",
                "data": {
                    "name": "search_images",
                    "error": None,
                    "result": {
                        "image_results": [
                            {"image_url": raw_image_url},
                            {"image_url": raw_image_url},
                        ]
                    },
                    "interrupts": [],
                },
            }
            yield {
                "type": "checkpoints",
                "data": {
                    "values": {
                        "user_prompt": raw_prompt,
                        "retrieval": raw_article,
                    },
                    "next": ["extract_claims"],
                    "tasks": [
                        {
                            "name": "extract_claims",
                            "state": {"prompt": raw_prompt},
                        }
                    ],
                },
            }
            yield {
                "type": "values",
                "data": {
                    "request_analysis": _request_analysis(in_scope=True),
                    "retrieval": workflow.RetrievalBundle(
                        canonical_context=raw_article,
                        emerging_context="",
                    ),
                    "rule": "Safe streamed rule.",
                },
            }

    monkeypatch.setattr(workflow, "_get_compiled_graph", lambda config: FakeGraph())

    items = list(
        workflow.stream_assistant_response(
            user_prompt=raw_prompt,
            config=config,
            last_request_goal=None,
            history=[],
            thread_id="chat-stream:1",
        )
    )

    progress_events = [
        item for item in items if isinstance(item, workflow.WorkflowProgressEvent)
    ]
    assert [event.event_type for event in progress_events] == [
        "task_started",
        "task_completed",
        "progress_summary",
        "task_completed",
        "progress_summary",
        "task_completed",
        "progress_summary",
        "task_completed",
        "progress_summary",
        "task_completed",
        "progress_summary",
        "checkpoint",
    ]
    assert [
        event.label
        for event in progress_events
        if event.event_type == "progress_summary"
    ] == [
        "Retrieving evidence... 3 sources found",
        "Extracting claims... 2 canonical / 1 emerging",
        "Structuring evidence... 3 tradeoffs identified",
        "Checking output boundary... passed",
        "Finding visual references... 2 selected",
    ]
    assert progress_events[-1].next_nodes == ["extract_claims"]
    progress_json = "\n".join(event.model_dump_json() for event in progress_events)
    assert raw_prompt not in progress_json
    assert raw_article not in progress_json
    assert raw_output not in progress_json
    assert raw_claim not in progress_json
    assert raw_draft not in progress_json
    assert raw_reasoning not in progress_json
    assert raw_image_url not in progress_json
    assert "graph_state" not in progress_json
    assert "common_rule" not in progress_json
    assert "canonical_claims" not in progress_json
    assert "values" not in progress_json
    assert "tasks" not in progress_json
    assert captured["stream_mode"] == ["tasks", "checkpoints", "values"]
    assert captured["version"] == "v2"
    assert captured["graph_input"].user_prompt == raw_prompt
    assert captured["invoke_config"]["configurable"]["thread_id"] == "chat-stream:1"
    assert items[-1].rule == "Safe streamed rule."


def test_generate_assistant_response_resumes_real_sqlite_checkpoint(
    monkeypatch,
    tmp_path,
):
    config = _make_config(tmp_path)
    thread_id = "resume-smoke:1"
    calls = {
        "analyze_request": 0,
        "retrieve_supporting_context": 0,
        "extract_claims": 0,
        "extract_structured_draft": 0,
        "generate_decision_support": 0,
        "reflect_on_final_answer": 0,
        "generate_query": 0,
        "search_images": 0,
    }

    def reset_compiled_graph() -> None:
        workflow._COMPILED_GRAPH = None
        if workflow._CHECKPOINTER_CONN is not None:
            workflow._CHECKPOINTER_CONN.close()
            workflow._CHECKPOINTER_CONN = None

    def fake_analyze_request(*, user_prompt, last_request_goal, history):
        calls["analyze_request"] += 1
        return _request_analysis(in_scope=True)

    def fake_retrieve_supporting_context(request_analysis, *, config):
        calls["retrieve_supporting_context"] += 1
        return _retrieval_with_evidence()

    def fake_extract_claims(*, canonical_context, emerging_context, request_goal):
        calls["extract_claims"] += 1
        if calls["extract_claims"] == 1:
            raise RuntimeError("intentional checkpoint smoke failure")
        return StructuredClaims(canonical_claims=[], emerging_claims=[])

    def fake_extract_structured_draft(
        *,
        canonical_claims,
        emerging_claims,
        request_goal,
    ):
        calls["extract_structured_draft"] += 1
        return _structured_draft()

    def fake_generate_decision_support(**kwargs):
        calls["generate_decision_support"] += 1
        return "Recovered checkpoint answer."

    def fake_reflect_on_final_answer(**kwargs):
        calls["reflect_on_final_answer"] += 1
        return _rubric()

    def fake_generate_query(*, request_goal, rule):
        calls["generate_query"] += 1
        return ExampleQuerySpec(item="wide leg jeans", silhouette="wide")

    def fake_search_images(*args, **kwargs):
        calls["search_images"] += 1
        return []

    monkeypatch.setattr(workflow.tracing, "_ENABLED", False)
    monkeypatch.setattr(workflow.tracing, "_CLIENT", None)
    monkeypatch.setattr(workflow, "analyze_request", fake_analyze_request)
    monkeypatch.setattr(
        workflow,
        "retrieve_supporting_context",
        fake_retrieve_supporting_context,
    )
    monkeypatch.setattr(workflow, "extract_claims", fake_extract_claims)
    monkeypatch.setattr(
        workflow,
        "extract_structured_draft",
        fake_extract_structured_draft,
    )
    monkeypatch.setattr(
        workflow,
        "generate_decision_support",
        fake_generate_decision_support,
    )
    monkeypatch.setattr(
        workflow,
        "reflect_on_final_answer",
        fake_reflect_on_final_answer,
    )
    monkeypatch.setattr(workflow, "generate_query", fake_generate_query)
    monkeypatch.setattr(workflow, "search_images", fake_search_images)
    monkeypatch.setattr(workflow, "select_top_k", lambda candidates, *, k: [])

    reset_compiled_graph()
    try:
        with pytest.raises(RuntimeError, match="intentional checkpoint smoke failure"):
            workflow.generate_assistant_response(
                user_prompt="What denim shapes are trending?",
                config=config,
                last_request_goal=None,
                history=[],
                thread_id=thread_id,
            )

        with sqlite3.connect(config.langgraph_sqlite_path) as conn:
            checkpoint_count = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()[0]
        assert checkpoint_count > 0

        graph = workflow._get_compiled_graph(config)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert snapshot.values["request_analysis"].request_goal == (
            "compare denim silhouettes"
        )
        assert snapshot.values["retrieval"].has_text_evidence() is True
        assert "extract_claims" in snapshot.next
        assert calls["analyze_request"] == 1
        assert calls["retrieve_supporting_context"] == 1

        result = workflow.generate_assistant_response(
            user_prompt="What denim shapes are trending?",
            config=config,
            last_request_goal=None,
            history=[],
            thread_id=thread_id,
            resume_from_checkpoint=True,
        )

        assert isinstance(result, workflow.AssistantResponseBundle)
        assert result.rule == "Recovered checkpoint answer."
        assert calls["analyze_request"] == 1
        assert calls["retrieve_supporting_context"] == 1
        assert calls["extract_claims"] == 2
        assert calls["extract_structured_draft"] == 1
        assert calls["generate_decision_support"] == 1
        assert calls["reflect_on_final_answer"] == 1
        assert calls["generate_query"] == 1
        assert calls["search_images"] == 1
    finally:
        reset_compiled_graph()


def _patch_in_scope_graph_dependencies(monkeypatch, *, reflections):
    calls = {
        "generate_decision_support": [],
        "reflect_on_final_answer": 0,
        "generate_query": [],
    }
    reflection_queue = list(reflections)

    def fake_analyze_request(*, user_prompt, last_request_goal, history):
        return _request_analysis(in_scope=True)

    def fake_retrieve_supporting_context(request_analysis, *, config):
        return _retrieval_with_evidence()

    def fake_extract_claims(*, canonical_context, emerging_context, request_goal):
        return StructuredClaims(canonical_claims=[], emerging_claims=[])

    def fake_extract_structured_draft(
        *,
        canonical_claims,
        emerging_claims,
        request_goal,
    ):
        return _structured_draft()

    def fake_generate_decision_support(**kwargs):
        calls["generate_decision_support"].append(kwargs)
        return f"answer {len(calls['generate_decision_support'])}"

    def fake_reflect_on_final_answer(**kwargs):
        calls["reflect_on_final_answer"] += 1
        return reflection_queue.pop(0)

    def fake_generate_query(*, request_goal, rule):
        calls["generate_query"].append({"request_goal": request_goal, "rule": rule})
        return ExampleQuerySpec(item="wide leg jeans", silhouette="wide")

    monkeypatch.setattr(workflow, "analyze_request", fake_analyze_request)
    monkeypatch.setattr(
        workflow,
        "retrieve_supporting_context",
        fake_retrieve_supporting_context,
    )
    monkeypatch.setattr(workflow, "extract_claims", fake_extract_claims)
    monkeypatch.setattr(
        workflow,
        "extract_structured_draft",
        fake_extract_structured_draft,
    )
    monkeypatch.setattr(
        workflow,
        "generate_decision_support",
        fake_generate_decision_support,
    )
    monkeypatch.setattr(
        workflow,
        "reflect_on_final_answer",
        fake_reflect_on_final_answer,
    )
    monkeypatch.setattr(workflow, "generate_query", fake_generate_query)
    monkeypatch.setattr(workflow, "search_images", lambda *args, **kwargs: [])
    monkeypatch.setattr(workflow, "select_top_k", lambda candidates, *, k: [])
    return calls


def _invoke_test_graph(tmp_path):
    graph = workflow._build_assistant_response_graph_builder().compile(
        checkpointer=None
    )
    return graph.invoke(
        workflow.AssistantResponseState(
            user_prompt="What denim shapes are trending?",
            config=_make_config(tmp_path),
            last_request_goal=None,
            history=[],
        )
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
        "reflect_on_final_answer",
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
    assert (
        builder.branches["reflect_on_final_answer"][
            "_route_after_final_answer_reflection"
        ].ends["regenerate"]
        == "generate_decision_support"
    )


def test_final_answer_rubric_pass_fail_computation():
    passing = _rubric()
    failing = _rubric(avoids_prescription=False, avoids_listicle_style=False)

    assert workflow._final_answer_reflection_passes(passing) is True
    assert workflow._failed_final_answer_rubric_criteria(passing) == []
    assert workflow._final_answer_reflection_passes(failing) is False
    assert workflow._failed_final_answer_rubric_criteria(failing) == [
        "avoids_prescription",
        "avoids_listicle_style",
    ]


def test_reflection_route_passes_when_rubric_passes():
    state = workflow.AssistantResponseState(
        final_answer_reflection_passed=True,
        final_answer_reflection_attempt_count=0,
    )

    assert workflow._route_after_final_answer_reflection(state) == "passed"


def test_reflection_route_stops_after_max_attempts():
    state = workflow.AssistantResponseState(
        final_answer_reflection_passed=False,
        final_answer_reflection_attempt_count=(
            workflow.MAX_FINAL_ANSWER_REGENERATION_ATTEMPTS
        ),
    )

    assert (
        workflow._route_after_final_answer_reflection(state)
        == "continue_with_best_available"
    )


def test_final_answer_reflection_route_passes_to_image_query(
    monkeypatch,
    tmp_path,
):
    calls = _patch_in_scope_graph_dependencies(
        monkeypatch,
        reflections=[_rubric()],
    )

    final_state = _invoke_test_graph(tmp_path)

    assert len(calls["generate_decision_support"]) == 1
    assert calls["reflect_on_final_answer"] == 1
    assert final_state["rule"] == "answer 1"
    assert final_state["final_answer_reflection_passed"] is True
    assert final_state["final_answer_reflection_attempt_count"] == 0
    assert calls["generate_query"] == [
        {"request_goal": "compare denim silhouettes", "rule": "answer 1"}
    ]


def test_final_answer_reflection_route_regenerates_once(
    monkeypatch,
    tmp_path,
):
    calls = _patch_in_scope_graph_dependencies(
        monkeypatch,
        reflections=[
            _rubric(avoids_prescription=False),
            _rubric(),
        ],
    )

    final_state = _invoke_test_graph(tmp_path)

    assert len(calls["generate_decision_support"]) == 2
    assert calls["reflect_on_final_answer"] == 2
    assert final_state["rule"] == "answer 2"
    assert final_state["final_answer_reflection_passed"] is True
    assert final_state["final_answer_reflection_attempt_count"] == 1
    assert calls["generate_decision_support"][1]["failed_criteria"] == [
        "avoids_prescription"
    ]
    assert calls["generate_query"] == [
        {"request_goal": "compare denim silhouettes", "rule": "answer 2"}
    ]


def test_final_answer_reflection_route_stops_after_one_regeneration(
    monkeypatch,
    tmp_path,
):
    calls = _patch_in_scope_graph_dependencies(
        monkeypatch,
        reflections=[
            _rubric(avoids_prescription=False),
            _rubric(avoids_prescription=False),
        ],
    )

    final_state = _invoke_test_graph(tmp_path)

    assert len(calls["generate_decision_support"]) == 2
    assert calls["reflect_on_final_answer"] == 2
    assert final_state["rule"] == "answer 2"
    assert final_state["final_answer_reflection_passed"] is False
    assert final_state["final_answer_reflection_attempt_count"] == 1
    assert final_state["final_answer_reflection_failure_reason"] == (
        "Rubric rationale."
    )
    assert calls["generate_query"] == [
        {"request_goal": "compare denim silhouettes", "rule": "answer 2"}
    ]


def test_final_answer_prompt_contains_thesis_constraints():
    prompt = TEMPLATE_DECISION_SUPPORT.module.system(
        request_goal="compare denim silhouettes",
        last_request_goal=None,
        failed_criteria=None,
        reflection_rationale=None,
        revision_instruction=None,
    )
    normalized = prompt.lower()

    assert "reference frame" in normalized
    assert "recommendation" in normalized
    assert "interpreted rules" in normalized
    assert "解釈ルール" in prompt
    assert "continuous prose" in normalized
    assert "internal reasoning substrate" in normalized
    assert "do not expose the schema shape" in normalized
    assert "now:" not in normalized


def test_final_answer_reflection_prompt_contains_localized_rules_constraints():
    prompt = TEMPLATE_FINAL_ANSWER_REFLECTION.module.system(
        request_goal="compare denim silhouettes",
    )
    normalized = prompt.lower()

    assert "interpreted rules" in normalized
    assert "解釈ルール" in prompt
    assert "explicit, not merely implied" in normalized
    assert "when [observable condition], it may signal [interpretation]" in normalized
    assert "[観測可能な条件] が見られるとき、[解釈] を示す可能性がある" in prompt
    assert "now:" not in normalized


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
