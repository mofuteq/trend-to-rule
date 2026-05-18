from pathlib import Path

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse

import api
from core.app_config import AppConfig
from core.models import RequestAnalysis, SearchQuery, WebSource
from services.api_models import PersistedTurnArtifacts
from services.chat_runtime import ChatTurnResult
from services.chat_session import open_chat_db
from services.chat_workflow import AssistantResponseBundle, RetrievalBundle
from services.image_search import ImageSearchResult


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
        api_base_url="http://testserver",
    )


def _image_result(label: str = "wide-leg denim") -> ImageSearchResult:
    return ImageSearchResult(
        title=f"Visual {label}",
        page_url=f"https://example.test/{label}/page",
        image_url=f"https://example.test/{label}/image.jpg",
        thumbnail_url=f"https://example.test/{label}/thumb.jpg",
        source="fixture",
        engine="fixture",
    )


def _retrieval_bundle(label: str = "denim") -> RetrievalBundle:
    canonical_url = f"https://example.test/{label}/canonical"
    emerging_url = f"https://example.test/{label}/emerging"
    return RetrievalBundle(
        canonical_context=f"Canonical context for {label}",
        emerging_context=f"Emerging context for {label}",
        canonical_sources=[
            WebSource(
                source_id="C01",
                query_kind="canonical",
                title=f"Canonical {label}",
                url=canonical_url,
                snippet=f"Canonical snippet for {label}",
                published_at="2026-01-01",
            )
        ],
        emerging_sources=[
            WebSource(
                source_id="E01",
                query_kind="emerging",
                title=f"Emerging {label}",
                url=emerging_url,
                snippet=f"Emerging snippet for {label}",
                published_at="2026-02-01",
            )
        ],
        canonical_rows=[
            {
                "source_id": "C01",
                "title": f"Canonical {label}",
                "url": canonical_url,
                "published_at": "2026-01-01",
                "provider": "tavily",
            }
        ],
        emerging_rows=[
            {
                "source_id": "E01",
                "title": f"Emerging {label}",
                "url": emerging_url,
                "published_at": "2026-02-01",
                "provider": "tavily",
            }
        ],
    )


def _assistant_response(
    request_goal: str = "compare denim silhouettes",
    *,
    image_results: list[ImageSearchResult] | None = None,
    retrieval: RetrievalBundle | None = None,
    image_query: str = "wide leg denim outfits",
):
    return AssistantResponseBundle(
        request_analysis=RequestAnalysis(
            request_goal=request_goal,
            candidate_queries=SearchQuery(
                canonical_query="classic denim silhouettes",
                emerging_query="emerging denim silhouettes",
            ),
            vertical="womens",
            is_in_scope=True,
        ),
        rule=f"Assistant rule for {request_goal}.",
        retrieval=retrieval or RetrievalBundle(
            canonical_context="",
            emerging_context="",
        ),
        image_query=image_query,
        image_results=image_results or [],
    )


def _install_test_api_config(monkeypatch, tmp_path: Path) -> AppConfig:
    config = _make_config(tmp_path)
    monkeypatch.setattr(api, "CONFIG", config)
    monkeypatch.setattr(api, "_CHAT_DB", None)
    monkeypatch.setattr(api, "generate_chat_title", lambda messages: "Denim title")
    return config


def test_health_endpoint_returns_ok():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_chat_endpoint_initializes_chat(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    client = TestClient(api.app)

    response = client.post(
        "/chats",
        json={"workspace_id": "workspace-a", "chat_id": "chat-created"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] == "chat-created"
    assert payload["workspace_id"] == "workspace-a"
    assert payload["messages"] == []
    assert payload["title"] == ""
    assert payload["last_request_goal"] == ""
    assert payload["chat_turn"] == 0
    assert payload["turn_artifacts"] == {}

    chat_db = open_chat_db(config)
    assert chat_db.get("workspace-a", config.user_db_name) == ["chat-created"]
    assert chat_db.get("chat-created", config.chat_meta_db_name)["chat_turn"] == 0


def test_list_chats_endpoint_returns_workspace_summaries(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    chat_db.put("workspace-a", ["older", "newer"], config.user_db_name)
    chat_db.put(
        "older",
        {"title": "Older chat", "updated_at_ts": 10.0, "chat_turn": 1},
        config.chat_meta_db_name,
    )
    chat_db.put(
        "newer",
        {
            "title": "Newer chat",
            "updated_at_ts": 20.0,
            "chat_turn": 1,
            "latest_workflow_status": "failed",
            "latest_chat_turn": 2,
            "latest_thread_id": "newer:2",
            "latest_workflow_error": "temporary failure",
        },
        config.chat_meta_db_name,
    )
    client = TestClient(api.app)

    response = client.get("/chats", params={"workspace_id": "workspace-a"})

    assert response.status_code == 200
    assert response.json() == {
        "workspace_id": "workspace-a",
        "chats": [
            {
                "chat_id": "newer",
                "title": "Newer chat",
                "updated_at_ts": 20.0,
                "latest_workflow_status": "failed",
                "latest_chat_turn": 2,
                "latest_thread_id": "newer:2",
                "latest_workflow_error": "temporary failure",
            },
            {
                "chat_id": "older",
                "title": "Older chat",
                "updated_at_ts": 10.0,
                "latest_workflow_status": "",
                "latest_chat_turn": None,
                "latest_thread_id": "",
                "latest_workflow_error": "",
            },
        ],
    }


def test_get_chat_endpoint_returns_messages_and_metadata(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    chat_db.put(
        "chat-read",
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        config.chat_db_name,
    )
    chat_db.put(
        "chat-read",
        {
            "title": "Read title",
            "last_request_goal": "read goal",
            "chat_turn": 1,
            "updated_at_ts": 30.0,
            "latest_workflow_status": "running",
            "latest_chat_turn": 2,
            "latest_thread_id": "chat-read:2",
            "latest_workflow_error": "",
        },
        config.chat_meta_db_name,
    )
    client = TestClient(api.app)

    response = client.get(
        "/chats/chat-read",
        params={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "chat_id": "chat-read",
        "workspace_id": "workspace-a",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        "title": "Read title",
        "last_request_goal": "read goal",
        "chat_turn": 1,
        "latest_workflow_status": "running",
        "latest_chat_turn": 2,
        "latest_thread_id": "chat-read:2",
        "latest_workflow_error": "",
        "turn_artifacts": {},
    }


def test_get_chat_endpoint_returns_persisted_turn_artifacts(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    chat_db.put(
        "chat-artifacts",
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        config.chat_db_name,
    )
    artifact = PersistedTurnArtifacts.from_assistant_response(
        chat_turn=1,
        assistant_response=_assistant_response(
            request_goal="read restored artifacts",
            image_results=[_image_result("restored")],
            retrieval=_retrieval_bundle("restored"),
            image_query="restored image query",
        ),
    )
    chat_db.put(
        "chat-artifacts",
        {
            "chat_turn": 1,
            "turn_artifacts": {"1": artifact.model_dump(mode="json")},
        },
        config.chat_meta_db_name,
    )
    client = TestClient(api.app)

    response = client.get(
        "/chats/chat-artifacts",
        params={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 200
    payload = response.json()
    restored = payload["turn_artifacts"]["1"]
    assert restored["chat_turn"] == 1
    assert restored["request_goal"] == "read restored artifacts"
    assert restored["is_in_scope"] is True
    assert restored["image_results"][0]["title"] == "Visual restored"
    assert restored["retrieval"]["canonical_rows"][0]["url"] == (
        "https://example.test/restored/canonical"
    )
    assert restored["retrieval"]["canonical_context"] == ""
    assert restored["retrieval"]["canonical_sources"] == []


def test_delete_chat_endpoint_removes_chat_and_workspace_entry(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    chat_db.put("workspace-a", ["keep-chat", "delete-chat"], config.user_db_name)
    chat_db.put(
        "keep-chat",
        {"title": "Keep", "updated_at_ts": 20.0, "chat_turn": 0},
        config.chat_meta_db_name,
    )
    chat_db.put(
        "delete-chat",
        [{"role": "user", "content": "Delete me"}],
        config.chat_db_name,
    )
    chat_db.put(
        "delete-chat",
        {"title": "Delete", "updated_at_ts": 30.0, "chat_turn": 1},
        config.chat_meta_db_name,
    )
    client = TestClient(api.app)

    response = client.delete(
        "/chats/delete-chat",
        params={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted_chat_id": "delete-chat",
        "workspace_id": "workspace-a",
        "remaining_chat_ids": ["keep-chat"],
        "chats": [
            {
                "chat_id": "keep-chat",
                "title": "Keep",
                "updated_at_ts": 20.0,
                "latest_workflow_status": "",
                "latest_chat_turn": None,
                "latest_thread_id": "",
                "latest_workflow_error": "",
            },
        ],
    }
    assert chat_db.get("delete-chat", config.chat_db_name) is None
    assert chat_db.get("delete-chat", config.chat_meta_db_name) is None
    assert chat_db.get("workspace-a", config.user_db_name) == ["keep-chat"]


def test_chat_endpoint_owns_turn_numbering_and_persistence(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    calls = []

    def fake_run_chat_turn(**kwargs):
        calls.append(kwargs)
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt=kwargs["user_prompt"],
            assistant_response=_assistant_response(
                request_goal=f"goal {kwargs['chat_turn']}",
                image_results=[_image_result(f"turn-{kwargs['chat_turn']}")],
                retrieval=_retrieval_bundle(f"turn-{kwargs['chat_turn']}"),
                image_query=f"image query {kwargs['chat_turn']}",
            ),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    first = client.post(
        "/chat",
        json={
            "chat_id": "chat-123",
            "workspace_id": "workspace-a",
            "message": "First turn",
        },
    )
    second = client.post(
        "/chat",
        json={
            "chat_id": "chat-123",
            "workspace_id": "workspace-a",
            "message": "Second turn",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["chat_turn"] == 1
    assert second.json()["chat_turn"] == 2
    assert first.json()["title"] == "Denim title"
    assert second.json()["title"] == "Denim title"
    assert [call["chat_turn"] for call in calls] == [1, 2]
    assert [call["thread_id"] for call in calls] == ["chat-123:1", "chat-123:2"]

    chat_db = open_chat_db(config)
    stored_messages = chat_db.get("chat-123", config.chat_db_name)
    assert stored_messages == [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "Assistant rule for goal 1."},
        {"role": "user", "content": "Second turn"},
        {"role": "assistant", "content": "Assistant rule for goal 2."},
    ]
    meta = chat_db.get("chat-123", config.chat_meta_db_name)
    assert meta["chat_turn"] == 2
    assert meta["last_request_goal"] == "goal 2"
    assert meta["title"] == "Denim title"
    assert set(meta["turn_artifacts"]) == {"1", "2"}
    assert meta["turn_artifacts"]["1"]["chat_turn"] == 1
    assert meta["turn_artifacts"]["1"]["image_query"] == "image query 1"
    assert meta["turn_artifacts"]["1"]["image_results"][0]["title"] == (
        "Visual turn-1"
    )
    assert meta["turn_artifacts"]["1"]["retrieval"]["canonical_rows"][0]["url"] == (
        "https://example.test/turn-1/canonical"
    )
    assert meta["turn_artifacts"]["1"]["retrieval"]["canonical_context"] == ""
    assert meta["turn_artifacts"]["1"]["retrieval"]["canonical_sources"] == []
    assert meta["turn_artifacts"]["2"]["chat_turn"] == 2
    assert meta["turn_artifacts"]["2"]["image_results"][0]["title"] == (
        "Visual turn-2"
    )


def test_chat_endpoint_replaces_artifacts_for_same_completed_turn(
    monkeypatch,
    tmp_path,
):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    existing_artifact = PersistedTurnArtifacts.from_assistant_response(
        chat_turn=1,
        assistant_response=_assistant_response(
            request_goal="old goal",
            image_results=[_image_result("old")],
            retrieval=_retrieval_bundle("old"),
            image_query="old image query",
        ),
    )
    chat_db.put(
        "chat-replace-artifacts",
        [
            {"role": "user", "content": "Original question"},
            {"role": "assistant", "content": "Original answer"},
        ],
        config.chat_db_name,
    )
    chat_db.put(
        "chat-replace-artifacts",
        {
            "chat_turn": 0,
            "turn_artifacts": {"1": existing_artifact.model_dump(mode="json")},
        },
        config.chat_meta_db_name,
    )

    def fake_run_chat_turn(**kwargs):
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt=kwargs["user_prompt"],
            assistant_response=_assistant_response(
                request_goal="replacement goal",
                image_results=[_image_result("replacement")],
                retrieval=_retrieval_bundle("replacement"),
                image_query="replacement image query",
            ),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "chat_id": "chat-replace-artifacts",
            "workspace_id": "workspace-a",
            "message": "Retry same turn",
        },
    )

    assert response.status_code == 200
    meta = chat_db.get("chat-replace-artifacts", config.chat_meta_db_name)
    assert list(meta["turn_artifacts"]) == ["1"]
    assert meta["turn_artifacts"]["1"]["image_query"] == "replacement image query"
    assert meta["turn_artifacts"]["1"]["image_results"][0]["title"] == (
        "Visual replacement"
    )
    assert chat_db.get("chat-replace-artifacts", config.chat_db_name) == [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "Original answer"},
    ]


def test_chat_endpoint_stores_workflow_metadata_running_then_completed(
    monkeypatch,
    tmp_path,
):
    config = _install_test_api_config(monkeypatch, tmp_path)
    running_meta = {}

    def fake_run_chat_turn(**kwargs):
        running_meta.update(
            api.get_api_chat_db().get("chat-meta", config.chat_meta_db_name)
        )
        assert api.get_api_chat_db().get("workspace-a", config.user_db_name) == [
            "chat-meta"
        ]
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt=kwargs["user_prompt"],
            assistant_response=_assistant_response(request_goal="metadata goal"),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "chat_id": "chat-meta",
            "workspace_id": "workspace-a",
            "message": "Track this run",
        },
    )

    assert response.status_code == 200
    assert running_meta["latest_thread_id"] == "chat-meta:1"
    assert running_meta["latest_chat_turn"] == 1
    assert running_meta["latest_workflow_status"] == "running"
    assert running_meta["latest_workflow_error"] == ""
    assert running_meta["latest_workflow_message"] == "Track this run"
    assert running_meta["latest_workflow_workspace_id"] == "workspace-a"
    assert isinstance(running_meta["latest_workflow_started_at_ts"], float)
    assert running_meta["latest_workflow_completed_at_ts"] is None

    chat_db = open_chat_db(config)
    meta = chat_db.get("chat-meta", config.chat_meta_db_name)
    assert meta["latest_thread_id"] == "chat-meta:1"
    assert meta["latest_chat_turn"] == 1
    assert meta["latest_workflow_status"] == "completed"
    assert meta["latest_workflow_error"] == ""
    assert isinstance(meta["latest_workflow_started_at_ts"], float)
    assert isinstance(meta["latest_workflow_completed_at_ts"], float)
    assert meta["chat_turn"] == 1


def test_chat_endpoint_stores_failed_workflow_metadata(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)

    def fake_run_chat_turn(**kwargs):
        raise RuntimeError("workflow exploded before completion")

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app, raise_server_exceptions=False)

    response = client.post(
        "/chat",
        json={
            "chat_id": "chat-fail",
            "workspace_id": "workspace-a",
            "message": "This will fail",
        },
    )

    assert response.status_code == 500
    chat_db = open_chat_db(config)
    assert chat_db.get("chat-fail", config.chat_db_name) is None
    meta = chat_db.get("chat-fail", config.chat_meta_db_name)
    assert meta["latest_thread_id"] == "chat-fail:1"
    assert meta["latest_chat_turn"] == 1
    assert meta["latest_workflow_status"] == "failed"
    assert meta["latest_workflow_error"] == "workflow exploded before completion"
    assert meta["latest_workflow_completed_at_ts"] is None
    assert meta["latest_workflow_message"] == "This will fail"
    assert chat_db.get("workspace-a", config.user_db_name) == ["chat-fail"]


def test_resume_endpoint_returns_non_resumable_without_failed_or_running_run(
    monkeypatch,
    tmp_path,
):
    _install_test_api_config(monkeypatch, tmp_path)
    client = TestClient(api.app)

    response = client.post(
        "/chats/chat-no-run/resume",
        json={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "No failed or running workflow run is available to resume."
    }


def test_resume_reuses_thread_id_and_persists_completed_turn_once(
    monkeypatch,
    tmp_path,
):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    chat_db.put(
        "chat-resume",
        [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ],
        config.chat_db_name,
    )
    chat_db.put(
        "chat-resume",
        {
            "title": "Existing title",
            "last_request_goal": "previous goal",
            "chat_turn": 1,
            "latest_thread_id": "chat-resume:2",
            "latest_chat_turn": 2,
            "latest_workflow_status": "failed",
            "latest_workflow_error": "transient failure",
            "latest_workflow_started_at_ts": 123.0,
            "latest_workflow_completed_at_ts": None,
            "latest_workflow_message": "Resume this turn",
            "latest_workflow_workspace_id": "workspace-a",
        },
        config.chat_meta_db_name,
    )
    calls = []

    def fake_run_chat_turn(**kwargs):
        calls.append(kwargs)
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt=kwargs["user_prompt"],
            assistant_response=_assistant_response(
                request_goal="resumed goal",
                image_results=[_image_result("resumed")],
                retrieval=_retrieval_bundle("resumed"),
                image_query="resumed image query",
            ),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chats/chat-resume/resume",
        json={"workspace_id": "workspace-a"},
    )
    second_response = client.post(
        "/chats/chat-resume/resume",
        json={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 200
    assert second_response.status_code == 409
    assert len(calls) == 1
    assert calls[0]["thread_id"] == "chat-resume:2"
    assert calls[0]["chat_turn"] == 2
    assert calls[0]["user_prompt"] == "Resume this turn"
    assert calls[0]["last_request_goal"] == "previous goal"
    assert calls[0]["resume_from_checkpoint"] is True
    assert len(calls[0]["history"]) == 2
    assert response.json()["chat_turn"] == 2
    assert response.json()["message"] == "Resume this turn"

    stored_messages = chat_db.get("chat-resume", config.chat_db_name)
    assert stored_messages == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Resume this turn"},
        {"role": "assistant", "content": "Assistant rule for resumed goal."},
    ]
    meta = chat_db.get("chat-resume", config.chat_meta_db_name)
    assert meta["chat_turn"] == 2
    assert meta["last_request_goal"] == "resumed goal"
    assert meta["latest_thread_id"] == "chat-resume:2"
    assert meta["latest_chat_turn"] == 2
    assert meta["latest_workflow_status"] == "completed"
    assert meta["latest_workflow_error"] == ""
    assert meta["turn_artifacts"]["2"]["chat_turn"] == 2
    assert meta["turn_artifacts"]["2"]["image_query"] == "resumed image query"
    assert meta["turn_artifacts"]["2"]["image_results"][0]["title"] == (
        "Visual resumed"
    )
    assert meta["turn_artifacts"]["2"]["retrieval"]["emerging_rows"][0]["url"] == (
        "https://example.test/resumed/emerging"
    )


def test_resume_does_not_duplicate_completed_turns(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    completed_messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    chat_db.put("chat-already-done", completed_messages, config.chat_db_name)
    chat_db.put(
        "chat-already-done",
        {
            "chat_turn": 1,
            "latest_thread_id": "chat-already-done:1",
            "latest_chat_turn": 1,
            "latest_workflow_status": "failed",
            "latest_workflow_error": "stale status",
            "latest_workflow_message": "Question",
            "latest_workflow_workspace_id": "workspace-a",
        },
        config.chat_meta_db_name,
    )
    calls = []

    def fake_run_chat_turn(**kwargs):
        calls.append(kwargs)
        raise AssertionError("completed turns should not be resumed")

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chats/chat-already-done/resume",
        json={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 409
    assert calls == []
    assert chat_db.get("chat-already-done", config.chat_db_name) == completed_messages
    meta = chat_db.get("chat-already-done", config.chat_meta_db_name)
    assert meta["latest_workflow_status"] == "completed"


def test_chat_endpoint_passes_last_request_goal_and_history(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    chat_db.put(
        "chat-abc",
        [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ],
        config.chat_db_name,
    )
    chat_db.put(
        "chat-abc",
        {"last_request_goal": "existing request goal", "chat_turn": 1},
        config.chat_meta_db_name,
    )
    captured = {}

    def fake_run_chat_turn(**kwargs):
        captured.update(kwargs)
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt=kwargs["user_prompt"],
            assistant_response=_assistant_response(request_goal="new request goal"),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "chat_id": "chat-abc",
            "workspace_id": "workspace-a",
            "message": "Follow-up question",
        },
    )

    assert response.status_code == 200
    assert response.json()["chat_turn"] == 2
    assert captured["last_request_goal"] == "existing request goal"
    assert captured["chat_turn"] == 2
    assert len(captured["history"]) == 2
    assert isinstance(captured["history"][0], ModelRequest)
    assert isinstance(captured["history"][1], ModelResponse)


def test_chat_endpoint_updates_last_request_goal_after_run(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)

    def fake_run_chat_turn(**kwargs):
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt=kwargs["user_prompt"],
            assistant_response=_assistant_response(request_goal="updated goal"),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={"chat_id": "chat-goal", "workspace_id": "workspace-a", "message": "Hi"},
    )

    assert response.status_code == 200
    chat_db = open_chat_db(config)
    meta = chat_db.get("chat-goal", config.chat_meta_db_name)
    assert meta["last_request_goal"] == "updated goal"


def test_chat_endpoint_adds_chat_id_to_workspace_list(monkeypatch, tmp_path):
    config = _install_test_api_config(monkeypatch, tmp_path)
    chat_db = open_chat_db(config)
    chat_db.put("workspace-a", ["existing-chat"], config.user_db_name)

    def fake_run_chat_turn(**kwargs):
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt=kwargs["user_prompt"],
            assistant_response=_assistant_response(),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "chat_id": "new-chat",
            "workspace_id": "workspace-a",
            "message": "Hello",
        },
    )

    assert response.status_code == 200
    assert chat_db.get("workspace-a", config.user_db_name) == [
        "existing-chat",
        "new-chat",
    ]
