from pathlib import Path

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse

import api
from core.app_config import AppConfig
from core.models import RequestAnalysis, SearchQuery
from services.chat_runtime import ChatTurnResult
from services.chat_session import open_chat_db
from services.chat_workflow import AssistantResponseBundle


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


def _assistant_response(request_goal: str = "compare denim silhouettes"):
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
        image_query="wide leg denim outfits",
        image_results=[],
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
                request_goal=f"goal {kwargs['chat_turn']}"
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

