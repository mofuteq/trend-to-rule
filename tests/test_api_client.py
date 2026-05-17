from pathlib import Path
import json

import httpx

from core.app_config import AppConfig
from services import api_client


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
        api_base_url="http://api.test",
    )


def _install_mock_transport(monkeypatch, calls, payloads):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.content:
            body = json.loads(request.content.decode())
        else:
            body = None
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "body": body,
            }
        )
        return httpx.Response(
            200,
            json=payloads[(request.method, request.url.path)],
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*, base_url, timeout):
        calls.append({"base_url": base_url, "timeout": timeout})
        return real_client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    monkeypatch.setattr(api_client.httpx, "Client", fake_client)


def test_list_chats_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_mock_transport(
        monkeypatch,
        calls,
        {
            ("GET", "/chats"): {
                "workspace_id": "workspace-a",
                "chats": [
                    {
                        "chat_id": "chat-1",
                        "title": "Chat one",
                        "updated_at_ts": 10.0,
                    }
                ],
            }
        },
    )

    response = api_client.list_chats(
        workspace_id="workspace-a",
        config=_make_config(tmp_path),
    )

    assert response.workspace_id == "workspace-a"
    assert response.chats[0].chat_id == "chat-1"
    assert response.chats[0].title == "Chat one"
    assert calls == [
        {"base_url": "http://api.test", "timeout": 120.0},
        {
            "method": "GET",
            "url": "http://api.test/chats?workspace_id=workspace-a",
            "body": None,
        },
    ]


def test_get_chat_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_mock_transport(
        monkeypatch,
        calls,
        {
            ("GET", "/chats/chat-1"): {
                "chat_id": "chat-1",
                "workspace_id": "workspace-a",
                "messages": [{"role": "user", "content": "Hello"}],
                "title": "Chat one",
                "last_request_goal": "goal",
                "chat_turn": 1,
            }
        },
    )

    response = api_client.get_chat(
        chat_id="chat-1",
        workspace_id="workspace-a",
        config=_make_config(tmp_path),
    )

    assert response.chat_id == "chat-1"
    assert response.messages[0].content == "Hello"
    assert response.last_request_goal == "goal"
    assert calls == [
        {"base_url": "http://api.test", "timeout": 120.0},
        {
            "method": "GET",
            "url": "http://api.test/chats/chat-1?workspace_id=workspace-a",
            "body": None,
        },
    ]


def test_create_chat_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_mock_transport(
        monkeypatch,
        calls,
        {
            ("POST", "/chats"): {
                "chat_id": "chat-1",
                "workspace_id": "workspace-a",
                "messages": [],
                "title": "",
                "last_request_goal": "",
                "chat_turn": 0,
            }
        },
    )

    response = api_client.create_chat(
        chat_id="chat-1",
        workspace_id="workspace-a",
        config=_make_config(tmp_path),
    )

    assert response.chat_id == "chat-1"
    assert response.messages == []
    assert calls == [
        {"base_url": "http://api.test", "timeout": 120.0},
        {
            "method": "POST",
            "url": "http://api.test/chats",
            "body": {"workspace_id": "workspace-a", "chat_id": "chat-1"},
        },
    ]


def test_delete_chat_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_mock_transport(
        monkeypatch,
        calls,
        {
            ("DELETE", "/chats/chat-1"): {
                "deleted_chat_id": "chat-1",
                "workspace_id": "workspace-a",
                "remaining_chat_ids": ["chat-2"],
                "chats": [
                    {
                        "chat_id": "chat-2",
                        "title": "Chat two",
                        "updated_at_ts": 20.0,
                    }
                ],
            }
        },
    )

    response = api_client.delete_chat(
        chat_id="chat-1",
        workspace_id="workspace-a",
        config=_make_config(tmp_path),
    )

    assert response.deleted_chat_id == "chat-1"
    assert response.remaining_chat_ids == ["chat-2"]
    assert response.chats[0].chat_id == "chat-2"
    assert calls == [
        {"base_url": "http://api.test", "timeout": 120.0},
        {
            "method": "DELETE",
            "url": "http://api.test/chats/chat-1?workspace_id=workspace-a",
            "body": None,
        },
    ]


def test_post_chat_turn_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_mock_transport(
        monkeypatch,
        calls,
        {
            ("POST", "/chat"): {
                "chat_id": "chat-123",
                "workspace_id": "workspace-a",
                "chat_turn": 3,
                "message": "What denim shapes are trending?",
                "title": "Denim title",
                "assistant_response": {
                    "request_analysis": {
                        "request_goal": "compare denim silhouettes",
                        "candidate_queries": {
                            "canonical_query": "classic denim silhouettes",
                            "emerging_query": "emerging denim silhouettes",
                        },
                        "vertical": "womens",
                        "is_in_scope": True,
                    },
                    "rule": "Assistant rule.",
                    "image_query": "wide leg denim outfits",
                    "image_results": [],
                },
            }
        },
    )

    response = api_client.post_chat_turn(
        chat_id="chat-123",
        workspace_id="workspace-a",
        message="What denim shapes are trending?",
        config=_make_config(tmp_path),
    )

    assert response.chat_id == "chat-123"
    assert response.workspace_id == "workspace-a"
    assert response.chat_turn == 3
    assert response.title == "Denim title"
    assert response.assistant_response.rule == "Assistant rule."
    assert calls == [
        {"base_url": "http://api.test", "timeout": 120.0},
        {
            "method": "POST",
            "url": "http://api.test/chat",
            "body": {
                "chat_id": "chat-123",
                "workspace_id": "workspace-a",
                "message": "What denim shapes are trending?",
            },
        },
    ]
