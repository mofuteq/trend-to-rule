from pathlib import Path

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


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _install_fake_client(monkeypatch, calls, payloads):
    class FakeClient:
        def __init__(self, *, base_url, timeout):
            calls.append({"base_url": base_url, "timeout": timeout})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path, *, params):
            calls.append({"method": "GET", "path": path, "params": params})
            return FakeResponse(payloads[("GET", path)])

        def post(self, path, *, json):
            calls.append({"method": "POST", "path": path, "json": json})
            return FakeResponse(payloads[("POST", path)])

        def delete(self, path, *, params):
            calls.append({"method": "DELETE", "path": path, "params": params})
            return FakeResponse(payloads[("DELETE", path)])

    monkeypatch.setattr(api_client.httpx, "Client", FakeClient)


def test_list_chats_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_fake_client(
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
    assert calls[-1] == {
        "method": "GET",
        "path": "/chats",
        "params": {"workspace_id": "workspace-a"},
    }


def test_get_chat_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_fake_client(
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
    assert calls[-1] == {
        "method": "GET",
        "path": "/chats/chat-1",
        "params": {"workspace_id": "workspace-a"},
    }


def test_create_chat_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_fake_client(
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
    assert calls[-1] == {
        "method": "POST",
        "path": "/chats",
        "json": {"workspace_id": "workspace-a", "chat_id": "chat-1"},
    }


def test_delete_chat_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_fake_client(
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
    assert calls[-1] == {
        "method": "DELETE",
        "path": "/chats/chat-1",
        "params": {"workspace_id": "workspace-a"},
    }


def test_post_chat_turn_parses_response(monkeypatch, tmp_path):
    calls = []
    _install_fake_client(
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
    assert calls[-1] == {
        "method": "POST",
        "path": "/chat",
        "json": {
            "chat_id": "chat-123",
            "workspace_id": "workspace-a",
            "message": "What denim shapes are trending?",
        },
    }

