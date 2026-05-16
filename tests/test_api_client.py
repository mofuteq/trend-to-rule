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


def test_post_chat_turn_parses_response(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
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

    class FakeClient:
        def __init__(self, *, base_url, timeout):
            calls.append({"base_url": base_url, "timeout": timeout})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, path, *, json):
            calls.append({"path": path, "json": json})
            return FakeResponse()

    monkeypatch.setattr(api_client.httpx, "Client", FakeClient)

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
            "path": "/chat",
            "json": {
                "chat_id": "chat-123",
                "workspace_id": "workspace-a",
                "message": "What denim shapes are trending?",
            },
        },
    ]

