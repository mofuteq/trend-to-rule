from fastapi.testclient import TestClient

import api
from core.models import RequestAnalysis, SearchQuery
from services.chat_runtime import ChatTurnResult
from services.chat_workflow import AssistantResponseBundle


def _assistant_response() -> AssistantResponseBundle:
    return AssistantResponseBundle(
        request_analysis=RequestAnalysis(
            request_goal="compare denim silhouettes",
            candidate_queries=SearchQuery(
                canonical_query="classic denim silhouettes",
                emerging_query="emerging denim silhouettes",
            ),
            vertical="womens",
            is_in_scope=True,
        ),
        rule="Assistant rule.",
        image_query="wide leg denim outfits",
        image_results=[],
    )


def test_health_endpoint_returns_ok():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_uses_shared_runtime_shape(monkeypatch):
    calls = []

    def fake_run_chat_turn(**kwargs):
        calls.append(kwargs)
        return ChatTurnResult(
            chat_id=kwargs["chat_id"],
            user_id=kwargs["user_id"],
            chat_turn=kwargs["chat_turn"],
            normalized_prompt="What denim shapes are trending?",
            assistant_response=_assistant_response(),
        )

    monkeypatch.setattr(api, "run_chat_turn", fake_run_chat_turn)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "chat_id": "chat-123",
            "workspace_id": "workspace-a",
            "message": "What denim shapes are trending?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] == "chat-123"
    assert payload["workspace_id"] == "workspace-a"
    assert payload["message"] == "What denim shapes are trending?"
    assert payload["assistant_response"]["rule"] == "Assistant rule."
    assert payload["assistant_response"]["request_analysis"] == {
        "request_goal": "compare denim silhouettes",
        "candidate_queries": {
            "canonical_query": "classic denim silhouettes",
            "emerging_query": "emerging denim silhouettes",
        },
        "vertical": "womens",
        "is_in_scope": True,
    }
    assert payload["assistant_response"]["image_query"] == "wide leg denim outfits"
    assert payload["assistant_response"]["image_results"] == []

    assert len(calls) == 1
    call = calls[0]
    assert call["user_prompt"] == "What denim shapes are trending?"
    assert call["chat_id"] == "chat-123"
    assert call["user_id"] == "workspace-a"
    assert call["config"] is api.CONFIG
    assert call["chat_turn"] == 1
    assert call["last_request_goal"] is None
    assert call["history"] == []
    assert call["thread_id"].startswith("chat-123:api:")

