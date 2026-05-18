from services.api_models import (
    ChatMessage,
    ChatSessionResponse,
    ChatSummary,
    PersistedTurnArtifacts,
)
from services.chat_workflow import RetrievalBundle
from ui import app_state
from ui.app_state import choose_initial_chat_id, find_latest_resumable_chat_id


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name, value):
        self[name] = value


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()


def test_find_latest_resumable_chat_id_prefers_sorted_unfinished_chat():
    chat_summaries = [
        ChatSummary(
            chat_id="completed-newer",
            updated_at_ts=30.0,
            latest_workflow_status="completed",
        ),
        ChatSummary(
            chat_id="failed-middle",
            updated_at_ts=20.0,
            latest_workflow_status="failed",
        ),
        ChatSummary(
            chat_id="running-older",
            updated_at_ts=10.0,
            latest_workflow_status="running",
        ),
    ]

    assert find_latest_resumable_chat_id(chat_summaries) == "failed-middle"


def test_find_latest_resumable_chat_id_returns_empty_without_unfinished_chat():
    chat_summaries = [
        ChatSummary(chat_id="completed", latest_workflow_status="completed"),
        ChatSummary(chat_id="empty"),
    ]

    assert find_latest_resumable_chat_id(chat_summaries) == ""


def test_choose_initial_chat_id_prefers_unfinished_chat_over_newer_completed():
    chat_summaries = [
        ChatSummary(
            chat_id="completed-newer",
            updated_at_ts=30.0,
            latest_workflow_status="completed",
        ),
        ChatSummary(
            chat_id="failed-middle",
            updated_at_ts=20.0,
            latest_workflow_status="failed",
        ),
    ]

    assert choose_initial_chat_id(chat_summaries) == "failed-middle"


def test_choose_initial_chat_id_uses_latest_existing_chat_when_none_unfinished():
    chat_summaries = [
        ChatSummary(
            chat_id="completed-newer",
            updated_at_ts=30.0,
            latest_workflow_status="completed",
        ),
        ChatSummary(chat_id="older", updated_at_ts=10.0),
    ]

    assert choose_initial_chat_id(chat_summaries) == "completed-newer"


def test_choose_initial_chat_id_returns_empty_without_history():
    assert choose_initial_chat_id([]) == ""


def test_sync_active_chat_session_preserves_turn_artifacts(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_state, "st", fake_st)
    artifact = PersistedTurnArtifacts(
        chat_turn=1,
        retrieval=RetrievalBundle(
            canonical_context="",
            emerging_context="",
            canonical_rows=[
                {
                    "source_id": "C01",
                    "title": "Canonical source",
                    "url": "https://example.test/canonical",
                    "published_at": "2026-01-01",
                    "provider": "tavily",
                }
            ],
        ),
        image_query="visual query",
        request_goal="compare silhouettes",
    )
    chat_session = ChatSessionResponse(
        chat_id="chat-1",
        workspace_id="workspace-a",
        messages=[
            ChatMessage(role="user", content="Question"),
            ChatMessage(role="assistant", content="Answer"),
        ],
        last_request_goal="compare silhouettes",
        chat_turn=1,
        turn_artifacts={"1": artifact},
    )

    app_state.sync_active_chat_session(chat_session)

    assert fake_st.session_state.messages == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    assert fake_st.session_state.turn_artifacts["1"].request_goal == (
        "compare silhouettes"
    )
    assert fake_st.session_state.turn_artifacts["1"].retrieval.canonical_rows[0][
        "url"
    ] == "https://example.test/canonical"
