from services.api_models import ChatSummary
from ui.app_state import find_latest_resumable_chat_id


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
