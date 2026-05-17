from services.api_models import ChatSummary
from ui.app_state import choose_initial_chat_id, find_latest_resumable_chat_id


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
