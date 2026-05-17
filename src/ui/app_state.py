import streamlit as st

from services.api_models import ChatMessage, ChatSessionResponse, ChatSummary

RESUMABLE_WORKFLOW_STATUSES = {"running", "failed"}


def init_session_state() -> None:
    """Initialize streamlit session state keys."""
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_request_goal", "")
    if (
        not st.session_state.last_request_goal
        and st.session_state.get("last_user_goal")
    ):
        st.session_state.last_request_goal = str(st.session_state.last_user_goal)
    st.session_state.setdefault("chat_id", "")
    st.session_state.setdefault("loaded_chat_id", "")
    st.session_state.setdefault("pending_delete_chat_id", "")
    st.session_state.setdefault("chat_turn", 0)
    st.session_state.setdefault("latest_workflow_status", "")
    st.session_state.setdefault("latest_chat_turn", None)
    st.session_state.setdefault("latest_thread_id", "")
    st.session_state.setdefault("latest_workflow_error", "")
    st.session_state.setdefault("attempted_auto_resume_thread_ids", [])
    if not isinstance(st.session_state.chat_id, str):
        st.session_state.chat_id = str(st.session_state.chat_id)
    attempted_thread_ids = st.session_state.attempted_auto_resume_thread_ids
    if isinstance(attempted_thread_ids, (set, tuple)):
        st.session_state.attempted_auto_resume_thread_ids = list(attempted_thread_ids)
    elif not isinstance(attempted_thread_ids, list):
        st.session_state.attempted_auto_resume_thread_ids = []


def reset_chat_selection() -> None:
    """Reset chat selection state after switching workspace."""
    st.session_state.chat_id = ""
    st.session_state.loaded_chat_id = ""
    st.session_state.last_request_goal = ""
    st.session_state.chat_turn = 0
    st.session_state.messages = []
    st.session_state.pending_delete_chat_id = ""
    reset_workflow_state()


def get_workspace_user_id(query_key: str, default_workspace_key: str) -> str:
    """Get the active workspace key from sidebar input and query params.

    Args:
        query_key: Query parameter key used to persist the workspace key.
        default_workspace_key: Workspace key used when none is provided.

    Returns:
        str: Active workspace key used as the chat-history user id.
    """
    query_workspace = st.query_params.get(query_key, "")
    if isinstance(query_workspace, list):
        query_workspace = query_workspace[0] if query_workspace else ""
    query_workspace = str(query_workspace or "").strip()

    fallback_workspace = str(default_workspace_key or "demo").strip() or "demo"
    current_workspace = str(st.session_state.get("user_id") or "").strip()
    initial_workspace = current_workspace or query_workspace or fallback_workspace

    workspace_key = st.sidebar.text_input(
        "Workspace key",
        value=initial_workspace,
        help="Use the same key later to reopen this workspace's chat history.",
    ).strip()
    if not workspace_key:
        workspace_key = fallback_workspace

    if workspace_key != current_workspace:
        reset_chat_selection()
        st.session_state.user_id = workspace_key
    else:
        st.session_state.user_id = workspace_key

    if query_workspace != workspace_key:
        st.query_params[query_key] = workspace_key
    return workspace_key


def start_new_chat_session(chat_id: str) -> None:
    """Reset chat-related session state for a new conversation."""
    st.session_state.chat_id = chat_id
    st.session_state.loaded_chat_id = ""
    st.session_state.last_request_goal = ""
    st.session_state.chat_turn = 0
    st.session_state.messages = []
    reset_workflow_state()


def sync_active_chat_session(chat_session: ChatSessionResponse) -> None:
    """Copy a backend-owned chat session into Streamlit display state."""
    st.session_state.chat_id = chat_session.chat_id
    st.session_state.loaded_chat_id = chat_session.chat_id
    st.session_state.last_request_goal = chat_session.last_request_goal
    st.session_state.chat_turn = chat_session.chat_turn
    st.session_state.messages = [
        _message_to_dict(message) for message in chat_session.messages
    ]
    st.session_state.latest_workflow_status = chat_session.latest_workflow_status
    st.session_state.latest_chat_turn = chat_session.latest_chat_turn
    st.session_state.latest_thread_id = chat_session.latest_thread_id
    st.session_state.latest_workflow_error = chat_session.latest_workflow_error


def reset_workflow_state() -> None:
    """Clear active workflow metadata from Streamlit session state."""
    st.session_state.latest_workflow_status = ""
    st.session_state.latest_chat_turn = None
    st.session_state.latest_thread_id = ""
    st.session_state.latest_workflow_error = ""


def find_latest_resumable_chat_id(chat_summaries: list[ChatSummary]) -> str:
    """Return the most recently updated chat id with an unfinished workflow."""
    for summary in chat_summaries:
        if is_resumable_workflow_status(summary.latest_workflow_status):
            return summary.chat_id
    return ""


def is_resumable_workflow_status(status: str) -> bool:
    """Return True when a workflow status can be auto-resumed."""
    return status in RESUMABLE_WORKFLOW_STATUSES


def has_attempted_auto_resume(thread_id: str) -> bool:
    """Return True if this Streamlit session already tried a thread id."""
    if not thread_id:
        return False
    return thread_id in st.session_state.attempted_auto_resume_thread_ids


def mark_auto_resume_attempted(thread_id: str) -> None:
    """Remember that auto-resume has been attempted for a thread id."""
    if not thread_id or has_attempted_auto_resume(thread_id):
        return
    st.session_state.attempted_auto_resume_thread_ids.append(thread_id)


def _message_to_dict(message: ChatMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}
