import streamlit as st

from services.chat_session import (
    get_chat_meta,
    load_chat_messages,
    load_user_chat_ids,
    messages_to_model_history,
    normalize_chat_id_list,
    set_chat_title,
    set_last_request_goal,
    sort_chat_ids_by_last_updated,
    update_chat_meta,
)
from storage.chat_db import ChatDB


def init_session_state() -> None:
    """Initialize streamlit session state keys."""
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
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
    if not isinstance(st.session_state.chat_id, str):
        st.session_state.chat_id = str(st.session_state.chat_id)


def get_chat_db(
    *,
    path: str,
    db_names: list[str],
) -> ChatDB:
    """Get chat db instance from session state.

    Args:
        path: Chat DB filesystem path.
        db_names: LMDB database names to open.

    Returns:
        ChatDB: Cached chat database instance.
    """
    if "chat_db" not in st.session_state:
        st.session_state.chat_db = ChatDB(path=path, db_names=db_names)
    return st.session_state.chat_db


def reset_chat_selection() -> None:
    """Reset chat selection state after switching workspace."""
    st.session_state.chat_id = ""
    st.session_state.loaded_chat_id = ""
    st.session_state.last_request_goal = ""
    st.session_state.chat_turn = 0
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.pending_delete_chat_id = ""


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
    st.session_state.history = []


def load_active_chat(chat_db: ChatDB, chat_db_name: str, chat_meta_db_name: str) -> None:
    """Load active chat messages from db into session state."""
    if st.session_state.loaded_chat_id != st.session_state.chat_id:
        saved_messages = load_chat_messages(
            chat_id=st.session_state.chat_id,
            chat_db=chat_db,
            chat_db_name=chat_db_name,
        )
        meta = get_chat_meta(
            chat_id=st.session_state.chat_id,
            chat_db=chat_db,
            chat_meta_db_name=chat_meta_db_name,
        )
        st.session_state.last_request_goal = str(
            meta.get("last_request_goal") or meta.get("last_user_goal") or ""
        )
        st.session_state.messages = saved_messages
        st.session_state.history = messages_to_model_history(saved_messages)
        st.session_state.loaded_chat_id = st.session_state.chat_id


def add_turn(
    user_content: str,
    assistant_content: str,
    chat_db: ChatDB,
    chat_db_name: str,
    chat_meta_db_name: str,
) -> None:
    """Append a completed user/assistant turn and persist it to db."""
    st.session_state.messages.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    )
    st.session_state.history.extend(
        messages_to_model_history(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        )
    )
    chat_db.put(
        key=st.session_state.chat_id,
        value=st.session_state.messages,
        db_name=chat_db_name,
    )
    update_chat_meta(
        chat_id=st.session_state.chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
