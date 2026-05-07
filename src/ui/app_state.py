from datetime import datetime, timezone

import streamlit as st
from google.genai.types import Content, Part

from storage.chat_db import ChatDB


def init_session_state() -> None:
    """Initialize streamlit session state keys."""
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_user_goal", "")
    st.session_state.setdefault("chat_id", "")
    st.session_state.setdefault("loaded_chat_id", "")
    st.session_state.setdefault("pending_delete_chat_id", "")
    st.session_state.setdefault("chat_turn", 0)
    st.session_state.setdefault("vector_searcher", None)
    st.session_state.setdefault("vector_error", "")
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
    st.session_state.last_user_goal = ""
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


def normalize_chat_id_list(raw_ids: list[object]) -> list[str]:
    """Normalize chat id list loaded from storage.

    Args:
        raw_ids: Raw value loaded from `user_db`.

    Returns:
        list[str]: Deduplicated normalized chat ids.
    """
    normalized: list[str] = []
    for raw_id in raw_ids:
        if isinstance(raw_id, str):
            if raw_id and raw_id not in normalized:
                normalized.append(raw_id)
            continue
        if isinstance(raw_id, list):
            for nested_id in raw_id:
                if isinstance(nested_id, str) and nested_id and nested_id not in normalized:
                    normalized.append(nested_id)
    return normalized


def load_user_chat_ids(user_id: str, chat_db: ChatDB, user_db_name: str) -> list[str]:
    """Load and normalize chat ids for one user.

    Args:
        user_id: Current user id.
        chat_db: Chat database instance.
        user_db_name: Database name for user/chat mappings.

    Returns:
        list[str]: User chat ids sorted by stored order.
    """
    user_chat_ids = chat_db.get(key=user_id, db_name=user_db_name)
    if not isinstance(user_chat_ids, list):
        return []
    normalized_user_chat_ids = normalize_chat_id_list(user_chat_ids)
    if normalized_user_chat_ids != user_chat_ids:
        chat_db.put(key=user_id, value=normalized_user_chat_ids, db_name=user_db_name)
    return normalized_user_chat_ids


def sort_chat_ids_by_last_updated(
    chat_ids: list[str],
    chat_db: ChatDB,
    chat_meta_db_name: str,
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Sort chat ids by last updated timestamp in descending order.

    Args:
        chat_ids: Chat ids to sort.
        chat_db: Chat database instance.
        chat_meta_db_name: Database name for chat metadata.

    Returns:
        tuple[list[str], dict[str, float], dict[str, str]]:
            Sorted chat ids, updated timestamps by chat id, and titles by chat id.
    """
    updated_at_by_chat: dict[str, float] = {}
    title_by_chat: dict[str, str] = {}
    for chat_id in chat_ids:
        meta = get_chat_meta(
            chat_id=chat_id,
            chat_db=chat_db,
            chat_meta_db_name=chat_meta_db_name,
        )
        updated_at_ts = meta.get("updated_at_ts")
        updated_at_by_chat[chat_id] = (
            float(updated_at_ts) if isinstance(updated_at_ts, (int, float)) else 0.0
        )
        title_by_chat[chat_id] = str(meta.get("title") or "").strip()

    sorted_chat_ids = sorted(
        chat_ids,
        key=lambda chat_id: updated_at_by_chat.get(chat_id, 0.0),
        reverse=True,
    )
    return sorted_chat_ids, updated_at_by_chat, title_by_chat


def start_new_chat_session(chat_id: str) -> None:
    """Reset chat-related session state for a new conversation."""
    st.session_state.chat_id = chat_id
    st.session_state.loaded_chat_id = ""
    st.session_state.last_user_goal = ""
    st.session_state.chat_turn = 0
    st.session_state.messages = []
    st.session_state.history = []


def load_active_chat(chat_db: ChatDB, chat_db_name: str, chat_meta_db_name: str) -> None:
    """Load active chat messages from db into session state."""
    if st.session_state.loaded_chat_id != st.session_state.chat_id:
        saved_messages = chat_db.get(
            key=st.session_state.chat_id,
            db_name=chat_db_name,
        )
        meta = get_chat_meta(
            chat_id=st.session_state.chat_id,
            chat_db=chat_db,
            chat_meta_db_name=chat_meta_db_name,
        )
        st.session_state.last_user_goal = str(meta.get("last_user_goal") or "")
        if isinstance(saved_messages, list):
            st.session_state.messages = saved_messages
            st.session_state.history = [
                Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[Part(text=msg["content"])],
                )
                for msg in saved_messages
            ]
        else:
            st.session_state.messages = []
            st.session_state.history = []
        st.session_state.loaded_chat_id = st.session_state.chat_id


def set_last_user_goal(
    chat_id: str,
    last_user_goal: str,
    chat_db: ChatDB,
    chat_meta_db_name: str,
) -> None:
    """Persist the latest inferred user goal to chat metadata.

    Args:
        chat_id: Chat id to update.
        last_user_goal: Inferred user goal from the completed turn.
        chat_db: Chat database instance.
        chat_meta_db_name: Database name for chat metadata.
    """
    meta = get_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
    meta["last_user_goal"] = last_user_goal
    chat_db.put(key=chat_id, value=meta, db_name=chat_meta_db_name)


def update_chat_meta(chat_id: str, chat_db: ChatDB, chat_meta_db_name: str) -> None:
    """Update chat metadata while preserving existing fields."""
    existing_meta = chat_db.get(
        key=chat_id,
        db_name=chat_meta_db_name,
    )
    meta = existing_meta if isinstance(existing_meta, dict) else {}
    meta["updated_at_ts"] = datetime.now(tz=timezone.utc).timestamp()
    chat_db.put(
        key=chat_id,
        value=meta,
        db_name=chat_meta_db_name,
    )


def get_chat_meta(chat_id: str, chat_db: ChatDB, chat_meta_db_name: str) -> dict:
    """Return chat metadata as a dict.

    Args:
        chat_id: Chat id to load.
        chat_db: Chat database instance.
        chat_meta_db_name: Database name for chat metadata.

    Returns:
        dict: Chat metadata or an empty dict.
    """
    meta = chat_db.get(key=chat_id, db_name=chat_meta_db_name)
    return meta if isinstance(meta, dict) else {}


def set_chat_title(
    chat_id: str,
    title: str,
    chat_db: ChatDB,
    chat_meta_db_name: str,
) -> None:
    """Set chat title while preserving existing metadata.

    Args:
        chat_id: Chat id to update.
        title: Generated chat title.
        chat_db: Chat database instance.
        chat_meta_db_name: Database name for chat metadata.
    """
    meta = get_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
    meta["title"] = title
    chat_db.put(
        key=chat_id,
        value=meta,
        db_name=chat_meta_db_name,
    )


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
        [
            Content(role="user", parts=[Part(text=user_content)]),
            Content(role="model", parts=[Part(text=assistant_content)]),
        ]
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
