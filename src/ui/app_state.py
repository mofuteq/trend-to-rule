from datetime import datetime, timezone
from typing import Literal

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


def get_or_create_anonymous_user_id(query_key: str) -> str:
    """Get anonymous user id from query params or create one.

    Args:
        query_key: Query parameter key used to persist the anonymous id.

    Returns:
        str: Stable anonymous user id for the current browser session.
    """
    current_user_id = str(st.session_state.get("user_id") or "").strip()
    if current_user_id:
        return current_user_id

    query_user_id = st.query_params.get(query_key, "")
    if isinstance(query_user_id, list):
        query_user_id = query_user_id[0] if query_user_id else ""
    query_user_id = str(query_user_id or "").strip()
    if not query_user_id:
        import uuid

        query_user_id = f"anon-{uuid.uuid4()}"
        st.query_params[query_key] = query_user_id

    st.session_state.user_id = query_user_id
    return query_user_id


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


def start_new_chat_session(chat_id: str) -> None:
    """Reset chat-related session state for a new conversation."""
    st.session_state.chat_id = chat_id
    st.session_state.loaded_chat_id = ""
    st.session_state.last_user_goal = ""
    st.session_state.messages = []
    st.session_state.history = []


def load_active_chat(chat_db: ChatDB, chat_db_name: str) -> None:
    """Load active chat messages from db into session state."""
    if st.session_state.loaded_chat_id != st.session_state.chat_id:
        saved_messages = chat_db.get(
            key=st.session_state.chat_id,
            db_name=chat_db_name,
        )
        st.session_state.last_user_goal = ""
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


def update_chat_meta(chat_id: str, chat_db: ChatDB, chat_meta_db_name: str) -> None:
    """Update chat modified time."""
    chat_db.put(
        key=chat_id,
        value={"updated_at_ts": datetime.now(tz=timezone.utc).timestamp()},
        db_name=chat_meta_db_name,
    )


def add_message(
    role: Literal["user", "assistant"],
    content: str,
    chat_db: ChatDB,
    chat_db_name: str,
    chat_meta_db_name: str,
) -> None:
    """Append message to session and persist to db."""
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.history.append(
        Content(
            role="user" if role == "user" else "model",
            parts=[Part(text=content)],
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
