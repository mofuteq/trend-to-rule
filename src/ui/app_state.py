from datetime import datetime, timezone
from typing import Literal

import streamlit as st
from google.genai.types import Content, Part

from storage.chat_db import ChatDB


def init_session_state() -> None:
    """Initialize streamlit session state keys."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_user_goal", "")
    st.session_state.setdefault("chat_id", "")
    st.session_state.setdefault("loaded_chat_id", "")
    st.session_state.setdefault("vector_searcher", None)
    st.session_state.setdefault("vector_error", "")
    if not isinstance(st.session_state.chat_id, str):
        st.session_state.chat_id = str(st.session_state.chat_id)


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
