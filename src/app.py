import logging
import uuid
from pathlib import Path

import streamlit as st

from core.app_config import load_app_config
from ui.app_sidebar import (
    setup_chat_selector,
    setup_vector_search_ui,
)
from ui.app_chat import render_chat_input, render_history
from ui.app_state import (
    get_chat_db,
    get_or_create_anonymous_user_id,
    init_session_state,
    load_active_chat,
    load_user_chat_ids,
    start_new_chat_session,
)
CONFIG = load_app_config()
SRC_ROOT = Path(__file__).resolve().parent


def inject_css_file(path: Path) -> None:
    """Load a CSS file and inject it into the Streamlit app."""
    st.markdown(
        f"<style>{path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run streamlit app."""
    logging.basicConfig(
        level=getattr(logging, CONFIG.app_log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    st.set_page_config(page_title="trend-to-rule", layout="centered")
    inject_css_file(SRC_ROOT / "ui" / "fonts.css")
    inject_css_file(SRC_ROOT / "ui" / "user_avatar.css")
    st.title("trend-to-rule")
    st.caption("Distill trends into rules, grounded in visuals.")

    init_session_state()
    user_id = get_or_create_anonymous_user_id(CONFIG.anonymous_user_query_key)
    if not st.session_state.chat_id:
        st.session_state.chat_id = str(uuid.uuid4())

    chat_db = get_chat_db(
        path=str(CONFIG.db_path),
        db_names=[
            CONFIG.chat_db_name,
            CONFIG.user_db_name,
            CONFIG.chat_meta_db_name,
        ],
    )
    user_chat_ids = load_user_chat_ids(
        user_id=user_id,
        chat_db=chat_db,
        user_db_name=CONFIG.user_db_name,
    )
    setup_chat_selector(
        user_chat_ids=user_chat_ids,
        chat_db=chat_db,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    if (
        st.session_state.pending_delete_chat_id
        and st.session_state.pending_delete_chat_id != st.session_state.chat_id
    ):
        st.session_state.pending_delete_chat_id = ""

    new_chat_col, delete_chat_col = st.sidebar.columns(2)
    if new_chat_col.button("New Chat", use_container_width=True):
        st.session_state.clear()
        init_session_state()
        start_new_chat_session(chat_id=str(uuid.uuid4()))
        st.rerun()

    if delete_chat_col.button("Delete Chat", use_container_width=True):
        st.session_state.pending_delete_chat_id = st.session_state.chat_id
        st.rerun()

    if st.session_state.pending_delete_chat_id:
        st.sidebar.caption("Delete this chat history?")
        confirm_col, cancel_col = st.sidebar.columns(2)
        if confirm_col.button("Confirm", use_container_width=True):
            chat_id_to_delete = st.session_state.pending_delete_chat_id
            chat_db.delete(chat_id_to_delete, CONFIG.chat_db_name)
            chat_db.delete(chat_id_to_delete, CONFIG.chat_meta_db_name)
            remaining_chat_ids = [
                chat_id for chat_id in user_chat_ids if chat_id != chat_id_to_delete
            ]
            chat_db.put(user_id, remaining_chat_ids, CONFIG.user_db_name)
            st.session_state.clear()
            init_session_state()
            start_new_chat_session(chat_id=str(uuid.uuid4()))
            st.rerun()

        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.pending_delete_chat_id = ""
            st.rerun()

    setup_vector_search_ui(
        vector_collection=CONFIG.vector_collection,
        vector_model_name=CONFIG.vector_model_name,
        vector_device=CONFIG.vector_device,
        vector_qdrant_url=CONFIG.vector_qdrant_url,
    )
    load_active_chat(
        chat_db=chat_db,
        chat_db_name=CONFIG.chat_db_name,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    render_history()
    render_chat_input(
        chat_db=chat_db,
        user_chat_ids=user_chat_ids,
        user_id=user_id,
        config=CONFIG,
    )


if __name__ == "__main__":
    main()
