import logging
import uuid
from pathlib import Path

import streamlit as st

from core.app_config import load_app_config
from services.api_client import (
    create_chat,
    delete_chat,
    get_chat,
    list_chats,
    resume_chat,
)
from services.api_models import ChatResponse
from ui.app_sidebar import setup_chat_selector
from ui.app_chat import (
    render_chat_input,
    render_history,
    render_response_artifacts,
    sync_rendered_turn,
)
from ui.app_state import (
    choose_initial_chat_id,
    get_workspace_user_id,
    has_attempted_auto_resume,
    init_session_state,
    is_resumable_workflow_status,
    mark_auto_resume_attempted,
    start_new_chat_session,
    sync_active_chat_session,
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
    user_id = get_workspace_user_id(
        query_key=CONFIG.workspace_query_key,
        default_workspace_key=CONFIG.default_workspace_key,
    )

    chat_summaries = list_chats(
        workspace_id=user_id,
        config=CONFIG,
    ).chats
    if not st.session_state.chat_id:
        initial_chat_id = choose_initial_chat_id(chat_summaries)
        if initial_chat_id:
            start_new_chat_session(chat_id=initial_chat_id)
        else:
            created_chat = create_chat(
                workspace_id=user_id,
                chat_id=str(uuid.uuid4()),
                config=CONFIG,
            )
            start_new_chat_session(chat_id=created_chat.chat_id)
            chat_summaries = list_chats(
                workspace_id=user_id,
                config=CONFIG,
            ).chats

    active_chat_id = st.session_state.chat_id
    active_known_chat_ids = {summary.chat_id for summary in chat_summaries}
    if active_chat_id and active_chat_id not in active_known_chat_ids:
        create_chat(
            workspace_id=user_id,
            chat_id=active_chat_id,
            config=CONFIG,
        )
        chat_summaries = list_chats(
            workspace_id=user_id,
            config=CONFIG,
        ).chats

    setup_chat_selector(chat_summaries)
    if st.session_state.loaded_chat_id != st.session_state.chat_id:
        st.session_state.messages = []
    if (
        st.session_state.pending_delete_chat_id
        and st.session_state.pending_delete_chat_id != st.session_state.chat_id
    ):
        st.session_state.pending_delete_chat_id = ""

    new_chat_col, delete_chat_col = st.sidebar.columns(2)
    if new_chat_col.button("New Chat", use_container_width=True):
        created_chat = create_chat(
            workspace_id=user_id,
            chat_id=str(uuid.uuid4()),
            config=CONFIG,
        )
        st.session_state.clear()
        init_session_state()
        start_new_chat_session(chat_id=created_chat.chat_id)
        st.rerun()

    if delete_chat_col.button("Delete Chat", use_container_width=True):
        st.session_state.pending_delete_chat_id = st.session_state.chat_id
        st.rerun()

    if st.session_state.pending_delete_chat_id:
        st.sidebar.caption("Delete this chat history?")
        confirm_col, cancel_col = st.sidebar.columns(2)
        if confirm_col.button("Confirm", use_container_width=True):
            chat_id_to_delete = st.session_state.pending_delete_chat_id
            delete_response = delete_chat(
                chat_id=chat_id_to_delete,
                workspace_id=user_id,
                config=CONFIG,
            )
            if delete_response.chats:
                next_chat_id = delete_response.chats[0].chat_id
            else:
                created_chat = create_chat(
                    workspace_id=user_id,
                    chat_id=str(uuid.uuid4()),
                    config=CONFIG,
                )
                next_chat_id = created_chat.chat_id
            st.session_state.clear()
            init_session_state()
            start_new_chat_session(chat_id=next_chat_id)
            st.rerun()

        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.pending_delete_chat_id = ""
            st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Evidence retrieval: Tavily web search")
    st.sidebar.caption(
        "Patterns, not persuasion. This agent organizes signals without steering "
        "your decision."
    )
    if st.session_state.loaded_chat_id != st.session_state.chat_id:
        sync_active_chat_session(
            get_chat(
                chat_id=st.session_state.chat_id,
                workspace_id=user_id,
                config=CONFIG,
            )
        )
    auto_resumed_response = maybe_auto_resume_active_chat(user_id=user_id, config=CONFIG)
    render_history()
    if auto_resumed_response is not None:
        render_response_artifacts(auto_resumed_response)
    render_chat_input(
        user_id=user_id,
        config=CONFIG,
    )


def maybe_auto_resume_active_chat(*, user_id: str, config) -> ChatResponse | None:
    """Resume one unfinished workflow run once per Streamlit session."""
    thread_id = str(st.session_state.get("latest_thread_id") or "")
    if not is_resumable_workflow_status(
        str(st.session_state.get("latest_workflow_status") or "")
    ):
        return None
    if not thread_id or has_attempted_auto_resume(thread_id):
        return None

    mark_auto_resume_attempted(thread_id)
    try:
        response = resume_chat(
            chat_id=st.session_state.chat_id,
            workspace_id=user_id,
            config=config,
            chat_turn=st.session_state.latest_chat_turn,
            thread_id=thread_id,
        )
    except Exception:
        st.error("Could not resume the unfinished response. Please try again later.")
        return None

    sync_rendered_turn(response)
    sync_active_chat_session(
        get_chat(
            chat_id=st.session_state.chat_id,
            workspace_id=user_id,
            config=config,
        )
    )
    return response


if __name__ == "__main__":
    main()
