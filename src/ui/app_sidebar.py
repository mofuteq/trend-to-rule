from datetime import datetime, timezone

import streamlit as st

from services.api_models import ChatSummary


def setup_chat_selector(chat_summaries: list[ChatSummary]) -> None:
    """Render sidebar chat selector."""
    if not chat_summaries:
        st.sidebar.caption("No chat history yet.")
        return

    chat_ids = [summary.chat_id for summary in chat_summaries]
    selected_index = (
        chat_ids.index(st.session_state.chat_id)
        if st.session_state.chat_id in chat_ids
        else 0
    )
    selected_chat_id = st.sidebar.selectbox(
        label="History",
        options=chat_ids,
        index=selected_index,
        format_func=_format_chat_summary(chat_summaries),
    )
    st.session_state.chat_id = str(selected_chat_id)


def _format_chat_summary(chat_summaries: list[ChatSummary]):
    summary_by_id = {summary.chat_id: summary for summary in chat_summaries}

    def _format(chat_id: str) -> str:
        summary = summary_by_id[chat_id]
        label = summary.title or chat_id
        if summary.updated_at_ts is None:
            return f"{label} (new)"
        updated_at = datetime.fromtimestamp(
            summary.updated_at_ts,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"{label} ({updated_at})"

    return _format
