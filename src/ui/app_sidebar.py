from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from storage.chat_db import ChatDB

try:
    from retrieval.search_vectors import HybridVectorSearcher
except Exception:
    HybridVectorSearcher = None  # type: ignore[assignment]


def setup_chat_selector(
    user_chat_ids: list[str],
    chat_db: ChatDB,
    chat_meta_db_name: str,
) -> None:
    """Render sidebar chat selector."""
    ids = list(user_chat_ids)
    if st.session_state.chat_id and st.session_state.chat_id not in ids:
        ids.insert(0, st.session_state.chat_id)

    updated_at_by_chat: dict[str, float] = {}
    for chat_id in ids:
        updated_at = 0.0
        meta = chat_db.get(key=chat_id, db_name=chat_meta_db_name)
        if isinstance(meta, dict):
            updated_at_ts = meta.get("updated_at_ts")
            if isinstance(updated_at_ts, (int, float)):
                updated_at = float(updated_at_ts)
        updated_at_by_chat[chat_id] = updated_at

    sorted_chat_ids = sorted(
        ids,
        key=lambda chat_id: updated_at_by_chat.get(chat_id, 0.0),
        reverse=True,
    )
    if not sorted_chat_ids:
        st.sidebar.caption("No chat history yet.")
        return

    selected_index = sorted_chat_ids.index(st.session_state.chat_id) if st.session_state.chat_id in sorted_chat_ids else 0
    selected_chat_id = st.sidebar.selectbox(
        label="chat_id",
        options=sorted_chat_ids,
        index=selected_index,
        format_func=lambda chat_id: (
            f"{chat_id} (new)"
            if updated_at_by_chat.get(chat_id, 0.0) <= 0
            and chat_id not in user_chat_ids
            else
            f"{chat_id} ({datetime.fromtimestamp(updated_at_by_chat.get(chat_id, 0.0), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})"
            if updated_at_by_chat.get(chat_id, 0.0) > 0
            else chat_id
        ),
    )
    st.session_state.chat_id = str(selected_chat_id)


def setup_vector_search_ui(
    *,
    vector_collection: str,
    vector_model_name: str,
    vector_device: str,
    vector_qdrant_path: Path | None,
    vector_qdrant_url: str,
) -> None:
    """Initialize vector search lazily (always enabled)."""
    st.sidebar.divider()

    if HybridVectorSearcher is None:
        st.session_state.vector_error = "Vector search dependencies are not available."
        st.sidebar.error(st.session_state.vector_error)
        return

    st.sidebar.caption("Vector search: always on")

    if st.session_state.vector_searcher is None:
        try:
            st.session_state.vector_searcher = HybridVectorSearcher(
                model_name=vector_model_name,
                device=vector_device,
                collection=vector_collection,
                qdrant_path=vector_qdrant_path,
                qdrant_url=vector_qdrant_url,
            )
            st.session_state.vector_error = ""
        except Exception as err:
            st.session_state.vector_error = str(err)

    if st.session_state.vector_error:
        st.sidebar.error(st.session_state.vector_error)
