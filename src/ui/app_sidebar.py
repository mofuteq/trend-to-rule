from datetime import datetime, timezone
import inspect
from pathlib import Path

import streamlit as st

from storage.chat_db import ChatDB
from ui.app_state import sort_chat_ids_by_last_updated

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

    sorted_chat_ids, updated_at_by_chat, title_by_chat = sort_chat_ids_by_last_updated(
        chat_ids=ids,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
    if not sorted_chat_ids:
        st.sidebar.caption("No chat history yet.")
        return

    selected_index = sorted_chat_ids.index(st.session_state.chat_id) if st.session_state.chat_id in sorted_chat_ids else 0
    selected_chat_id = st.sidebar.selectbox(
        label="History",
        options=sorted_chat_ids,
        index=selected_index,
        format_func=lambda chat_id: (
            f"{title_by_chat.get(chat_id) or chat_id} (new)"
            if updated_at_by_chat.get(chat_id, 0.0) <= 0
            and chat_id not in user_chat_ids
            else
            f"{title_by_chat.get(chat_id) or chat_id} ({datetime.fromtimestamp(updated_at_by_chat.get(chat_id, 0.0), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})"
            if updated_at_by_chat.get(chat_id, 0.0) > 0
            else title_by_chat.get(chat_id) or chat_id
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

    needs_reinit = st.session_state.vector_searcher is None
    if not needs_reinit and st.session_state.vector_searcher is not None:
        try:
            signature = inspect.signature(
                st.session_state.vector_searcher.hybrid_search_with_filter
            )
            needs_reinit = "vertical_match_value" not in signature.parameters
        except (TypeError, ValueError, AttributeError):
            needs_reinit = True

    if needs_reinit:
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
    else:
        st.sidebar.caption(
            "AI-generated content may contain mistakes. Please verify important details with the original sources."
        )
