import os
import uuid
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from retrieval.app_retrieval import (
    build_retrieved_results_html_table,
    points_to_prompt_context,
    points_to_table_rows,
    retrieve_vector_results_by_queries,
)
from ui.app_sidebar import (
    setup_chat_selector,
    setup_vector_search_ui
)
from ui.app_state import (
    add_message,
    init_session_state,
    load_active_chat,
    start_new_chat_session,
)
from services.chat import (
    analyze_user_needs,
    extract_structured_draft,
    extract_claims,
    generate_decision_support
)
from storage.chat_db import ChatDB
from core.text_utils import normalize_text_nfkc

# Load Settings
SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
ENV_PATH = SRC_ROOT / ".env"
if ENV_PATH.is_file():
    load_dotenv(ENV_PATH)


def resolve_project_path(raw_path: str) -> Path:
    """Resolve path relative to project root unless absolute."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


# DB Settings
DB_PATH = resolve_project_path(os.getenv("CHAT_DB_PATH", ".data/chat_db"))
CHAT_DB_NAME = "chat_db"
USER_DB_NAME = "user_db"
CHAT_META_DB_NAME = "chat_meta_db"

# Vector Search Settings
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "article_markdown_bge_m3")
VECTOR_MODEL_NAME = os.getenv("VECTOR_MODEL_NAME", "BAAI/bge-m3")
VECTOR_DEVICE = os.getenv("VECTOR_DEVICE", "auto")
VECTOR_CANDIDATE_K = int(os.getenv("VECTOR_CANDIDATE_K", "50"))
_vector_qdrant_path_raw = os.getenv("VECTOR_QDRANT_PATH", "").strip()
VECTOR_QDRANT_PATH = (
    resolve_project_path(
        _vector_qdrant_path_raw) if _vector_qdrant_path_raw else None
)
VECTOR_QDRANT_URL = os.getenv("VECTOR_QDRANT_URL", "http://localhost:6333")
VECTOR_PER_QUERY_TOP_K = int(os.getenv("VECTOR_PER_QUERY_TOP_K", "5"))
VECTOR_MMR_DIVERSITY = float(os.getenv("VECTOR_MMR_DIVERSITY", "0.3"))
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()


def get_chat_db() -> ChatDB:
    """Get chat db instance from session state."""
    if "chat_db" not in st.session_state:
        st.session_state.chat_db = ChatDB(
            path=DB_PATH,
            db_names=[CHAT_DB_NAME, USER_DB_NAME, CHAT_META_DB_NAME],
        )
    return st.session_state.chat_db


def normalize_chat_id_list(raw_ids: list[object]) -> list[str]:
    """Normalize chat id list loaded from db."""
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


def load_user_chat_ids(user_id: str, chat_db: ChatDB) -> list[str]:
    """Load and normalize chat ids for user."""
    user_chat_ids = chat_db.get(key=user_id, db_name=USER_DB_NAME)
    if not isinstance(user_chat_ids, list):
        return []
    normalized_user_chat_ids = normalize_chat_id_list(user_chat_ids)
    if normalized_user_chat_ids != user_chat_ids:
        chat_db.put(key=user_id, value=normalized_user_chat_ids,
                    db_name=USER_DB_NAME)
    return normalized_user_chat_ids


def render_history() -> None:
    """Render stored chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_prompt(
    user_prompt: str,
    chat_db: ChatDB,
    user_chat_ids: list[str],
    user_id: str
) -> None:
    """Process user input and generate assistant response."""
    normalized_prompt = normalize_text_nfkc(user_prompt)

    add_message(
        role="user",
        content=normalized_prompt,
        chat_db=chat_db,
        chat_db_name=CHAT_DB_NAME,
        chat_meta_db_name=CHAT_META_DB_NAME,
    )
    with st.chat_message("user"):
        st.markdown(normalized_prompt)
    with st.status("Analyzing..."):
        user_needs = analyze_user_needs(
            user_prompt=normalized_prompt,
            last_user_goal=st.session_state.last_user_goal,
        )
        st.write(user_needs)

    canonical_points: list[object] = []
    emerging_points: list[object] = []
    canonical_rows: list[dict[str, str]] = []
    emerging_rows: list[dict[str, str]] = []
    try:
        candidate_queries = user_needs.candidate_queries
        retrieved = retrieve_vector_results_by_queries(
            canonical_query=str(candidate_queries.canonical_query or ""),
            emerging_query=str(candidate_queries.emerging_query or ""),
            vector_candidate_k=VECTOR_CANDIDATE_K,
            mmr_diversity=VECTOR_MMR_DIVERSITY,
            per_query_top_k=VECTOR_PER_QUERY_TOP_K,
        )
        canonical_points = retrieved.get("canonical", [])
        emerging_points = retrieved.get("emerging", [])
        canonical_context = points_to_prompt_context(
            canonical_points,
            label="Canonical"
        )
        emerging_context = points_to_prompt_context(
            emerging_points,
            label="Emerging"
        )
        canonical_rows = points_to_table_rows(canonical_points)
        emerging_rows = points_to_table_rows(emerging_points)
    except Exception as err:
        st.warning(f"Vector search failed: {err}")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        structured_claims = extract_claims(
            canonical_context=canonical_context,
            emerging_context=emerging_context,
            user_goal=user_needs.user_goal,
            last_user_goal=st.session_state.last_user_goal,
            history=st.session_state.history
        )
        st.write(structured_claims)
        structured_draft = extract_structured_draft(
            canonical_claims=structured_claims.canonical_claims,
            emerging_claims=structured_claims.emerging_claims,
            user_goal=user_needs.user_goal,
            last_user_goal=st.session_state.last_user_goal,
            history=st.session_state.history,
        )
        st.write(structured_draft)
        resp = generate_decision_support(
            user_prompt=user_prompt,
            user_goal=user_needs.user_goal,
            last_user_goal=st.session_state.last_user_goal,
            history=st.session_state.history,
            structured_draft=structured_draft
        )
        text = normalize_text_nfkc(resp or "")
        buf = ""
        for ch in text:
            buf += ch
            placeholder.markdown(buf)
        if canonical_rows or emerging_rows:
            st.caption("Retrieved vectors used for this answer")
            if canonical_rows:
                st.caption("Canonical query results")
                st.markdown(
                    build_retrieved_results_html_table(canonical_rows),
                    unsafe_allow_html=True,
                )
            if emerging_rows:
                st.caption("Emerging query results")
                st.markdown(
                    build_retrieved_results_html_table(emerging_rows),
                    unsafe_allow_html=True,
                )

    add_message(
        role="assistant",
        content=text,
        chat_db=chat_db,
        chat_db_name=CHAT_DB_NAME,
        chat_meta_db_name=CHAT_META_DB_NAME,
    )
    st.session_state.last_user_goal = user_needs.user_goal
    if st.session_state.chat_id not in user_chat_ids:
        user_chat_ids.append(st.session_state.chat_id)
        chat_db.put(key=user_id, value=user_chat_ids, db_name=USER_DB_NAME)


def render_chat_input(chat_db: ChatDB, user_chat_ids: list[str], user_id: str) -> None:
    """Handle chat input event."""
    user_prompt = st.chat_input("Let's Chat🚀")
    if user_prompt:
        process_user_prompt(
            user_prompt=user_prompt,
            chat_db=chat_db,
            user_chat_ids=user_chat_ids,
            user_id=user_id,
        )


def main(user_id: str) -> None:
    """Run streamlit app."""
    logging.basicConfig(
        level=getattr(logging, APP_LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    st.set_page_config(page_title="trend-to-rule", layout="centered")
    st.title("🧩trend-to-rule")

    init_session_state()
    if not st.session_state.chat_id:
        st.session_state.chat_id = str(uuid.uuid4())

    chat_db = get_chat_db()
    if st.sidebar.button("New Chat", use_container_width=True):
        st.session_state.clear()
        init_session_state()
        start_new_chat_session(
            chat_id=str(uuid.uuid4())
        )
        st.rerun()

    user_chat_ids = load_user_chat_ids(
        user_id=user_id,
        chat_db=chat_db
    )
    setup_chat_selector(
        user_chat_ids=user_chat_ids,
        chat_db=chat_db,
        chat_meta_db_name=CHAT_META_DB_NAME,
    )
    setup_vector_search_ui(
        vector_collection=VECTOR_COLLECTION,
        vector_model_name=VECTOR_MODEL_NAME,
        vector_device=VECTOR_DEVICE,
        vector_qdrant_path=VECTOR_QDRANT_PATH,
        vector_qdrant_url=VECTOR_QDRANT_URL,
    )
    load_active_chat(
        chat_db=chat_db,
        chat_db_name=CHAT_DB_NAME
    )
    render_history()
    render_chat_input(
        chat_db=chat_db,
        user_chat_ids=user_chat_ids,
        user_id=user_id
    )


if __name__ == "__main__":
    try:
        current_user = os.getlogin()
    except OSError:
        current_user = os.getenv("USER", "unknown")
    main(user_id=current_user)
