import logging

import streamlit as st

from core.app_config import AppConfig
from core.text_utils import normalize_text_nfkc
from services import tracing
from services.chat import generate_chat_title
from services.chat_runtime import run_chat_turn
from services.chat_workflow import RetrievalBundle
from services.image_search import ImageSearchResult
from services.web_search import build_web_sources_html_table
from storage.chat_db import ChatDB
from ui.app_state import (
    add_turn,
    get_chat_meta,
    set_chat_title,
    set_last_request_goal,
)

logger = logging.getLogger(__name__)
ASSISTANT_AVATAR = ":material/auto_awesome:"


def render_history() -> None:
    """Render stored chat history."""
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=ASSISTANT_AVATAR):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar=None):
                st.markdown(message["content"])


def render_image_results(image_results: list[ImageSearchResult]) -> None:
    """Render image results as a simple card grid.

    Args:
        image_results: Deduplicated image search results.
    """
    if not image_results:
        return
    st.caption("Visual references")
    num_columns = min(3, max(1, len(image_results)))
    columns = st.columns(num_columns)
    for idx, item in enumerate(image_results):
        with columns[idx % num_columns]:
            with st.container(border=True):
                title = item.title or "Untitled"
                if item.page_url:
                    st.markdown(f"**[{title}]({item.page_url})**")
                else:
                    st.markdown(f"**{title}**")
                if item.thumbnail_url or item.image_url:
                    image_src = item.thumbnail_url or item.image_url
                    if item.image_url:
                        st.markdown(
                            (
                                f'<a href="{item.image_url}"'
                                ' target="_blank" rel="noopener noreferrer">'
                                f'<img src="{image_src}"'
                                ' style="width: 100%; border-radius: 0.5rem;" />'
                                "</a>"
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.image(
                            image_src,
                            use_container_width=True,
                        )


def render_retrieved_results(retrieval: RetrievalBundle) -> None:
    """Render retrieved canonical and emerging web evidence tables.

    Args:
        retrieval: Retrieved web evidence bundle for the current turn.
    """
    if not retrieval.canonical_rows and not retrieval.emerging_rows:
        return
    st.caption("Web evidence used for this answer")
    if retrieval.canonical_rows:
        st.caption("Canonical sources")
        st.markdown(
            build_web_sources_html_table(retrieval.canonical_rows),
            unsafe_allow_html=True,
        )
    if retrieval.emerging_rows:
        st.caption("Emerging sources")
        st.markdown(
            build_web_sources_html_table(retrieval.emerging_rows),
            unsafe_allow_html=True,
        )


def stream_markdown_text(text: str) -> None:
    """Stream markdown text into the current chat message placeholder.

    Args:
        text: Text to render incrementally.
    """
    placeholder = st.empty()
    buf = ""
    for ch in text:
        buf += ch
        placeholder.markdown(buf)


@tracing.observe(name="chat_turn")
def process_user_prompt(
    user_prompt: str,
    *,
    chat_db: ChatDB,
    user_chat_ids: list[str],
    user_id: str,
    config: AppConfig,
) -> None:
    """Process user input and generate the assistant response.

    Args:
        user_prompt: Raw user prompt.
        chat_db: Chat database instance.
        user_chat_ids: Current user's chat id list.
        user_id: Current workspace key.
        config: App runtime config.
    """
    st.session_state.chat_turn += 1
    normalized_prompt = normalize_text_nfkc(user_prompt)
    prior_history = list(st.session_state.history)
    try:
        with st.chat_message("user", avatar=None):
            st.markdown(normalized_prompt)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.status("Analyzing...", expanded=False) as status:
                chat_turn_result = run_chat_turn(
                    user_prompt=normalized_prompt,
                    chat_id=st.session_state.chat_id,
                    user_id=user_id,
                    config=config,
                    chat_turn=st.session_state.chat_turn,
                    last_request_goal=st.session_state.last_request_goal,
                    history=prior_history,
                )
                assistant_response = chat_turn_result.assistant_response
                request_analysis = assistant_response.request_analysis
                retrieval = assistant_response.retrieval
                assistant_rule = assistant_response.rule
                st.write(request_analysis)
                if not request_analysis.is_in_scope:
                    status.update(
                        label="Outside app scope", state="complete", expanded=False
                    )
                else:
                    st.write(assistant_response.structured_claims)
                    st.write(assistant_response.structured_draft)
                    st.write(assistant_response.image_query)
                    status.update(
                        label="Thinking complete", state="complete", expanded=False
                    )

            stream_markdown_text(assistant_rule)
            if request_analysis.is_in_scope:
                render_image_results(assistant_response.image_results)
            render_retrieved_results(retrieval)

        add_turn(
            user_content=normalized_prompt,
            assistant_content=assistant_rule,
            chat_db=chat_db,
            chat_db_name=config.chat_db_name,
            chat_meta_db_name=config.chat_meta_db_name,
        )
        chat_meta = get_chat_meta(
            chat_id=st.session_state.chat_id,
            chat_db=chat_db,
            chat_meta_db_name=config.chat_meta_db_name,
        )
        if not str(chat_meta.get("title") or "").strip():
            try:
                generated_title = generate_chat_title(
                    messages=st.session_state.messages,
                )
                if generated_title:
                    set_chat_title(
                        chat_id=st.session_state.chat_id,
                        title=generated_title,
                        chat_db=chat_db,
                        chat_meta_db_name=config.chat_meta_db_name,
                    )
            except Exception as err:
                logger.warning("Failed to generate chat title: %s", err)
        st.session_state.last_request_goal = request_analysis.request_goal
        set_last_request_goal(
            chat_id=st.session_state.chat_id,
            last_request_goal=request_analysis.request_goal,
            chat_db=chat_db,
            chat_meta_db_name=config.chat_meta_db_name,
        )
        if st.session_state.chat_id not in user_chat_ids:
            user_chat_ids.append(st.session_state.chat_id)
            chat_db.put(
                key=user_id,
                value=user_chat_ids,
                db_name=config.user_db_name,
            )
    finally:
        tracing.flush()


def render_chat_input(
    *,
    chat_db: ChatDB,
    user_chat_ids: list[str],
    user_id: str,
    config: AppConfig,
) -> None:
    """Handle the chat input interaction.

    Args:
        chat_db: Chat database instance.
        user_chat_ids: Current user's chat id list.
        user_id: Current workspace key.
        config: App runtime config.
    """
    user_prompt = st.chat_input("Start with a question")
    if user_prompt:
        process_user_prompt(
            user_prompt=user_prompt,
            chat_db=chat_db,
            user_chat_ids=user_chat_ids,
            user_id=user_id,
            config=config,
        )
