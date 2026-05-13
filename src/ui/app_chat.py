import logging

import streamlit as st

from core.app_config import AppConfig
from core.text_utils import normalize_text_nfkc
from services import tracing
from services.chat import generate_chat_title
from services.llm_client import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_OPENROUTER_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
)
from services.chat_workflow import (
    RetrievalBundle,
    generate_assistant_response,
)
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
WORKFLOW_VERSION = "v1"


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
                if item.clip_score is not None:
                    st.caption(f"CLIP similarity: {item.clip_score:.3f}")
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
        tracing.update_current_trace(
            name=f"chat_turn_{st.session_state.chat_turn}",
            user_id=user_id,
            session_id=st.session_state.chat_id,
            input=normalized_prompt,
            tags=[
                *tracing.get_repoa_trace_tags(),
                "chat_turn",
                f"workflow:{WORKFLOW_VERSION}",
            ],
            metadata={
                **tracing.get_repoa_trace_metadata(),
                "chat_id": st.session_state.chat_id,
                "chat_turn": st.session_state.chat_turn,
                "workflow_version": WORKFLOW_VERSION,
            },
        )

        with st.chat_message("user", avatar=None):
            st.markdown(normalized_prompt)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.status("Analyzing...", expanded=False) as status:
                assistant_response = generate_assistant_response(
                    user_prompt=normalized_prompt,
                    config=config,
                    last_request_goal=st.session_state.last_request_goal,
                    history=prior_history,
                    thread_id=(
                        f"{st.session_state.chat_id}:{st.session_state.chat_turn}"
                    ),
                    langfuse_session_id=st.session_state.chat_id,
                    langfuse_user_id=user_id,
                )
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

        tracing.update_current_trace(
            output={
                "rule": assistant_rule,
                "request_goal": request_analysis.request_goal,
                "vertical": request_analysis.vertical,
                "is_in_scope": request_analysis.is_in_scope,
                "candidate_queries": request_analysis.candidate_queries.model_dump(),
                "structured_claims": (
                    assistant_response.structured_claims.model_dump()
                    if assistant_response.structured_claims is not None
                    else None
                ),
                "structured_draft": (
                    assistant_response.structured_draft.model_dump()
                    if assistant_response.structured_draft is not None
                    else None
                ),
                "image_query": assistant_response.image_query,
                "image_results": [
                    item.model_dump() for item in assistant_response.image_results
                ],
                "retrieval": {
                    "canonical_sources": [
                        source.model_dump() for source in retrieval.canonical_sources
                    ],
                    "emerging_sources": [
                        source.model_dump() for source in retrieval.emerging_sources
                    ],
                    "error_message": retrieval.error_message,
                },
            },
            tags=[
                *tracing.get_repoa_trace_tags(),
                "chat_turn",
                f"workflow:{WORKFLOW_VERSION}",
                f"vertical:{request_analysis.vertical}",
                f"in_scope:{request_analysis.is_in_scope}",
            ],
            metadata={
                **tracing.get_repoa_trace_metadata(),
                "chat_turn": st.session_state.chat_turn,
                "request_goal": request_analysis.request_goal,
                "vertical": request_analysis.vertical,
                "is_in_scope": request_analysis.is_in_scope,
                "image_query": assistant_response.image_query,
                "image_result_count": len(assistant_response.image_results),
                "text_retrieval_backend": "tavily",
                "canonical_source_count": len(retrieval.canonical_sources),
                "emerging_source_count": len(retrieval.emerging_sources),
                "total_source_count": retrieval.total_source_count(),
                "canonical_claim_count": (
                    len(assistant_response.structured_claims.canonical_claims)
                    if assistant_response.structured_claims is not None
                    else 0
                ),
                "emerging_claim_count": (
                    len(assistant_response.structured_claims.emerging_claims)
                    if assistant_response.structured_claims is not None
                    else 0
                ),
                "model_params": {
                    "model": DEFAULT_OPENROUTER_MODEL,
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                    "seed": DEFAULT_SEED,
                    "reasoning_effort": DEFAULT_OPENROUTER_REASONING_EFFORT,
                },
            },
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
