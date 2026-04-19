import logging

import streamlit as st

from core.app_config import AppConfig
from core.text_utils import normalize_text_nfkc
from retrieval.app_retrieval import build_retrieved_results_html_table
from services import tracing
from services.chat import analyze_user_needs, generate_chat_title
from services.chat_workflow import (
    RetrievalBundle,
    generate_assistant_response,
    retrieve_supporting_context,
)
from services.image_search import ImageSearchResult
from storage.chat_db import ChatDB
from ui.app_state import add_message, get_chat_meta, set_chat_title

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
                if item.thumbnail_url or item.image_url:
                    image_src = item.thumbnail_url or item.image_url
                    if item.image_url:
                        st.markdown(
                            (
                                f'<a href="{item.image_url}" target="_blank" rel="noopener noreferrer">'
                                f'<img src="{image_src}" style="width: 100%; border-radius: 0.5rem;" />'
                                "</a>"
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.image(
                            image_src,
                            use_container_width=True,
                        )
                links: list[str] = []


def render_retrieved_results(retrieval: RetrievalBundle) -> None:
    """Render retrieved canonical and emerging vector search tables.

    Args:
        retrieval: Retrieved vector-search bundle for the current turn.
    """
    if not retrieval.canonical_rows and not retrieval.emerging_rows:
        return
    st.caption("Retrieved vectors used for this answer")
    if retrieval.canonical_rows:
        st.caption("Canonical query results")
        st.markdown(
            build_retrieved_results_html_table(retrieval.canonical_rows),
            unsafe_allow_html=True,
        )
    if retrieval.emerging_rows:
        st.caption("Emerging query results")
        st.markdown(
            build_retrieved_results_html_table(retrieval.emerging_rows),
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
        user_id: Current anonymous user id.
        config: App runtime config.
    """
    normalized_prompt = normalize_text_nfkc(user_prompt)
    tracing.update_current_trace(
        user_id=user_id,
        session_id=st.session_state.chat_id,
        input=normalized_prompt,
        tags=["trend-to-rule", "chat_turn", f"workflow:{WORKFLOW_VERSION}"],
        metadata={
            "chat_id": st.session_state.chat_id,
            "workflow_version": WORKFLOW_VERSION,
        },
    )

    add_message(
        role="user",
        content=normalized_prompt,
        chat_db=chat_db,
        chat_db_name=config.chat_db_name,
        chat_meta_db_name=config.chat_meta_db_name,
    )
    with st.chat_message("user", avatar=None):
        st.markdown(normalized_prompt)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.status("Analyzing...", expanded=False) as status:
            user_needs = analyze_user_needs(
                user_prompt=normalized_prompt,
                last_user_goal=st.session_state.last_user_goal,
            )
            st.write(user_needs)
            status.update(label="Retrieving context...", expanded=False)

            retrieval = RetrievalBundle(
                canonical_context="",
                emerging_context="",
                canonical_rows=[],
                emerging_rows=[],
            )
            try:
                retrieval = retrieve_supporting_context(
                    user_needs,
                    config=config,
                )
            except Exception as err:
                st.warning(f"Vector search failed: {err}")

            status.update(label="Thinking...", expanded=False)
            assistant_response = generate_assistant_response(
                user_prompt=user_prompt,
                user_needs=user_needs,
                retrieval=retrieval,
                config=config,
                last_user_goal=st.session_state.last_user_goal,
                history=st.session_state.history,
            )
            st.write(assistant_response.structured_claims)
            st.write(assistant_response.structured_draft)
            st.write(assistant_response.image_query)
            status.update(label="Thinking complete", state="complete", expanded=False)

        stream_markdown_text(assistant_response.rule)
        render_image_results(assistant_response.image_results)
        render_retrieved_results(retrieval)

    add_message(
        role="assistant",
        content=assistant_response.rule,
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
    st.session_state.last_user_goal = user_needs.user_goal
    if st.session_state.chat_id not in user_chat_ids:
        user_chat_ids.append(st.session_state.chat_id)
        chat_db.put(key=user_id, value=user_chat_ids,
                    db_name=config.user_db_name)

    tracing.update_current_trace(
        output={
            "rule": assistant_response.rule,
            "user_goal": user_needs.user_goal,
            "vertical": user_needs.vertical,
            "candidate_queries": user_needs.candidate_queries.model_dump(),
            "structured_claims": assistant_response.structured_claims.model_dump(),
            "structured_draft": assistant_response.structured_draft.model_dump(),
            "image_query": assistant_response.image_query,
            "image_results": [
                item.model_dump() for item in assistant_response.image_results
            ],
            "retrieval": {
                "canonical_rows": retrieval.canonical_rows,
                "emerging_rows": retrieval.emerging_rows,
            },
        },
        tags=[
            "trend-to-rule",
            "chat_turn",
            f"workflow:{WORKFLOW_VERSION}",
            f"vertical:{user_needs.vertical}",
        ],
        metadata={
            "user_goal": user_needs.user_goal,
            "vertical": user_needs.vertical,
            "image_query": assistant_response.image_query,
            "image_result_count": len(assistant_response.image_results),
            "canonical_hits": len(retrieval.canonical_rows),
            "emerging_hits": len(retrieval.emerging_rows),
            "canonical_claim_count": len(
                assistant_response.structured_claims.canonical_claims
            ),
            "emerging_claim_count": len(
                assistant_response.structured_claims.emerging_claims
            ),
        },
    )


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
        user_id: Current anonymous user id.
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
