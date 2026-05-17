import streamlit as st

from core.app_config import AppConfig
from core.text_utils import normalize_text_nfkc
from services.api_client import post_chat_turn
from services.api_models import ChatResponse
from services.chat_workflow import RetrievalBundle
from services.image_search import ImageSearchResult
from services.web_search import build_web_sources_html_table

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


def sync_rendered_turn(response: ChatResponse) -> None:
    """Reflect an API-owned completed turn in Streamlit session state."""
    turn_messages = [
        {"role": "user", "content": response.message},
        {"role": "assistant", "content": response.assistant_response.rule},
    ]
    st.session_state.messages.extend(turn_messages)
    st.session_state.chat_turn = response.chat_turn
    st.session_state.last_request_goal = (
        response.assistant_response.request_analysis.request_goal
    )


def process_user_prompt(
    user_prompt: str,
    *,
    user_id: str,
    config: AppConfig,
) -> None:
    """Process user input and generate the assistant response.

    Args:
        user_prompt: Raw user prompt.
        user_id: Current workspace key.
        config: App runtime config.
    """
    normalized_prompt = normalize_text_nfkc(user_prompt)
    with st.chat_message("user", avatar=None):
        st.markdown(normalized_prompt)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.status("Analyzing...", expanded=False) as status:
            response = post_chat_turn(
                chat_id=st.session_state.chat_id,
                workspace_id=user_id,
                message=normalized_prompt,
                config=config,
            )
            assistant_response = response.assistant_response
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

    sync_rendered_turn(response)


def render_chat_input(
    *,
    user_id: str,
    config: AppConfig,
) -> None:
    """Handle the chat input interaction.

    Args:
        user_id: Current workspace key.
        config: App runtime config.
    """
    user_prompt = st.chat_input("Start with a question")
    if user_prompt:
        process_user_prompt(
            user_prompt=user_prompt,
            user_id=user_id,
            config=config,
        )
