import streamlit as st

from core.app_config import AppConfig
from core.text_utils import normalize_text_nfkc
from services.api_client import stream_chat_turn
from services.api_models import (
    ChatResponse,
    PersistedTurnArtifacts,
    WorkflowStreamEvent,
)
from services.chat_workflow import RetrievalBundle
from services.image_search import ImageSearchResult
from services.web_search import build_web_sources_html_table

ASSISTANT_AVATAR = ":material/auto_awesome:"


def render_history() -> None:
    """Render stored chat history."""
    assistant_turn = 0
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            assistant_turn += 1
            with st.chat_message(message["role"], avatar=ASSISTANT_AVATAR):
                st.markdown(message["content"])
                artifact = get_persisted_turn_artifact(assistant_turn)
                if artifact is not None:
                    render_turn_artifacts(artifact)
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


def render_turn_artifacts(artifacts: PersistedTurnArtifacts) -> None:
    """Render persisted non-text artifacts for one completed assistant turn."""
    if artifacts.is_in_scope:
        render_image_results(artifacts.image_results)
    render_retrieved_results(artifacts.retrieval)


def render_response_artifacts(response: ChatResponse) -> None:
    """Render non-text artifacts carried by a completed chat response."""
    render_turn_artifacts(
        PersistedTurnArtifacts.from_assistant_response(
            chat_turn=response.chat_turn,
            assistant_response=response.assistant_response,
        )
    )


def get_persisted_turn_artifact(chat_turn: int) -> PersistedTurnArtifacts | None:
    """Return persisted artifacts for a turn from Streamlit session state."""
    raw_artifacts = st.session_state.get("turn_artifacts", {})
    if not isinstance(raw_artifacts, dict):
        return None
    raw_artifact = raw_artifacts.get(str(chat_turn)) or raw_artifacts.get(chat_turn)
    if raw_artifact is None:
        return None
    try:
        return PersistedTurnArtifacts.model_validate(raw_artifact)
    except Exception:
        return None


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
    if st.session_state.messages[-2:] != turn_messages and (
        st.session_state.messages
        and st.session_state.messages[-1] == turn_messages[0]
    ):
        st.session_state.messages.append(turn_messages[1])
    elif st.session_state.messages[-2:] != turn_messages:
        st.session_state.messages.extend(turn_messages)
    st.session_state.chat_turn = response.chat_turn
    st.session_state.last_request_goal = (
        response.assistant_response.request_analysis.request_goal
    )
    turn_artifacts = st.session_state.get("turn_artifacts", {})
    if not isinstance(turn_artifacts, dict):
        turn_artifacts = {}
    turn_artifacts[str(response.chat_turn)] = (
        PersistedTurnArtifacts.from_assistant_response(
            chat_turn=response.chat_turn,
            assistant_response=response.assistant_response,
        )
    )
    st.session_state.turn_artifacts = turn_artifacts


def update_workflow_status_from_event(status, event: WorkflowStreamEvent) -> None:
    """Update the existing Streamlit status component from one workflow event."""
    label = event.label or "Running workflow..."
    if event.event_type == "error" or event.event_type == "task_failed":
        status.update(label=label, state="error", expanded=False)
    elif event.event_type == "final_response":
        status.update(label=label, state="complete", expanded=False)
    else:
        status.update(label=label, expanded=False)


def stream_chat_response(
    *,
    chat_id: str,
    workspace_id: str,
    message: str,
    config: AppConfig,
    status,
) -> ChatResponse:
    """Consume workflow SSE events and return the final chat response."""
    last_error = ""
    for event in stream_chat_turn(
        chat_id=chat_id,
        workspace_id=workspace_id,
        message=message,
        config=config,
    ):
        update_workflow_status_from_event(status, event)
        if event.event_type == "final_response" and event.response is not None:
            return event.response
        if event.event_type == "error":
            last_error = event.error
    raise RuntimeError(last_error or "Workflow stream ended without a response.")


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
        with st.status("Running workflow...", expanded=False) as status:
            response = stream_chat_response(
                chat_id=st.session_state.chat_id,
                workspace_id=user_id,
                message=normalized_prompt,
                config=config,
                status=status,
            )
            assistant_rule = response.assistant_response.rule

        stream_markdown_text(assistant_rule)
        render_response_artifacts(response)

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
