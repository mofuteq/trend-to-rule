"""Typed request/response models for the FastAPI chat boundary."""

from typing import Literal

from pydantic import BaseModel, Field

from services.chat_workflow import AssistantResponseBundle, RetrievalBundle
from services.image_search import ImageSearchResult


def _empty_retrieval_bundle() -> RetrievalBundle:
    return RetrievalBundle(canonical_context="", emerging_context="")


def _display_retrieval_bundle(retrieval: RetrievalBundle) -> RetrievalBundle:
    return RetrievalBundle(
        canonical_context="",
        emerging_context="",
        canonical_rows=[dict(row) for row in retrieval.canonical_rows],
        emerging_rows=[dict(row) for row in retrieval.emerging_rows],
        error_message=retrieval.error_message,
    )


class ChatMessage(BaseModel):
    """Stored chat message shape shared by API and Streamlit."""

    role: Literal["user", "assistant"]
    content: str


class PersistedTurnArtifacts(BaseModel):
    """Display artifacts persisted for a completed assistant turn."""

    chat_turn: int = Field(ge=1)
    image_results: list[ImageSearchResult] = Field(default_factory=list)
    retrieval: RetrievalBundle = Field(default_factory=_empty_retrieval_bundle)
    image_query: str = ""
    request_goal: str = ""
    is_in_scope: bool = True

    @classmethod
    def from_assistant_response(
        cls,
        *,
        chat_turn: int,
        assistant_response: AssistantResponseBundle,
    ) -> "PersistedTurnArtifacts":
        """Build a display-oriented persisted artifact bundle."""
        request_analysis = assistant_response.request_analysis
        return cls(
            chat_turn=chat_turn,
            image_results=list(assistant_response.image_results),
            retrieval=_display_retrieval_bundle(assistant_response.retrieval),
            image_query=assistant_response.image_query,
            request_goal=request_analysis.request_goal,
            is_in_scope=request_analysis.is_in_scope,
        )


ChatTurnArtifactMap = dict[str, PersistedTurnArtifacts]


class ChatSummary(BaseModel):
    """Sidebar-ready summary for one stored chat."""

    chat_id: str
    title: str = ""
    updated_at_ts: float | None = None
    latest_workflow_status: str = ""
    latest_chat_turn: int | None = None
    latest_thread_id: str = ""
    latest_workflow_error: str = ""


class ListChatsResponse(BaseModel):
    """Stored chat summaries for one workspace."""

    workspace_id: str
    chats: list[ChatSummary]


class ChatSessionResponse(BaseModel):
    """Stored chat session loaded from the backend."""

    chat_id: str
    workspace_id: str
    messages: list[ChatMessage]
    title: str = ""
    last_request_goal: str = ""
    chat_turn: int = 0
    latest_workflow_status: str = ""
    latest_chat_turn: int | None = None
    latest_thread_id: str = ""
    latest_workflow_error: str = ""
    turn_artifacts: ChatTurnArtifactMap = Field(default_factory=dict)


class CreateChatRequest(BaseModel):
    """Create or initialize an empty chat in a workspace."""

    workspace_id: str
    chat_id: str | None = None


class CreateChatResponse(ChatSessionResponse):
    """Created or initialized chat session."""


class DeleteChatResponse(BaseModel):
    """Result of deleting a chat from a workspace."""

    deleted_chat_id: str
    workspace_id: str
    remaining_chat_ids: list[str]
    chats: list[ChatSummary]


class ChatRequest(BaseModel):
    """Minimal chat request accepted by the API boundary."""

    chat_id: str
    message: str
    workspace_id: str | None = None


class ResumeChatRequest(BaseModel):
    """Resume the latest checkpoint-backed workflow run for a chat."""

    workspace_id: str | None = None
    chat_turn: int | None = None
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response data needed by a UI to render the assistant turn."""

    chat_id: str
    workspace_id: str
    chat_turn: int
    message: str
    assistant_response: AssistantResponseBundle
    title: str = ""


class WorkflowStreamEvent(BaseModel):
    """Display-safe workflow event streamed by the chat SSE endpoint."""

    event_type: Literal[
        "task_started",
        "task_completed",
        "task_failed",
        "checkpoint",
        "progress_summary",
        "final_response",
        "error",
    ]
    node: str | None = None
    label: str
    chat_id: str | None = None
    chat_turn: int | None = None
    thread_id: str | None = None
    next_nodes: list[str] = Field(default_factory=list)
    error: str = ""
    response: ChatResponse | None = None
