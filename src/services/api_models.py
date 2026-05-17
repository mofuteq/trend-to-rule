"""Typed request/response models for the FastAPI chat boundary."""

from typing import Literal

from pydantic import BaseModel

from services.chat_workflow import AssistantResponseBundle


class ChatMessage(BaseModel):
    """Stored chat message shape shared by API and Streamlit."""

    role: Literal["user", "assistant"]
    content: str


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
