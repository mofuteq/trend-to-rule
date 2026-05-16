"""Typed request/response models for the FastAPI chat boundary."""

from pydantic import BaseModel

from services.chat_workflow import AssistantResponseBundle


class ChatRequest(BaseModel):
    """Minimal chat request accepted by the API boundary."""

    chat_id: str
    message: str
    workspace_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response data needed by a UI to render the assistant turn."""

    chat_id: str
    workspace_id: str
    chat_turn: int
    message: str
    assistant_response: AssistantResponseBundle
    title: str = ""

