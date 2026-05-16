"""Minimal FastAPI boundary for the existing chat workflow."""

import uuid
from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.app_config import load_app_config  # noqa: E402
from services import tracing  # noqa: E402
from services.chat_runtime import ChatTurnResult, run_chat_turn  # noqa: E402
from services.chat_workflow import AssistantResponseBundle  # noqa: E402

CONFIG = load_app_config()

app = FastAPI(title="trend-to-rule API")


class ChatRequest(BaseModel):
    """Minimal chat request accepted by the API boundary."""

    chat_id: str
    message: str
    workspace_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response data needed by a UI to render the assistant turn."""

    chat_id: str
    workspace_id: str
    message: str
    assistant_response: AssistantResponseBundle


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@tracing.observe(name="chat_turn")
def _run_observed_chat_request(
    request: ChatRequest,
    *,
    workspace_id: str,
) -> ChatTurnResult:
    return run_chat_turn(
        user_prompt=request.message,
        chat_id=request.chat_id,
        user_id=workspace_id,
        config=CONFIG,
        chat_turn=1,
        last_request_goal=None,
        history=[],
        thread_id=f"{request.chat_id}:api:{uuid.uuid4()}",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the existing LangGraph chat workflow for one HTTP request."""
    workspace_id = request.workspace_id or CONFIG.default_workspace_key
    try:
        result = _run_observed_chat_request(
            request,
            workspace_id=workspace_id,
        )
        return ChatResponse(
            chat_id=request.chat_id,
            workspace_id=workspace_id,
            message=result.normalized_prompt,
            assistant_response=result.assistant_response,
        )
    finally:
        tracing.flush()
