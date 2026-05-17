"""Minimal FastAPI boundary for the existing chat workflow."""

import logging
from pathlib import Path
import sys
import uuid

from fastapi import FastAPI
from pydantic_ai.messages import ModelMessage

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.app_config import load_app_config  # noqa: E402
from services import tracing  # noqa: E402
from services.api_models import (  # noqa: E402
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
    CreateChatRequest,
    CreateChatResponse,
    DeleteChatResponse,
    ListChatsResponse,
)
from services.chat import generate_chat_title  # noqa: E402
from services.chat_runtime import ChatTurnResult, run_chat_turn  # noqa: E402
from services.chat_session import (  # noqa: E402
    LoadedChatSession,
    append_chat_turn,
    delete_chat_session,
    ensure_user_chat_id,
    initialize_chat_session,
    list_chat_summaries,
    load_chat_session,
    open_chat_db,
    set_chat_title,
    set_last_request_goal,
)
from storage.chat_db import ChatDB  # noqa: E402

CONFIG = load_app_config()
_CHAT_DB: ChatDB | None = None
logger = logging.getLogger(__name__)

app = FastAPI(title="trend-to-rule API")


def get_api_chat_db() -> ChatDB:
    """Return the API process chat database handle."""
    global _CHAT_DB
    if _CHAT_DB is None:
        _CHAT_DB = open_chat_db(CONFIG)
    return _CHAT_DB


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.get("/chats", response_model=ListChatsResponse)
def list_chats(workspace_id: str | None = None) -> ListChatsResponse:
    """Return stored chats for a workspace."""
    resolved_workspace_id = workspace_id or CONFIG.default_workspace_key
    chat_db = get_api_chat_db()
    return ListChatsResponse(
        workspace_id=resolved_workspace_id,
        chats=list_chat_summaries(
            user_id=resolved_workspace_id,
            chat_db=chat_db,
            user_db_name=CONFIG.user_db_name,
            chat_meta_db_name=CONFIG.chat_meta_db_name,
        ),
    )


@app.post("/chats", response_model=CreateChatResponse)
def create_chat(request: CreateChatRequest) -> CreateChatResponse:
    """Create or initialize an empty chat for a workspace."""
    chat_id = request.chat_id or str(uuid.uuid4())
    chat_db = get_api_chat_db()
    session = initialize_chat_session(
        user_id=request.workspace_id,
        chat_id=chat_id,
        chat_db=chat_db,
        chat_db_name=CONFIG.chat_db_name,
        user_db_name=CONFIG.user_db_name,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    return _chat_session_response(
        chat_id=chat_id,
        workspace_id=request.workspace_id,
        session=session,
    )


@app.get("/chats/{chat_id}", response_model=ChatSessionResponse)
def get_chat(
    chat_id: str,
    workspace_id: str | None = None,
) -> ChatSessionResponse:
    """Return one stored chat session."""
    resolved_workspace_id = workspace_id or CONFIG.default_workspace_key
    session = load_chat_session(
        chat_id=chat_id,
        chat_db=get_api_chat_db(),
        chat_db_name=CONFIG.chat_db_name,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    return _chat_session_response(
        chat_id=chat_id,
        workspace_id=resolved_workspace_id,
        session=session,
    )


@app.delete("/chats/{chat_id}", response_model=DeleteChatResponse)
def delete_chat(
    chat_id: str,
    workspace_id: str | None = None,
) -> DeleteChatResponse:
    """Delete one chat and remove it from the workspace chat list."""
    resolved_workspace_id = workspace_id or CONFIG.default_workspace_key
    chat_db = get_api_chat_db()
    remaining_chat_ids = delete_chat_session(
        user_id=resolved_workspace_id,
        chat_id=chat_id,
        chat_db=chat_db,
        chat_db_name=CONFIG.chat_db_name,
        user_db_name=CONFIG.user_db_name,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    return DeleteChatResponse(
        deleted_chat_id=chat_id,
        workspace_id=resolved_workspace_id,
        remaining_chat_ids=remaining_chat_ids,
        chats=list_chat_summaries(
            user_id=resolved_workspace_id,
            chat_db=chat_db,
            user_db_name=CONFIG.user_db_name,
            chat_meta_db_name=CONFIG.chat_meta_db_name,
        ),
    )


@tracing.observe(name="chat_turn")
def _run_observed_chat_request(
    request: ChatRequest,
    *,
    workspace_id: str,
    chat_turn: int,
    last_request_goal: str | None,
    history: list[ModelMessage],
) -> ChatTurnResult:
    return run_chat_turn(
        user_prompt=request.message,
        chat_id=request.chat_id,
        user_id=workspace_id,
        config=CONFIG,
        chat_turn=chat_turn,
        last_request_goal=last_request_goal,
        history=history,
        thread_id=f"{request.chat_id}:{chat_turn}",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the existing LangGraph chat workflow for one HTTP request."""
    workspace_id = request.workspace_id or CONFIG.default_workspace_key
    chat_db = get_api_chat_db()
    session = load_chat_session(
        chat_id=request.chat_id,
        chat_db=chat_db,
        chat_db_name=CONFIG.chat_db_name,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    try:
        result = _run_observed_chat_request(
            request,
            workspace_id=workspace_id,
            chat_turn=session.next_chat_turn,
            last_request_goal=session.last_request_goal or None,
            history=session.history,
        )
        messages = append_chat_turn(
            chat_id=request.chat_id,
            user_content=result.normalized_prompt,
            assistant_content=result.assistant_response.rule,
            chat_db=chat_db,
            chat_db_name=CONFIG.chat_db_name,
            chat_meta_db_name=CONFIG.chat_meta_db_name,
            messages=session.messages,
            chat_turn=session.next_chat_turn,
        )
        request_goal = result.assistant_response.request_analysis.request_goal
        set_last_request_goal(
            chat_id=request.chat_id,
            last_request_goal=request_goal,
            chat_db=chat_db,
            chat_meta_db_name=CONFIG.chat_meta_db_name,
        )
        title = _ensure_chat_title(
            chat_id=request.chat_id,
            messages=messages,
            chat_db=chat_db,
            existing_title=str(session.meta.get("title") or ""),
        )
        ensure_user_chat_id(
            user_id=workspace_id,
            chat_id=request.chat_id,
            chat_db=chat_db,
            user_db_name=CONFIG.user_db_name,
        )
        return ChatResponse(
            chat_id=request.chat_id,
            workspace_id=workspace_id,
            chat_turn=session.next_chat_turn,
            message=result.normalized_prompt,
            assistant_response=result.assistant_response,
            title=title,
        )
    finally:
        tracing.flush()


def _ensure_chat_title(
    *,
    chat_id: str,
    messages: list[dict[str, str]],
    chat_db: ChatDB,
    existing_title: str,
) -> str:
    title = existing_title.strip()
    if title:
        return title
    try:
        generated_title = generate_chat_title(messages=messages)
    except Exception as err:
        logger.warning("Failed to generate chat title: %s", err)
        return ""
    title = generated_title.strip()
    if title:
        set_chat_title(
            chat_id=chat_id,
            title=title,
            chat_db=chat_db,
            chat_meta_db_name=CONFIG.chat_meta_db_name,
        )
    return title


def _chat_session_response(
    *,
    chat_id: str,
    workspace_id: str,
    session: LoadedChatSession,
) -> ChatSessionResponse:
    return ChatSessionResponse(
        chat_id=chat_id,
        workspace_id=workspace_id,
        messages=session.messages,
        title=str(session.meta.get("title") or ""),
        last_request_goal=session.last_request_goal,
        chat_turn=max(0, session.next_chat_turn - 1),
    )
