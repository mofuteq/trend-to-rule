"""Minimal FastAPI boundary for the existing chat workflow."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import uuid

from fastapi import FastAPI, HTTPException
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
    ResumeChatRequest,
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
    messages_to_model_history,
    open_chat_db,
    set_chat_title,
    set_last_request_goal,
    update_chat_meta,
)
from storage.chat_db import ChatDB  # noqa: E402

CONFIG = load_app_config()
_CHAT_DB: ChatDB | None = None
logger = logging.getLogger(__name__)

WORKFLOW_STATUS_RUNNING = "running"
WORKFLOW_STATUS_COMPLETED = "completed"
WORKFLOW_STATUS_FAILED = "failed"
RESUMABLE_WORKFLOW_STATUSES = {
    WORKFLOW_STATUS_RUNNING,
    WORKFLOW_STATUS_FAILED,
}
WORKFLOW_ERROR_LIMIT = 500

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


def _now_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def _workflow_thread_id(chat_id: str, chat_turn: int) -> str:
    return f"{chat_id}:{chat_turn}"


def _short_error_string(err: BaseException) -> str:
    text = " ".join(str(err).split())
    if not text:
        text = err.__class__.__name__
    if len(text) <= WORKFLOW_ERROR_LIMIT:
        return text
    return f"{text[: WORKFLOW_ERROR_LIMIT - 3]}..."


def _mark_workflow_running(
    *,
    chat_id: str,
    chat_turn: int,
    thread_id: str,
    user_prompt: str,
    workspace_id: str,
    chat_db: ChatDB,
    started_at_ts: float | None = None,
) -> None:
    update_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
        updates={
            "latest_thread_id": thread_id,
            "latest_chat_turn": chat_turn,
            "latest_workflow_status": WORKFLOW_STATUS_RUNNING,
            "latest_workflow_error": "",
            "latest_workflow_started_at_ts": started_at_ts or _now_ts(),
            "latest_workflow_completed_at_ts": None,
            "latest_workflow_message": user_prompt,
            "latest_workflow_workspace_id": workspace_id,
        },
    )


def _mark_workflow_completed(
    *,
    chat_id: str,
    chat_turn: int,
    thread_id: str,
    chat_db: ChatDB,
) -> None:
    update_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
        updates={
            "latest_thread_id": thread_id,
            "latest_chat_turn": chat_turn,
            "latest_workflow_status": WORKFLOW_STATUS_COMPLETED,
            "latest_workflow_error": "",
            "latest_workflow_completed_at_ts": _now_ts(),
        },
    )


def _mark_workflow_failed(
    *,
    chat_id: str,
    chat_turn: int,
    thread_id: str,
    chat_db: ChatDB,
    err: BaseException,
) -> None:
    update_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
        updates={
            "latest_thread_id": thread_id,
            "latest_chat_turn": chat_turn,
            "latest_workflow_status": WORKFLOW_STATUS_FAILED,
            "latest_workflow_error": _short_error_string(err),
            "latest_workflow_completed_at_ts": None,
        },
    )


@tracing.observe(name="chat_turn")
def _run_observed_chat_turn(
    *,
    user_prompt: str,
    chat_id: str,
    workspace_id: str,
    chat_turn: int,
    last_request_goal: str | None,
    history: list[ModelMessage],
    thread_id: str,
    resume_from_checkpoint: bool = False,
) -> ChatTurnResult:
    return run_chat_turn(
        user_prompt=user_prompt,
        chat_id=chat_id,
        user_id=workspace_id,
        config=CONFIG,
        chat_turn=chat_turn,
        last_request_goal=last_request_goal,
        history=history,
        thread_id=thread_id,
        resume_from_checkpoint=resume_from_checkpoint,
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
    chat_turn = session.next_chat_turn
    thread_id = _workflow_thread_id(request.chat_id, chat_turn)
    ensure_user_chat_id(
        user_id=workspace_id,
        chat_id=request.chat_id,
        chat_db=chat_db,
        user_db_name=CONFIG.user_db_name,
    )
    _mark_workflow_running(
        chat_id=request.chat_id,
        chat_turn=chat_turn,
        thread_id=thread_id,
        user_prompt=request.message,
        workspace_id=workspace_id,
        chat_db=chat_db,
    )
    try:
        result = _run_observed_chat_turn(
            user_prompt=request.message,
            chat_id=request.chat_id,
            workspace_id=workspace_id,
            chat_turn=chat_turn,
            last_request_goal=session.last_request_goal or None,
            history=session.history,
            thread_id=thread_id,
        )
        response = _persist_completed_chat_response(
            chat_id=request.chat_id,
            workspace_id=workspace_id,
            chat_turn=chat_turn,
            session=session,
            result=result,
            chat_db=chat_db,
        )
        _mark_workflow_completed(
            chat_id=request.chat_id,
            chat_turn=chat_turn,
            thread_id=thread_id,
            chat_db=chat_db,
        )
        return response
    except Exception as err:
        _mark_workflow_failed(
            chat_id=request.chat_id,
            chat_turn=chat_turn,
            thread_id=thread_id,
            chat_db=chat_db,
            err=err,
        )
        raise
    finally:
        tracing.flush()


@app.post("/chats/{chat_id}/resume", response_model=ChatResponse)
def resume_chat(
    chat_id: str,
    request: ResumeChatRequest | None = None,
) -> ChatResponse:
    """Resume the latest failed or interrupted workflow run for a chat."""
    resume_request = request or ResumeChatRequest()
    chat_db = get_api_chat_db()
    session = load_chat_session(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_db_name=CONFIG.chat_db_name,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    meta = session.meta
    status = str(meta.get("latest_workflow_status") or "")
    if status not in RESUMABLE_WORKFLOW_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="No failed or running workflow run is available to resume.",
        )

    chat_turn = meta.get("latest_chat_turn")
    thread_id = str(meta.get("latest_thread_id") or "")
    if not isinstance(chat_turn, int) or chat_turn < 1 or not thread_id:
        raise HTTPException(
            status_code=409,
            detail="Stored workflow run metadata is incomplete and cannot resume.",
        )
    if resume_request.chat_turn is not None and resume_request.chat_turn != chat_turn:
        raise HTTPException(
            status_code=409,
            detail="Only the latest workflow run can be resumed.",
        )
    if resume_request.thread_id is not None and resume_request.thread_id != thread_id:
        raise HTTPException(
            status_code=409,
            detail="Only the latest workflow run can be resumed.",
        )
    if _completed_chat_turn_persisted(session.messages, chat_turn):
        _mark_workflow_completed(
            chat_id=chat_id,
            chat_turn=chat_turn,
            thread_id=thread_id,
            chat_db=chat_db,
        )
        raise HTTPException(
            status_code=409,
            detail="The latest workflow run is already completed.",
        )

    user_prompt = str(meta.get("latest_workflow_message") or "")
    if not user_prompt.strip():
        raise HTTPException(
            status_code=409,
            detail="Stored workflow run is missing the original user message.",
        )
    workspace_id = (
        resume_request.workspace_id
        or str(meta.get("latest_workflow_workspace_id") or "")
        or CONFIG.default_workspace_key
    )
    started_at_ts = meta.get("latest_workflow_started_at_ts")
    _mark_workflow_running(
        chat_id=chat_id,
        chat_turn=chat_turn,
        thread_id=thread_id,
        user_prompt=user_prompt,
        workspace_id=workspace_id,
        chat_db=chat_db,
        started_at_ts=(
            float(started_at_ts)
            if isinstance(started_at_ts, (int, float))
            else None
        ),
    )
    try:
        result = _run_observed_chat_turn(
            user_prompt=user_prompt,
            chat_id=chat_id,
            workspace_id=workspace_id,
            chat_turn=chat_turn,
            last_request_goal=session.last_request_goal or None,
            history=_history_before_chat_turn(session.messages, chat_turn),
            thread_id=thread_id,
            resume_from_checkpoint=True,
        )
        response = _persist_completed_chat_response(
            chat_id=chat_id,
            workspace_id=workspace_id,
            chat_turn=chat_turn,
            session=session,
            result=result,
            chat_db=chat_db,
        )
        _mark_workflow_completed(
            chat_id=chat_id,
            chat_turn=chat_turn,
            thread_id=thread_id,
            chat_db=chat_db,
        )
        return response
    except Exception as err:
        _mark_workflow_failed(
            chat_id=chat_id,
            chat_turn=chat_turn,
            thread_id=thread_id,
            chat_db=chat_db,
            err=err,
        )
        raise
    finally:
        tracing.flush()


def _history_before_chat_turn(
    messages: list[dict[str, str]],
    chat_turn: int,
) -> list[ModelMessage]:
    current_turn_index = max(0, (chat_turn - 1) * 2)
    return messages_to_model_history(messages[:current_turn_index])


def _completed_chat_turn_persisted(
    messages: list[dict[str, str]],
    chat_turn: int,
) -> bool:
    user_index = (chat_turn - 1) * 2
    assistant_index = user_index + 1
    return (
        len(messages) > assistant_index
        and messages[user_index].get("role") == "user"
        and messages[assistant_index].get("role") == "assistant"
    )


def _append_or_complete_chat_turn(
    *,
    chat_id: str,
    user_content: str,
    assistant_content: str,
    chat_db: ChatDB,
    messages: list[dict[str, str]],
    chat_turn: int,
) -> list[dict[str, str]]:
    user_index = (chat_turn - 1) * 2
    assistant_index = user_index + 1
    if _completed_chat_turn_persisted(messages, chat_turn):
        update_chat_meta(
            chat_id=chat_id,
            chat_db=chat_db,
            chat_meta_db_name=CONFIG.chat_meta_db_name,
            updates={"chat_turn": chat_turn},
        )
        return messages
    if (
        len(messages) == assistant_index
        and messages[user_index].get("role") == "user"
    ):
        updated_messages = [
            *messages,
            {"role": "assistant", "content": assistant_content},
        ]
        chat_db.put(
            key=chat_id,
            value=updated_messages,
            db_name=CONFIG.chat_db_name,
        )
        update_chat_meta(
            chat_id=chat_id,
            chat_db=chat_db,
            chat_meta_db_name=CONFIG.chat_meta_db_name,
            updates={"chat_turn": chat_turn},
        )
        return updated_messages
    return append_chat_turn(
        chat_id=chat_id,
        user_content=user_content,
        assistant_content=assistant_content,
        chat_db=chat_db,
        chat_db_name=CONFIG.chat_db_name,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
        messages=messages,
        chat_turn=chat_turn,
    )


def _persist_completed_chat_response(
    *,
    chat_id: str,
    workspace_id: str,
    chat_turn: int,
    session: LoadedChatSession,
    result: ChatTurnResult,
    chat_db: ChatDB,
) -> ChatResponse:
    messages = _append_or_complete_chat_turn(
        chat_id=chat_id,
        user_content=result.normalized_prompt,
        assistant_content=result.assistant_response.rule,
        chat_db=chat_db,
        messages=session.messages,
        chat_turn=chat_turn,
    )
    request_goal = result.assistant_response.request_analysis.request_goal
    set_last_request_goal(
        chat_id=chat_id,
        last_request_goal=request_goal,
        chat_db=chat_db,
        chat_meta_db_name=CONFIG.chat_meta_db_name,
    )
    title = _ensure_chat_title(
        chat_id=chat_id,
        messages=messages,
        chat_db=chat_db,
        existing_title=str(session.meta.get("title") or ""),
    )
    ensure_user_chat_id(
        user_id=workspace_id,
        chat_id=chat_id,
        chat_db=chat_db,
        user_db_name=CONFIG.user_db_name,
    )
    return ChatResponse(
        chat_id=chat_id,
        workspace_id=workspace_id,
        chat_turn=chat_turn,
        message=result.normalized_prompt,
        assistant_response=result.assistant_response,
        title=title,
    )


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
    latest_chat_turn = session.meta.get("latest_chat_turn")
    return ChatSessionResponse(
        chat_id=chat_id,
        workspace_id=workspace_id,
        messages=session.messages,
        title=str(session.meta.get("title") or ""),
        last_request_goal=session.last_request_goal,
        chat_turn=max(0, session.next_chat_turn - 1),
        latest_workflow_status=str(session.meta.get("latest_workflow_status") or ""),
        latest_chat_turn=(
            latest_chat_turn if isinstance(latest_chat_turn, int) else None
        ),
        latest_thread_id=str(session.meta.get("latest_thread_id") or ""),
        latest_workflow_error=str(session.meta.get("latest_workflow_error") or ""),
    )
