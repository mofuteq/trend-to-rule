"""Chat session persistence helpers shared by Streamlit and FastAPI."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from core.app_config import AppConfig
from storage.chat_db import ChatDB

ChatMessage = dict[str, str]


@dataclass(frozen=True)
class LoadedChatSession:
    """Stored chat session state needed before running a workflow turn."""

    messages: list[ChatMessage]
    meta: dict[str, Any]
    history: list[ModelMessage]
    last_request_goal: str
    next_chat_turn: int


def open_chat_db(config: AppConfig) -> ChatDB:
    """Open the configured chat databases."""
    return ChatDB(
        path=config.db_path,
        db_names=[
            config.chat_db_name,
            config.user_db_name,
            config.chat_meta_db_name,
        ],
    )


def normalize_chat_id_list(raw_ids: list[object]) -> list[str]:
    """Normalize a stored workspace chat id list."""
    normalized: list[str] = []
    for raw_id in raw_ids:
        if isinstance(raw_id, str):
            if raw_id and raw_id not in normalized:
                normalized.append(raw_id)
            continue
        if isinstance(raw_id, list):
            for nested_id in raw_id:
                if (
                    isinstance(nested_id, str)
                    and nested_id
                    and nested_id not in normalized
                ):
                    normalized.append(nested_id)
    return normalized


def normalize_chat_messages(raw_messages: object) -> list[ChatMessage]:
    """Return stored chat messages in the UI/API message shape."""
    if not isinstance(raw_messages, list):
        return []
    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role") or "").strip()
        content = str(raw_message.get("content") or "")
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": content})
    return messages


def messages_to_model_history(messages: list[ChatMessage]) -> list[ModelMessage]:
    """Convert stored chat messages to Pydantic AI model history."""
    history: list[ModelMessage] = []
    for message in messages:
        if message["role"] == "user":
            history.append(
                ModelRequest(parts=[UserPromptPart(content=message["content"])])
            )
        elif message["role"] == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=message["content"])]))
    return history


def load_chat_messages(
    *,
    chat_id: str,
    chat_db: ChatDB,
    chat_db_name: str,
) -> list[ChatMessage]:
    """Load normalized stored messages for a chat id."""
    return normalize_chat_messages(chat_db.get(key=chat_id, db_name=chat_db_name))


def load_user_chat_ids(user_id: str, chat_db: ChatDB, user_db_name: str) -> list[str]:
    """Load and normalize chat ids for one workspace/user."""
    user_chat_ids = chat_db.get(key=user_id, db_name=user_db_name)
    if not isinstance(user_chat_ids, list):
        return []
    normalized_user_chat_ids = normalize_chat_id_list(user_chat_ids)
    if normalized_user_chat_ids != user_chat_ids:
        chat_db.put(key=user_id, value=normalized_user_chat_ids, db_name=user_db_name)
    return normalized_user_chat_ids


def ensure_user_chat_id(
    *,
    user_id: str,
    chat_id: str,
    chat_db: ChatDB,
    user_db_name: str,
) -> list[str]:
    """Persist a chat id in the workspace chat list if it is missing."""
    user_chat_ids = load_user_chat_ids(
        user_id=user_id,
        chat_db=chat_db,
        user_db_name=user_db_name,
    )
    if chat_id not in user_chat_ids:
        user_chat_ids.append(chat_id)
        chat_db.put(key=user_id, value=user_chat_ids, db_name=user_db_name)
    return user_chat_ids


def remove_user_chat_id(
    *,
    user_id: str,
    chat_id: str,
    chat_db: ChatDB,
    user_db_name: str,
) -> list[str]:
    """Remove a chat id from the workspace chat list."""
    user_chat_ids = [
        existing_chat_id
        for existing_chat_id in load_user_chat_ids(
            user_id=user_id,
            chat_db=chat_db,
            user_db_name=user_db_name,
        )
        if existing_chat_id != chat_id
    ]
    chat_db.put(key=user_id, value=user_chat_ids, db_name=user_db_name)
    return user_chat_ids


def sort_chat_ids_by_last_updated(
    chat_ids: list[str],
    chat_db: ChatDB,
    chat_meta_db_name: str,
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Sort chat ids by last updated timestamp in descending order."""
    updated_at_by_chat: dict[str, float] = {}
    title_by_chat: dict[str, str] = {}
    for chat_id in chat_ids:
        meta = get_chat_meta(
            chat_id=chat_id,
            chat_db=chat_db,
            chat_meta_db_name=chat_meta_db_name,
        )
        updated_at_ts = meta.get("updated_at_ts")
        updated_at_by_chat[chat_id] = (
            float(updated_at_ts) if isinstance(updated_at_ts, (int, float)) else 0.0
        )
        title_by_chat[chat_id] = str(meta.get("title") or "").strip()

    sorted_chat_ids = sorted(
        chat_ids,
        key=lambda chat_id: updated_at_by_chat.get(chat_id, 0.0),
        reverse=True,
    )
    return sorted_chat_ids, updated_at_by_chat, title_by_chat


def list_chat_summaries(
    *,
    user_id: str,
    chat_db: ChatDB,
    user_db_name: str,
    chat_meta_db_name: str,
) -> list[dict[str, object]]:
    """Return sidebar-ready chat summaries for one workspace."""
    chat_ids = load_user_chat_ids(
        user_id=user_id,
        chat_db=chat_db,
        user_db_name=user_db_name,
    )
    sorted_chat_ids, updated_at_by_chat, title_by_chat = sort_chat_ids_by_last_updated(
        chat_ids=chat_ids,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
    return [
        {
            "chat_id": chat_id,
            "title": title_by_chat.get(chat_id, ""),
            "updated_at_ts": (
                updated_at_by_chat[chat_id]
                if updated_at_by_chat.get(chat_id, 0.0) > 0
                else None
            ),
        }
        for chat_id in sorted_chat_ids
    ]


def get_chat_meta(chat_id: str, chat_db: ChatDB, chat_meta_db_name: str) -> dict:
    """Return chat metadata as a dict."""
    meta = chat_db.get(key=chat_id, db_name=chat_meta_db_name)
    return meta if isinstance(meta, dict) else {}


def update_chat_meta(
    chat_id: str,
    chat_db: ChatDB,
    chat_meta_db_name: str,
    *,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update chat metadata while preserving existing fields."""
    meta = get_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
    meta["updated_at_ts"] = datetime.now(tz=timezone.utc).timestamp()
    if updates:
        meta.update(updates)
    chat_db.put(
        key=chat_id,
        value=meta,
        db_name=chat_meta_db_name,
    )
    return meta


def set_last_request_goal(
    chat_id: str,
    last_request_goal: str,
    chat_db: ChatDB,
    chat_meta_db_name: str,
) -> None:
    """Persist the latest inferred request goal to chat metadata."""
    update_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
        updates={"last_request_goal": last_request_goal},
    )


def set_chat_title(
    chat_id: str,
    title: str,
    chat_db: ChatDB,
    chat_meta_db_name: str,
) -> None:
    """Persist the chat title while preserving existing metadata."""
    update_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
        updates={"title": title},
    )


def load_chat_session(
    *,
    chat_id: str,
    chat_db: ChatDB,
    chat_db_name: str,
    chat_meta_db_name: str,
) -> LoadedChatSession:
    """Load messages, metadata, model history, and next turn for a chat."""
    messages = load_chat_messages(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_db_name=chat_db_name,
    )
    meta = get_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
    return LoadedChatSession(
        messages=messages,
        meta=meta,
        history=messages_to_model_history(messages),
        last_request_goal=_get_last_request_goal(meta),
        next_chat_turn=_get_next_chat_turn(messages=messages, meta=meta),
    )


def initialize_chat_session(
    *,
    user_id: str,
    chat_id: str,
    chat_db: ChatDB,
    chat_db_name: str,
    user_db_name: str,
    chat_meta_db_name: str,
) -> LoadedChatSession:
    """Create backend metadata for a chat if missing and attach it to a workspace."""
    ensure_user_chat_id(
        user_id=user_id,
        chat_id=chat_id,
        chat_db=chat_db,
        user_db_name=user_db_name,
    )
    messages = load_chat_messages(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_db_name=chat_db_name,
    )
    meta = get_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
    )
    if not meta:
        update_chat_meta(
            chat_id=chat_id,
            chat_db=chat_db,
            chat_meta_db_name=chat_meta_db_name,
            updates={
                "title": "",
                "chat_turn": max(
                    0,
                    _get_next_chat_turn(messages=messages, meta={}) - 1,
                ),
                "last_request_goal": "",
            },
        )
    return load_chat_session(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_db_name=chat_db_name,
        chat_meta_db_name=chat_meta_db_name,
    )


def delete_chat_session(
    *,
    user_id: str,
    chat_id: str,
    chat_db: ChatDB,
    chat_db_name: str,
    user_db_name: str,
    chat_meta_db_name: str,
) -> list[str]:
    """Delete one chat and detach it from a workspace."""
    chat_db.delete(chat_id, chat_db_name)
    chat_db.delete(chat_id, chat_meta_db_name)
    return remove_user_chat_id(
        user_id=user_id,
        chat_id=chat_id,
        chat_db=chat_db,
        user_db_name=user_db_name,
    )


def append_chat_turn(
    *,
    chat_id: str,
    user_content: str,
    assistant_content: str,
    chat_db: ChatDB,
    chat_db_name: str,
    chat_meta_db_name: str,
    messages: list[ChatMessage] | None = None,
    chat_turn: int | None = None,
) -> list[ChatMessage]:
    """Append a completed user/assistant turn and persist it to ChatDB."""
    updated_messages = list(messages) if messages is not None else load_chat_messages(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_db_name=chat_db_name,
    )
    updated_messages.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    )
    chat_db.put(
        key=chat_id,
        value=updated_messages,
        db_name=chat_db_name,
    )
    updates: dict[str, Any] = {}
    if chat_turn is not None:
        updates["chat_turn"] = chat_turn
    update_chat_meta(
        chat_id=chat_id,
        chat_db=chat_db,
        chat_meta_db_name=chat_meta_db_name,
        updates=updates,
    )
    return updated_messages


def _get_last_request_goal(meta: dict[str, Any]) -> str:
    return str(meta.get("last_request_goal") or meta.get("last_user_goal") or "")


def _get_next_chat_turn(
    *,
    messages: list[ChatMessage],
    meta: dict[str, Any],
) -> int:
    stored_turn = meta.get("chat_turn")
    if isinstance(stored_turn, int) and stored_turn >= 0:
        return stored_turn + 1
    completed_user_turns = sum(1 for message in messages if message["role"] == "user")
    return completed_user_turns + 1
