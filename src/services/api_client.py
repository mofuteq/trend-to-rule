"""HTTP client for the FastAPI chat execution boundary."""

import httpx

from core.app_config import AppConfig
from services.api_models import (
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
    CreateChatRequest,
    CreateChatResponse,
    DeleteChatResponse,
    ListChatsResponse,
)


def list_chats(
    *,
    workspace_id: str,
    config: AppConfig,
) -> ListChatsResponse:
    """Load chat summaries for a workspace from the FastAPI backend."""
    with _client(config) as client:
        response = client.get("/chats", params={"workspace_id": workspace_id})
        response.raise_for_status()
        return ListChatsResponse.model_validate(response.json())


def get_chat(
    *,
    chat_id: str,
    workspace_id: str,
    config: AppConfig,
) -> ChatSessionResponse:
    """Load one chat session from the FastAPI backend."""
    with _client(config) as client:
        response = client.get(
            f"/chats/{chat_id}",
            params={"workspace_id": workspace_id},
        )
        response.raise_for_status()
        return ChatSessionResponse.model_validate(response.json())


def create_chat(
    *,
    workspace_id: str,
    config: AppConfig,
    chat_id: str | None = None,
) -> CreateChatResponse:
    """Create or initialize a chat through the FastAPI backend."""
    request = CreateChatRequest(workspace_id=workspace_id, chat_id=chat_id)
    with _client(config) as client:
        response = client.post("/chats", json=request.model_dump())
        response.raise_for_status()
        return CreateChatResponse.model_validate(response.json())


def delete_chat(
    *,
    chat_id: str,
    workspace_id: str,
    config: AppConfig,
) -> DeleteChatResponse:
    """Delete one chat through the FastAPI backend."""
    with _client(config) as client:
        response = client.delete(
            f"/chats/{chat_id}",
            params={"workspace_id": workspace_id},
        )
        response.raise_for_status()
        return DeleteChatResponse.model_validate(response.json())


def post_chat_turn(
    *,
    chat_id: str,
    message: str,
    workspace_id: str,
    config: AppConfig,
) -> ChatResponse:
    """Post a chat turn to the configured FastAPI backend."""
    request = ChatRequest(
        chat_id=chat_id,
        workspace_id=workspace_id,
        message=message,
    )
    with _client(config) as client:
        response = client.post("/chat", json=request.model_dump())
        response.raise_for_status()
        return ChatResponse.model_validate(response.json())


def _client(config: AppConfig) -> httpx.Client:
    base_url = config.api_base_url.rstrip("/")
    if not base_url:
        raise ValueError("T2R_API_BASE_URL is required")
    return httpx.Client(base_url=base_url, timeout=120.0)
