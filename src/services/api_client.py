"""HTTP client for the FastAPI chat execution boundary."""

import httpx

from core.app_config import AppConfig
from services.api_models import ChatRequest, ChatResponse


def post_chat_turn(
    *,
    chat_id: str,
    message: str,
    workspace_id: str,
    config: AppConfig,
) -> ChatResponse:
    """Post a chat turn to the configured FastAPI backend."""
    base_url = config.api_base_url.rstrip("/")
    if not base_url:
        raise ValueError("T2R_API_BASE_URL is required")

    request = ChatRequest(
        chat_id=chat_id,
        workspace_id=workspace_id,
        message=message,
    )
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        response = client.post("/chat", json=request.model_dump())
        response.raise_for_status()
        return ChatResponse.model_validate(response.json())

