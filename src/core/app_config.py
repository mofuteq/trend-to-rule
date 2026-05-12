import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.hf_cache import configure_huggingface_cache


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
ENV_PATH = SRC_ROOT / ".env"
if ENV_PATH.is_file():
    load_dotenv(ENV_PATH)
configure_huggingface_cache()


def resolve_project_path(raw_path: str) -> Path:
    """Resolve path relative to project root unless absolute.

    Args:
        raw_path: Raw filesystem path from environment or default.

    Returns:
        Absolute path under the project root when input is relative.
    """
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the Streamlit app."""

    db_path: Path
    chat_db_name: str
    user_db_name: str
    chat_meta_db_name: str
    app_log_level: str
    tavily_api_key: str
    tavily_text_max_results: int
    tavily_search_depth: str
    tavily_include_raw_content: bool
    tavily_image_fetch_limit: int
    tavily_image_limit: int
    tavily_include_image_descriptions: bool
    workspace_query_key: str
    default_workspace_key: str


def load_app_config() -> AppConfig:
    """Load app configuration from environment variables.

    Returns:
        AppConfig: Parsed runtime configuration.
    """
    return AppConfig(
        db_path=resolve_project_path(
            os.getenv("CHAT_DB_PATH", ".data/chat_db")
        ),
        chat_db_name="chat_db",
        user_db_name="user_db",
        chat_meta_db_name="chat_meta_db",
        app_log_level=os.getenv("APP_LOG_LEVEL", "INFO").upper(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        tavily_text_max_results=int(os.getenv("TAVILY_TEXT_MAX_RESULTS", "5")),
        tavily_search_depth=_parse_tavily_search_depth(
            os.getenv("TAVILY_SEARCH_DEPTH", "basic")
        ),
        tavily_include_raw_content=_parse_bool(
            os.getenv("TAVILY_INCLUDE_RAW_CONTENT", "false")
        ),
        tavily_image_fetch_limit=int(os.getenv("TAVILY_IMAGE_FETCH_LIMIT", "10")),
        tavily_image_limit=int(os.getenv("TAVILY_IMAGE_LIMIT", "3")),
        tavily_include_image_descriptions=_parse_bool(
            os.getenv("TAVILY_INCLUDE_IMAGE_DESCRIPTIONS", "true")
        ),
        workspace_query_key="workspace",
        default_workspace_key=os.getenv("T2R_DEFAULT_WORKSPACE", "demo"),
    )


def _parse_bool(value: str) -> bool:
    """Parse common environment boolean strings."""
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_tavily_search_depth(value: str) -> str:
    """Parse Tavily search depth with a conservative default."""
    normalized = value.strip().lower()
    if normalized in {"basic", "advanced"}:
        return normalized
    return "basic"
