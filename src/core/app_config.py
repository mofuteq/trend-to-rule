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
    vector_collection: str
    vector_model_name: str
    vector_device: str
    vector_candidate_k: int
    vector_qdrant_url: str
    vector_per_query_top_k: int
    vector_mmr_diversity: float
    app_log_level: str
    searxng_base_url: str
    searxng_image_fetch_limit: int
    searxng_image_limit: int
    anonymous_user_query_key: str


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
        vector_collection=os.getenv(
            "VECTOR_COLLECTION", "article_markdown_bge_m3"
        ),
        vector_model_name=os.getenv("VECTOR_MODEL_NAME", "BAAI/bge-m3"),
        vector_device=os.getenv("VECTOR_DEVICE", "auto"),
        vector_candidate_k=int(os.getenv("VECTOR_CANDIDATE_K", "50")),
        vector_qdrant_url=os.getenv(
            "VECTOR_QDRANT_URL", "http://localhost:6333"
        ),
        vector_per_query_top_k=int(os.getenv("VECTOR_PER_QUERY_TOP_K", "5")),
        vector_mmr_diversity=float(os.getenv("VECTOR_MMR_DIVERSITY", "0.3")),
        app_log_level=os.getenv("APP_LOG_LEVEL", "INFO").upper(),
        searxng_base_url=os.getenv("SEARXNG_BASE_URL", "http://localhost:8008"),
        searxng_image_fetch_limit=int(
            os.getenv("SEARXNG_IMAGE_FETCH_LIMIT", "10")
        ),
        searxng_image_limit=int(os.getenv("SEARXNG_IMAGE_LIMIT", "3")),
        anonymous_user_query_key="uid",
    )
