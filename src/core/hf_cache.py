"""Hugging Face cache configuration shared by model-loading code."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
ENV_PATH = SRC_ROOT / ".env"
DEFAULT_HF_HOME = PROJECT_ROOT / ".data" / "huggingface"


def configure_huggingface_cache() -> Path:
    """Ensure Hugging Face libraries use the shared project cache directory.

    The default `.env` value is intentionally relative (`.data/huggingface`) so it
    resolves to the shared bind mount in Docker (`/app/.data/huggingface`) and to
    the local repo's `.data/huggingface` when run from the project root.
    """
    if ENV_PATH.is_file():
        load_dotenv(ENV_PATH, override=False)

    cache_root = _resolve_cache_path(os.getenv("HF_HOME", ""))
    cache_root.mkdir(parents=True, exist_ok=True)

    hub_cache = _resolve_cache_path(
        os.getenv("HF_HUB_CACHE", ""),
        default=cache_root / "hub",
    )
    transformers_cache = _resolve_cache_path(
        os.getenv("TRANSFORMERS_CACHE", ""),
        default=hub_cache,
    )
    sentence_transformers_cache = _resolve_cache_path(
        os.getenv("SENTENCE_TRANSFORMERS_HOME", ""),
        default=hub_cache,
    )

    for path in (hub_cache, transformers_cache, sentence_transformers_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_cache)
    return cache_root


def clear_incomplete_huggingface_downloads(
    cache_root: Path,
    *,
    model_name: str = "",
) -> int:
    """Remove interrupted Hugging Face hub downloads for a model cache.

    Hugging Face writes blobs through ``*.incomplete`` files. If a previous
    download is interrupted or raced across processes, later model loads can
    fail with FileNotFoundError for that transient path. Removing the partials
    lets the hub client recreate them cleanly on retry.
    """
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", cache_root / "hub"))
    search_root = hub_cache
    if model_name and "/" in model_name:
        model_cache_name = f"models--{model_name.replace('/', '--')}"
        model_cache_root = hub_cache / model_cache_name
        if model_cache_root.exists():
            search_root = model_cache_root

    removed = 0
    for partial_path in search_root.rglob("*.incomplete"):
        try:
            partial_path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def clear_huggingface_model_cache(cache_root: Path, *, model_name: str) -> bool:
    """Remove a specific Hugging Face hub model cache directory."""
    if not model_name or "/" not in model_name:
        return False
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", cache_root / "hub"))
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_root = hub_cache / model_cache_name
    if not model_cache_root.exists():
        return False
    shutil.rmtree(model_cache_root)
    return True


def _resolve_cache_path(raw_path: str, *, default: Path = DEFAULT_HF_HOME) -> Path:
    """Resolve configured Hugging Face cache path relative to the project root."""
    path_text = raw_path.strip()
    if not path_text:
        return default.resolve()
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
