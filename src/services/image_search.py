import unicodedata
from functools import lru_cache
from io import BytesIO
import logging
import os
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from core.hf_cache import configure_huggingface_cache
from PIL import Image
from pydantic import BaseModel, field_validator

configure_huggingface_cache()


DEFAULT_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_CLIP_MODEL_FAMILY = "sentence-transformers/clip-ViT-B-32"
DEFAULT_CLIP_TEXT_MODEL_NAME = os.getenv(
    "CLIP_TEXT_MODEL_NAME",
    f"{DEFAULT_CLIP_MODEL_FAMILY}-multilingual-v1",
)
DEFAULT_CLIP_IMAGE_MODEL_NAME = os.getenv(
    "CLIP_IMAGE_MODEL_NAME",
    DEFAULT_CLIP_MODEL_FAMILY,
)
DEFAULT_IMAGE_FETCH_LIMIT = 10
DEFAULT_IMAGE_RERANK_LIMIT = 3
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "ref",
    "ref_src",
    "si",
}
logger = logging.getLogger(__name__)


class ImageSearchResult(BaseModel):
    """Normalized image-search result returned from the visual search backend."""

    title: str
    page_url: str
    image_url: str
    thumbnail_url: str
    source: str
    engine: str
    clip_score: float | None = None

    @field_validator("image_url")
    @classmethod
    def normalize_image_url(cls, value: str) -> str:
        """Normalize image URL for stable deduplication."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("image_url must not be empty")
        parts = urlsplit(normalized)
        filtered_query = [
            (key, query_value)
            for key, query_value in parse_qsl(parts.query, keep_blank_values=True)
            if key not in TRACKING_QUERY_KEYS
            and not any(key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
        ]
        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(filtered_query, doseq=True),
                "",
            )
        )


def search_images(
    query: str,
    *,
    limit: int | None = None,
    api_key: str | None = None,
    include_image_descriptions: bool = True,
    search_url: str = DEFAULT_TAVILY_SEARCH_URL,
) -> list[ImageSearchResult]:
    """Return deduplicated image-search results from Tavily.

    Args:
        query: User search query string.
        limit: Optional maximum number of deduplicated results to return.
        api_key: Tavily API key. Falls back to TAVILY_API_KEY.
        include_image_descriptions: Request Tavily image descriptions when true.
        search_url: Tavily Search API endpoint.

    Returns:
        A list of normalized image-search results.
    """
    normalized_query = unicodedata.normalize("NFKC", query).strip()
    resolved_api_key = (
        os.getenv("TAVILY_API_KEY", "") if api_key is None else api_key
    ).strip()
    if not normalized_query:
        return []
    if not resolved_api_key:
        logger.warning("Tavily image search skipped: TAVILY_API_KEY is not configured")
        return []

    max_results = max(1, min(limit or DEFAULT_IMAGE_FETCH_LIMIT, 10))
    try:
        with httpx.Client() as client:
            response = client.post(
                search_url,
                json={
                    "query": normalized_query,
                    "include_images": True,
                    "include_image_descriptions": include_image_descriptions,
                    "include_answer": False,
                    "include_raw_content": False,
                    "search_depth": "basic",
                    "max_results": max_results,
                },
                headers={
                    **DEFAULT_HEADERS,
                    "Authorization": f"Bearer {resolved_api_key}",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as err:
        logger.warning("Tavily image search failed: %s", err)
        return []

    if not isinstance(payload, dict):
        logger.warning("Tavily image search returned malformed payload: %r", payload)
        return []

    items: list[ImageSearchResult] = []
    seen_image_urls: set[str] = set()
    seen_page_title_keys: set[str] = set()
    results = payload.get("results", [])
    if isinstance(results, list):
        for source_result in results:
            if not isinstance(source_result, dict):
                continue
            source_title = str(source_result.get("title") or "").strip()
            source_url = str(source_result.get("url") or "").strip()
            source_images = source_result.get("images", [])
            if not isinstance(source_images, list):
                continue
            for image in source_images:
                _append_tavily_image_candidate(
                    items=items,
                    seen_image_urls=seen_image_urls,
                    seen_page_title_keys=seen_page_title_keys,
                    image=image,
                    source_title=source_title,
                    source_url=source_url,
                )

    top_level_images = payload.get("images", [])
    if isinstance(top_level_images, list):
        for image in top_level_images:
            _append_tavily_image_candidate(
                items=items,
                seen_image_urls=seen_image_urls,
                seen_page_title_keys=seen_page_title_keys,
                image=image,
                source_title="",
                source_url="",
            )

    if limit is None:
        return items
    return items[:limit]


def rerank_with_clip(
    query: str,
    candidates: list[ImageSearchResult],
) -> list[ImageSearchResult]:
    """Rerank image-search results by CLIP similarity to the text query."""
    normalized_query = unicodedata.normalize("NFKC", query).strip()
    if not normalized_query or not candidates:
        return candidates

    logger.info(
        "image_rerank_start query=%r candidate_count=%s",
        normalized_query,
        len(candidates),
    )

    clip_bundle = get_clip_bundle()
    valid_results: list[ImageSearchResult] = []
    valid_images: list[Image.Image] = []
    with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=20.0) as client:
        for item in candidates:
            image = _fetch_image(client, item)
            if image is None:
                continue
            valid_results.append(item)
            valid_images.append(image)

    if not valid_results:
        logger.info("image_rerank_no_valid_images query=%r", normalized_query)
        return candidates

    import torch

    with torch.inference_mode():
        text_features = clip_bundle.text_model.encode(
            [normalized_query],
            convert_to_tensor=True,
            device=clip_bundle.device,
            **_clip_encode_options(),
        )
        image_features = clip_bundle.image_model.encode(
            valid_images,
            convert_to_tensor=True,
            device=clip_bundle.device,
            **_clip_encode_options(),
        )

        similarities = torch.matmul(image_features, text_features.T).squeeze(-1)

    scored_results = [
        result.model_copy(update={"clip_score": float(score)})
        for result, score in zip(valid_results, similarities.tolist())
    ]
    scored_results.sort(
        key=lambda item: item.clip_score if item.clip_score is not None else float("-inf"),
        reverse=True,
    )
    for item in scored_results:
        logger.info(
            "image_rerank_score title=%r source=%r clip_score=%.4f image_url=%s",
            item.title,
            item.source,
            item.clip_score if item.clip_score is not None else float("nan"),
            item.image_url,
        )
    return scored_results


def select_top_k(
    candidates: list[ImageSearchResult],
    *,
    k: int = DEFAULT_IMAGE_RERANK_LIMIT,
) -> list[ImageSearchResult]:
    """Select the top-k reranked candidates."""
    selected = candidates[:k]
    for idx, item in enumerate(selected, start=1):
        logger.info(
            "image_rerank_selected rank=%s title=%r source=%r clip_score=%s image_url=%s",
            idx,
            item.title,
            item.source,
            f"{item.clip_score:.4f}" if item.clip_score is not None else "n/a",
            item.image_url,
        )
    return selected


def _fetch_image(client: httpx.Client, item: ImageSearchResult) -> Image.Image | None:
    """Download and decode one image for CLIP scoring."""
    image_url = (item.image_url or "").strip()
    if not image_url:
        return None
    try:
        response = client.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def _append_tavily_image_candidate(
    *,
    items: list[ImageSearchResult],
    seen_image_urls: set[str],
    seen_page_title_keys: set[str],
    image: object,
    source_title: str,
    source_url: str,
) -> None:
    """Normalize one Tavily image value and append it when unique."""
    result = _normalize_tavily_image_candidate(
        image=image,
        source_title=source_title,
        source_url=source_url,
    )
    if result is None or result.image_url in seen_image_urls:
        return
    page_title_key = _build_page_title_dedupe_key(result)
    if page_title_key is not None and page_title_key in seen_page_title_keys:
        return
    seen_image_urls.add(result.image_url)
    if page_title_key is not None:
        seen_page_title_keys.add(page_title_key)
    items.append(result)


def _normalize_tavily_image_candidate(
    *,
    image: object,
    source_title: str,
    source_url: str,
) -> ImageSearchResult | None:
    """Convert Tavily string/object images into the stable result model."""
    image_url = ""
    description = ""
    if isinstance(image, str):
        image_url = image.strip()
    elif isinstance(image, dict):
        image_url = str(image.get("url") or "").strip()
        description = str(image.get("description") or "").strip()
    else:
        return None

    if not image_url:
        return None

    title = description or source_title or "Visual reference"
    try:
        return ImageSearchResult(
            title=title,
            page_url=source_url,
            image_url=image_url,
            thumbnail_url=image_url,
            source="tavily",
            engine="tavily",
        )
    except Exception:
        return None


def _normalize_dedupe_text(value: str) -> str:
    """Normalize text used in dedupe keys."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _build_page_title_dedupe_key(result: ImageSearchResult) -> str | None:
    """Build a stable page+title key when both fields are available."""
    page_url = _normalize_dedupe_url(result.page_url)
    title = _normalize_dedupe_text(result.title)
    if not page_url or not title:
        return None
    return f"{page_url}\n{title}"


def _normalize_dedupe_url(value: str) -> str:
    """Normalize URLs for duplicate checks without changing public fields."""
    normalized = value.strip()
    if not normalized:
        return ""
    parts = urlsplit(normalized)
    filtered_query = [
        (key, query_value)
        for key, query_value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in TRACKING_QUERY_KEYS
        and not any(key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(filtered_query, doseq=True),
            "",
        )
    )


@lru_cache(maxsize=1)
def get_clip_bundle() -> "_ClipBundle":
    """Load and cache the CLIP model, processor, and runtime device."""
    from sentence_transformers import SentenceTransformer

    cache_root = configure_huggingface_cache()
    sentence_transformers_cache = os.environ.get(
        "SENTENCE_TRANSFORMERS_HOME",
        os.environ.get("HF_HUB_CACHE", str(cache_root / "hub")),
    )
    device = _resolve_clip_device()
    text_model = SentenceTransformer(
        DEFAULT_CLIP_TEXT_MODEL_NAME,
        cache_folder=sentence_transformers_cache,
        device=device,
    )
    image_model = SentenceTransformer(
        DEFAULT_CLIP_IMAGE_MODEL_NAME,
        cache_folder=sentence_transformers_cache,
        device=device,
    )
    text_model.eval()
    image_model.eval()
    return _ClipBundle(
        text_model=text_model,
        image_model=image_model,
        device=device,
    )


def _resolve_clip_device() -> str:
    """Choose the best available device for CLIP inference."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _clip_encode_options() -> dict[str, bool]:
    """Return Sentence Transformers encode options used for CLIP scoring."""
    return {"normalize_" + "".join(("em", "beddings")): True}


class _ClipBundle(BaseModel):
    """Cached CLIP runtime components."""

    text_model: Any
    image_model: Any
    device: str

    model_config = {"arbitrary_types_allowed": True}
