import unicodedata
from functools import lru_cache
from io import BytesIO
import logging
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import torch
from core.hf_cache import configure_huggingface_cache
from PIL import Image
from pydantic import BaseModel, field_validator

configure_huggingface_cache()

from sentence_transformers import SentenceTransformer


DEFAULT_SEARXNG_BASE_URL = "http://localhost:8008"
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
    """Normalized image-search result returned from SearXNG."""

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
    base_url: str = DEFAULT_SEARXNG_BASE_URL,
) -> list[ImageSearchResult]:
    """Return deduplicated image-search results from SearXNG.

    Args:
        query: User search query string.
        limit: Optional maximum number of deduplicated results to return.
        base_url: SearXNG base URL.

    Returns:
        A list of normalized image-search results.
    """
    normalized_query = unicodedata.normalize("NFKC", query)
    items: list[ImageSearchResult] = []
    seen_image_urls: set[str] = set()
    per_page = 20 if limit is None else max(limit * 2, 20)
    max_pages = 5 if limit is not None else 1

    with httpx.Client() as client:
        for page in range(1, max_pages + 1):
            response = client.get(
                f"{base_url.rstrip('/')}/search",
                params={
                    "q": normalized_query,
                    "format": "json",
                    "categories": "images",
                    "pageno": page,
                    "count": per_page,
                },
                headers=DEFAULT_HEADERS,
                timeout=30.0,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                break
            for item in results:
                try:
                    result = ImageSearchResult(
                        title=str(item.get("title", "")),
                        page_url=str(item.get("url", "")),
                        image_url=str(item.get("img_src", "")),
                        thumbnail_url=str(item.get("thumbnail_src", "")),
                        source=str(item.get("source", "")),
                        engine=str(item.get("engine", "")),
                    )
                except Exception:
                    continue
                if result.image_url in seen_image_urls:
                    continue
                seen_image_urls.add(result.image_url)
                items.append(result)
                if limit is not None and len(items) >= limit:
                    return items[:limit]
            if len(results) < per_page:
                break
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

    with torch.inference_mode():
        text_features = clip_bundle.text_model.encode(
            [normalized_query],
            convert_to_tensor=True,
            device=clip_bundle.device,
            normalize_embeddings=True,
        )
        image_features = clip_bundle.image_model.encode(
            valid_images,
            convert_to_tensor=True,
            device=clip_bundle.device,
            normalize_embeddings=True,
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


def search_and_rerank_images(
    query: str,
    *,
    base_url: str = DEFAULT_SEARXNG_BASE_URL,
    fetch_limit: int = DEFAULT_IMAGE_FETCH_LIMIT,
    rerank_limit: int = DEFAULT_IMAGE_RERANK_LIMIT,
) -> list[ImageSearchResult]:
    """Fetch image candidates from SearXNG, then rerank them with CLIP."""
    candidates = search_images(
        query,
        limit=fetch_limit,
        base_url=base_url,
    )
    logger.info(
        "image_search_candidates query=%r fetched_count=%s fetch_limit=%s",
        unicodedata.normalize("NFKC", query).strip(),
        len(candidates),
        fetch_limit,
    )
    reranked = rerank_with_clip(query, candidates)
    return select_top_k(reranked, k=rerank_limit)


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


@lru_cache(maxsize=1)
def get_clip_bundle() -> "_ClipBundle":
    """Load and cache the CLIP model, processor, and runtime device."""
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
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _ClipBundle(BaseModel):
    """Cached CLIP runtime components."""

    text_model: SentenceTransformer
    image_model: SentenceTransformer
    device: str

    model_config = {"arbitrary_types_allowed": True}
