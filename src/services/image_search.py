import unicodedata
import logging
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, field_validator


DEFAULT_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_IMAGE_FETCH_LIMIT = 10
DEFAULT_IMAGE_RESULT_LIMIT = 3
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
        A list of normalized image-search results in Tavily-provided order.
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


def select_top_k(
    candidates: list[ImageSearchResult],
    *,
    k: int = DEFAULT_IMAGE_RESULT_LIMIT,
) -> list[ImageSearchResult]:
    """Select the first top-k candidates from the backend-provided order."""
    selected = candidates[:k]
    for idx, item in enumerate(selected, start=1):
        logger.info(
            "image_candidate_selected rank=%s title=%r source=%r image_url=%s",
            idx,
            item.title,
            item.source,
            item.image_url,
        )
    return selected


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
