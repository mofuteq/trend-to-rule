import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, field_validator


DEFAULT_SEARXNG_BASE_URL = "http://localhost:8008"
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


class ImageSearchResult(BaseModel):
    """Normalized image-search result returned from SearXNG."""

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
