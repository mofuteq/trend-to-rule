"""Tavily text evidence retrieval and source normalization."""

import logging
import os
import unicodedata
from html import escape
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from core.models import WebSource

DEFAULT_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_TEXT_MAX_RESULTS = 5
MAX_TAVILY_RESULTS = 20
MAX_SNIPPET_CHARS = 2400
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

QueryKind = Literal["canonical", "emerging"]
logger = logging.getLogger(__name__)


def search_text_sources(
    query: str,
    *,
    query_kind: QueryKind,
    max_results: int | None = None,
    api_key: str | None = None,
    search_depth: str = "basic",
    include_raw_content: bool = False,
    search_url: str = DEFAULT_TAVILY_SEARCH_URL,
) -> list[WebSource]:
    """Return normalized Tavily text results for one evidence lane.

    Tavily responses are normalized into ``WebSource`` before any downstream
    prompt context is built; raw provider payloads never flow into the LLM path.
    """
    normalized_query = unicodedata.normalize("NFKC", query or "").strip()
    resolved_api_key = (
        os.getenv("TAVILY_API_KEY", "") if api_key is None else api_key
    ).strip()
    if not normalized_query:
        return []
    if not resolved_api_key:
        logger.warning("Tavily text search skipped: TAVILY_API_KEY is not configured")
        return []

    normalized_depth = _normalize_search_depth(search_depth)
    result_limit = max(1, min(max_results or DEFAULT_TEXT_MAX_RESULTS, MAX_TAVILY_RESULTS))
    try:
        with httpx.Client() as client:
            response = client.post(
                search_url,
                json={
                    "query": normalized_query,
                    "include_answer": False,
                    "include_images": False,
                    "include_raw_content": include_raw_content,
                    "search_depth": normalized_depth,
                    "max_results": result_limit,
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
        logger.warning("Tavily text search failed: %s", err)
        return []

    return normalize_tavily_text_results(
        payload,
        query_kind=query_kind,
        include_raw_content=include_raw_content,
    )


def normalize_tavily_text_results(
    payload: Any,
    *,
    query_kind: QueryKind,
    include_raw_content: bool = False,
) -> list[WebSource]:
    """Normalize a Tavily text-search payload into stable source models."""
    if not isinstance(payload, dict):
        logger.warning("Tavily text search returned malformed payload: %r", payload)
        return []
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []

    sources: list[WebSource] = []
    seen_urls: set[str] = set()
    prefix = _source_id_prefix(query_kind)
    for result in results:
        if not isinstance(result, dict):
            continue
        source = _normalize_tavily_text_result(
            result,
            query_kind=query_kind,
            source_id=f"{prefix}{len(sources) + 1:02d}",
            include_raw_content=include_raw_content,
        )
        if source is None:
            continue
        dedupe_key = normalize_source_url(source.url)
        if not dedupe_key or dedupe_key in seen_urls:
            continue
        seen_urls.add(dedupe_key)
        sources.append(source.model_copy(update={"url": dedupe_key}))
    return sources


def dedupe_sources_by_url(
    *,
    canonical_sources: list[WebSource],
    emerging_sources: list[WebSource],
) -> tuple[list[WebSource], list[WebSource]]:
    """Deduplicate canonical and emerging sources by normalized URL."""
    seen_urls: set[str] = set()
    deduped_canonical = _dedupe_source_lane(
        canonical_sources,
        query_kind="canonical",
        seen_urls=seen_urls,
    )
    deduped_emerging = _dedupe_source_lane(
        emerging_sources,
        query_kind="emerging",
        seen_urls=seen_urls,
    )
    return deduped_canonical, deduped_emerging


def sources_to_prompt_context(
    sources: list[WebSource],
    *,
    label: str,
) -> str:
    """Build compact LLM context from normalized source models."""
    context_items: list[str] = []
    for source in sources:
        lines = [
            f"[{source.source_id}]",
            f"- lane={label}",
            f"- title={source.title}",
            f"- url={source.url}",
        ]
        if source.published_at:
            lines.append(f"- published_at={source.published_at}")
        if source.score is not None:
            lines.append(f"- score={source.score:.4f}")
        lines.append(f"- snippet={source.snippet}")
        context_items.append("\n".join(lines))
    return "\n\n".join(context_items)


def sources_to_table_rows(sources: list[WebSource]) -> list[dict[str, str]]:
    """Convert normalized web sources into UI table rows."""
    rows: list[dict[str, str]] = []
    for source in sources:
        rows.append(
            {
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "published_at": source.published_at or "",
                "provider": source.provider,
            }
        )
    return rows


def build_web_sources_html_table(rows: list[dict[str, str]]) -> str:
    """Build an HTML table for normalized web evidence sources."""
    if not rows:
        return ""

    body_rows: list[str] = []
    for row in rows:
        source_id = escape(row.get("source_id", ""))
        title = escape(row.get("title", ""))
        published_at = escape(row.get("published_at", ""))
        provider = escape(row.get("provider", ""))
        url = row.get("url", "").strip()
        title_cell = (
            f'<a href="{escape(url, quote=True)}" target="_blank">{title}</a>'
            if url
            else title
        )
        body_rows.append(
            "<tr>"
            f"<td>{source_id}</td>"
            f"<td>{title_cell}</td>"
            f"<td>{published_at}</td>"
            f"<td>{provider}</td>"
            "</tr>"
        )

    body_html = "".join(body_rows)
    return (
        "<table>"
        "<thead><tr><th>source_id</th><th>title</th><th>published_at</th>"
        "<th>provider</th></tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def normalize_source_url(raw_url: str) -> str:
    """Normalize source URLs for stable deduplication."""
    normalized = unicodedata.normalize("NFKC", raw_url or "").strip()
    if not normalized:
        return ""
    parts = urlsplit(normalized)
    if not parts.scheme or not parts.netloc:
        return normalized
    filtered_query = [
        (key, query_value)
        for key, query_value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in TRACKING_QUERY_KEYS
        and not any(key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            urlencode(filtered_query, doseq=True),
            "",
        )
    )


def _normalize_tavily_text_result(
    result: dict[str, Any],
    *,
    query_kind: QueryKind,
    source_id: str,
    include_raw_content: bool,
) -> WebSource | None:
    title = _compact_text(str(result.get("title") or "")).strip() or "Untitled"
    url = normalize_source_url(str(result.get("url") or ""))
    if not url:
        return None

    snippet = ""
    if include_raw_content:
        snippet = _compact_text(str(result.get("raw_content") or ""))
    if not snippet:
        snippet = _compact_text(str(result.get("content") or ""))
    if not snippet:
        snippet = title
    snippet = _truncate_text(snippet, MAX_SNIPPET_CHARS)

    return WebSource(
        source_id=source_id,
        query_kind=query_kind,
        title=title,
        url=url,
        snippet=snippet,
        published_at=_extract_published_at(result),
        score=_extract_score(result),
    )


def _dedupe_source_lane(
    sources: list[WebSource],
    *,
    query_kind: QueryKind,
    seen_urls: set[str],
) -> list[WebSource]:
    deduped: list[WebSource] = []
    prefix = _source_id_prefix(query_kind)
    for source in sources:
        dedupe_key = normalize_source_url(source.url)
        if not dedupe_key or dedupe_key in seen_urls:
            continue
        seen_urls.add(dedupe_key)
        deduped.append(
            source.model_copy(
                update={
                    "source_id": f"{prefix}{len(deduped) + 1:02d}",
                    "query_kind": query_kind,
                    "url": dedupe_key,
                }
            )
        )
    return deduped


def _normalize_search_depth(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"basic", "advanced"}:
        return normalized
    return "basic"


def _source_id_prefix(query_kind: QueryKind) -> str:
    return "C" if query_kind == "canonical" else "E"


def _extract_published_at(result: dict[str, Any]) -> str | None:
    for key in ("published_at", "published_date", "date"):
        value = str(result.get(key) or "").strip()
        if value:
            return value
    return None


def _extract_score(result: dict[str, Any]) -> float | None:
    value = result.get("score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").split())


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."
