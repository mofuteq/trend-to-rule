import argparse
import hashlib
import html as html_lib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import lmdb
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

try:
    from readability import Document

    READABILITY_AVAILABLE = True
except ImportError:
    Document = None  # type: ignore[assignment]
    READABILITY_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data"
INPUT_LMDB = DATA_DIR / "rss_db"
OUTPUT_LMDB = DATA_DIR / "article_db"
DEBUG_JSONL_PATH = DATA_DIR / "extracted_articles.jsonl"
LMDB_MAP_SIZE = 512 * 1024 * 1024  # 512MB
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_INTERVAL_SECONDS = 1.5
USER_AGENT = "Mozilla/5.0 (compatible; trend-to-rule/0.1; +https://news.google.com)"

GOOGLE_NEWS_HOSTS = {"news.google.com", "google.com", "www.google.com"}
BLOCKED_HOST_SUFFIXES = (".googleusercontent.com", ".gstatic.com", ".w3.org")
BLOCKED_EXACT_URLS = {
    "https://www.w3.org/2000/svg",
    "http://www.w3.org/2000/svg",
    "https://www.w3.org/1999/xhtml",
    "http://www.w3.org/1999/xhtml",
    "https://www.w3.org/1999/xlink",
    "http://www.w3.org/1999/xlink",
    "https://www.w3.org/XML/1998/namespace",
    "http://www.w3.org/XML/1998/namespace",
}


def parse_args() -> argparse.Namespace:
    """Parse command line options.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fetch article HTML from RSS items and convert body to Markdown."
    )
    parser.add_argument("--input-lmdb", type=Path, default=INPUT_LMDB, help="Input RSS LMDB directory path")
    parser.add_argument("--output-lmdb", type=Path, default=OUTPUT_LMDB, help="Output extracted LMDB directory path")
    parser.add_argument(
        "--debug-jsonl",
        type=Path,
        default=DEBUG_JSONL_PATH,
        help="Debug JSONL output path (append mode)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum items to process (0 = all)")
    parser.add_argument(
        "--interval",
        type=float,
        default=REQUEST_INTERVAL_SECONDS,
        help="Sleep seconds between requests",
    )
    parser.add_argument(
        "--no-readability",
        action="store_true",
        help="Disable readability even if installed",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Process only records not present in output LMDB",
    )
    return parser.parse_args()


def iter_lmdb(path: Path):
    """Yield JSON objects from LMDB values.

    Args:
        path: Source LMDB directory.

    Yields:
        Parsed JSON records.
    """
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=True,
        readahead=False,
        subdir=True,
    )
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield json.loads(value.decode("utf-8"))
    finally:
        env.close()


def load_lmdb_keys(path: Path) -> set[str]:
    """Load all keys from LMDB as UTF-8 strings."""
    if not path.exists():
        return set()

    env = lmdb.open(
        str(path),
        readonly=True,
        lock=True,
        readahead=False,
        subdir=True,
    )
    keys: set[str] = set()
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                keys.add(key.decode("utf-8"))
    finally:
        env.close()
    return keys


def save_lmdb(path: Path, items: list[dict[str, Any]]) -> tuple[int, int]:
    """Write extracted items to LMDB.

    Args:
        path: Output LMDB directory.
        items: Extracted records.

    Returns:
        Tuple of `(inserted_count, updated_count)`.
    """
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(path),
        map_size=LMDB_MAP_SIZE,
        subdir=True,
        max_dbs=1,
        lock=True,
        readahead=False,
    )
    inserted = 0
    updated = 0
    try:
        with env.begin(write=True) as txn:
            for item in items:
                key = str(item.get("dedupe_key", "")).encode("utf-8")
                value = json.dumps(item, ensure_ascii=False).encode("utf-8")
                if txn.put(key, value, overwrite=False):
                    inserted += 1
                else:
                    txn.put(key, value, overwrite=True)
                    updated += 1
    finally:
        env.sync()
        env.close()
    return inserted, updated


def append_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    """Append extracted items to JSONL for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def fm_value(value: Any) -> str:
    """Render safe scalar for YAML-like frontmatter."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_host(host: str) -> str:
    """Normalize hostname for comparison."""
    host = host.lower().strip()
    return host[4:] if host.startswith("www.") else host


def host_matches_expected(candidate_host: str, expected_host: str) -> bool:
    """Check whether candidate host matches expected publisher host."""
    cand = normalize_host(candidate_host)
    exp = normalize_host(expected_host)
    return cand == exp or cand.endswith("." + exp)


def is_google_news_url(url: str) -> bool:
    """Return True if URL points to Google News intermediate page."""
    parsed = urlparse(url)
    return parsed.netloc.lower() in GOOGLE_NEWS_HOSTS and (
        parsed.path.startswith("/rss/articles/")
        or parsed.path.startswith("/articles/")
        or parsed.path.startswith("/read/")
    )


def extract_google_article_id(url: str) -> str:
    """Extract Google News article ID from known path formats."""
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if len(parts) >= 3 and parts[0] == "rss" and parts[1] == "articles":
        return parts[2]
    if len(parts) >= 2 and parts[0] in {"articles", "read"}:
        return parts[1]
    return ""


def build_google_fallback_urls(google_url: str) -> list[str]:
    """Build alternative Google News article endpoints from any article URL."""
    article_id = extract_google_article_id(google_url)
    if not article_id:
        return []
    base = "https://news.google.com"
    return [
        f"{base}/articles/{article_id}",
        f"{base}/read/{article_id}",
        f"{base}/rss/articles/{article_id}",
    ]


def decode_unicode_escaped_url(value: str) -> str:
    """Decode common unicode-escaped URL forms in JSON strings."""
    value = value.replace("\\u002f", "/")
    value = value.replace("\\u003d", "=")
    value = value.replace("\\u0026", "&")
    value = value.replace("\\u003f", "?")
    return html_lib.unescape(value)


def normalize_external_candidate(url: str) -> str:
    """Normalize candidate URL and drop obviously bad targets."""
    if not url.startswith(("http://", "https://")) or url in BLOCKED_EXACT_URLS:
        return ""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host == "w3.org" or host.endswith(".w3.org"):
        return ""
    if any(host.endswith(suffix) for suffix in BLOCKED_HOST_SUFFIXES):
        return ""

    if host in GOOGLE_NEWS_HOSTS or host.endswith(".google.com"):
        qs = parse_qs(parsed.query)
        for key in ("url", "q", "u"):
            if key in qs and qs[key]:
                nested = unquote(qs[key][0]).strip()
                nested_host = urlparse(nested).netloc.lower()
                if nested.startswith(("http://", "https://")) and nested_host and not (
                    nested_host in GOOGLE_NEWS_HOSTS or nested_host.endswith(".google.com")
                ):
                    return nested
        return ""

    if re.search(r"\.(css|js|png|jpg|jpeg|gif|svg|webp|ico|woff2?|ttf)(\?|$)", parsed.path, re.I):
        return ""
    if parsed.path.startswith("/_/"):
        return ""
    return url


def rank_external_candidate(url: str, expected_publisher_host: str = "") -> int:
    """Rank URL quality for article likelihood."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    score = 0

    if expected_publisher_host and host_matches_expected(host, expected_publisher_host):
        score += 100
    if any(token in path.lower() for token in ("article", "news", "story", "fashion", "style")):
        score += 20
    if len(path) > 20:
        score += 10
    if path.count("/") >= 2:
        score += 5
    if re.search(r"\d{4}/\d{2}", path):
        score += 5
    return score


def extract_external_urls_from_google_news_html(
    html: str,
    base_url: str,
    expected_publisher_host: str = "",
) -> list[str]:
    """Extract plausible external article URLs from Google News page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for a in soup.find_all("a", href=True):
        candidates.append(urljoin(base_url, a["href"]).strip())

    unescaped_html = html_lib.unescape(html)
    candidates.extend(re.findall(r"https?://[^\s\"'<>\\]+", unescaped_html))

    for match in re.findall(r"https?:\\\\/\\\\/[^\s\"'<>]+", html):
        candidates.append(match.replace("\\/", "/"))
    for match in re.findall(r"https?:\\u002f\\u002f[^\s\"'<>]+", html):
        candidates.append(decode_unicode_escaped_url(match))

    deduped: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        candidate = normalize_external_candidate(raw)
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)

    return sorted(
        deduped,
        key=lambda u: rank_external_candidate(u, expected_publisher_host=expected_publisher_host),
        reverse=True,
    )


def looks_non_article_response(url: str, html: str) -> bool:
    """Heuristic guard against known non-article pages."""
    if url in BLOCKED_EXACT_URLS:
        return True
    if "is an XML namespace" in html and "Scalable Vector Graphics" in html:
        return True
    if "Copyright (c) 2010-2026 Google LLC" in html and "angular.dev/license" in html:
        return True
    return False


def try_external_candidates(
    external_urls: list[str],
    *,
    strict_host: str,
    headers: dict[str, str],
) -> tuple[str, str] | None:
    """Try external URL candidates and return first valid HTML response."""
    for external_url in external_urls:
        if strict_host and not host_matches_expected(urlparse(external_url).netloc, strict_host):
            continue
        try:
            response = requests.get(
                external_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                continue
            if strict_host and not host_matches_expected(urlparse(response.url).netloc, strict_host):
                continue
            if looks_non_article_response(response.url, response.text):
                continue
            return response.url, response.text
        except requests.RequestException:
            continue
    return None


def decode_google_news_url_via_batchexecute(google_url: str, headers: dict[str, str]) -> str:
    """Decode Google News article URL using batchexecute RPC."""
    article_id = extract_google_article_id(google_url)
    if not article_id:
        return ""

    res = requests.get(
        google_url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    marker = soup.select_one("[data-n-a-sg][data-n-a-ts]")
    if not marker:
        return ""

    signature = str(marker.get("data-n-a-sg") or "").strip()
    timestamp = str(marker.get("data-n-a-ts") or "").strip()
    if not signature or not timestamp:
        return ""

    rpc_payload = [
        [
            "Fbv4je",
            (
                "[\"garturlreq\",[[\"en-US\",\"US\",[\"FINANCE_TOP_INDICES\",\"WEB_TEST_1_0_0\"],"
                "null,null,1,1,\"US:en\",null,180,null,null,null,null,null,0,null,null,"
                "[1608992183,723341000]],\"en-US\",\"US\",1,[2,3,4,8],1,0,\"655000234\",0,0,null,0],"
                f"\"{article_id}\",{timestamp},\"{signature}\"]"
            ),
            None,
            "generic",
        ]
    ]

    rpc_headers = dict(headers)
    rpc_headers["Content-Type"] = "application/x-www-form-urlencoded;charset=UTF-8"
    rpc_headers["Referer"] = "https://news.google.com/"

    rpc = requests.post(
        "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
        data={"f.req": json.dumps([rpc_payload], separators=(",", ":"))},
        headers=rpc_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    rpc.raise_for_status()

    text = html_lib.unescape(rpc.text)
    for raw in re.findall(r"https?://[^\s\"'<>\\]+", text):
        candidate = normalize_external_candidate(decode_unicode_escaped_url(raw))
        if candidate:
            return candidate
    return ""


def resolve_and_fetch_html(url: str, expected_publisher_host: str = "") -> tuple[str, str]:
    """Fetch HTML and resolve Google News intermediate URLs to publisher pages."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    response.raise_for_status()

    final_url = response.url
    html = response.text
    if not is_google_news_url(final_url):
        return final_url, html

    strict_host = normalize_host(expected_publisher_host)

    candidate_pages = [(final_url, html)]
    for fallback_url in build_google_fallback_urls(final_url):
        try:
            fallback = requests.get(
                fallback_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            fallback.raise_for_status()
            candidate_pages.append((fallback.url, fallback.text))
        except requests.RequestException:
            continue

    for page_url, page_html in candidate_pages:
        external_urls = extract_external_urls_from_google_news_html(
            page_html,
            page_url,
            expected_publisher_host=expected_publisher_host,
        )
        resolved = try_external_candidates(external_urls, strict_host=strict_host, headers=headers)
        if resolved:
            return resolved
        resolved = try_external_candidates(external_urls, strict_host="", headers=headers)
        if resolved:
            return resolved

    try:
        decoded_url = decode_google_news_url_via_batchexecute(final_url, headers=headers)
        if decoded_url:
            resolved = try_external_candidates([decoded_url], strict_host=strict_host, headers=headers)
            if resolved:
                return resolved
            resolved = try_external_candidates([decoded_url], strict_host="", headers=headers)
            if resolved:
                return resolved
    except requests.RequestException:
        pass

    return final_url, html


def infer_query(record: dict[str, Any]) -> str:
    """Infer search query string from RSS metadata."""
    feed_url = str(record.get("feed_url") or "").strip()
    if not feed_url:
        return ""
    query_values = parse_qs(urlparse(feed_url).query).get("q", [])
    return query_values[0].replace("+", " ").strip() if query_values else ""


def infer_locale_country_from_feed_url(feed_url: str) -> tuple[str, str]:
    """Infer locale and country from Google News RSS URL query params."""
    if not feed_url:
        return "", ""
    qs = parse_qs(urlparse(feed_url).query)
    hl = (qs.get("hl") or [""])[0].strip()
    gl = (qs.get("gl") or [""])[0].strip().upper()
    ceid = (qs.get("ceid") or [""])[0].strip()

    locale = hl
    country = gl
    if ceid and ":" in ceid:
        ceid_country, ceid_lang = ceid.split(":", 1)
        if not country and ceid_country:
            country = ceid_country.strip().upper()
        if not locale and ceid_lang:
            locale = ceid_lang.strip()
    return locale, country


def infer_locale_country_from_html(html: str) -> tuple[str, str]:
    """Infer locale and country from page html tags when available."""
    soup = BeautifulSoup(html, "html.parser")

    locale = ""
    html_tag = soup.find("html")
    if isinstance(html_tag, Tag):
        locale = str(html_tag.get("lang") or "").strip()

    if not locale:
        og_locale = soup.find("meta", attrs={"property": "og:locale"})
        if isinstance(og_locale, Tag):
            locale = str(og_locale.get("content") or "").strip()

    locale = locale.replace("_", "-")
    country = ""
    if "-" in locale:
        maybe_country = locale.split("-")[-1].strip()
        if len(maybe_country) == 2:
            country = maybe_country.upper()

    return locale, country


def infer_expected_publisher_host(record: dict[str, Any]) -> str:
    """Infer publisher host from RSS record metadata."""
    raw_entry = record.get("raw_entry")
    if not isinstance(raw_entry, dict):
        return ""
    source = raw_entry.get("source")
    if not isinstance(source, dict):
        return ""
    source_href = str(source.get("href") or "").strip()
    return normalize_host(urlparse(source_href).netloc) if source_href else ""


def is_unresolved_article_url(url: str) -> bool:
    """Return True when URL still looks like non-article target."""
    host = urlparse(url).netloc.lower()
    return host in GOOGLE_NEWS_HOSTS or host.endswith(".google.com") or host.endswith("w3.org")


def pick_main_container(soup: BeautifulSoup) -> Tag:
    """Pick the most likely main article container."""
    for name in ("article", "main"):
        found = soup.find(name)
        if isinstance(found, Tag):
            return found

    candidates = soup.find_all(["section", "div"])
    best: Tag = soup.body if isinstance(soup.body, Tag) else soup  # type: ignore[assignment]
    best_score = -1

    for tag in candidates:
        text = clean_text(tag.get_text(" ", strip=True))
        if len(text) < 200:
            continue
        class_id = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
        penalty = 500 if re.search(r"nav|menu|footer|header|sidebar|ad|promo|banner", class_id, re.I) else 0
        score = len(text) - penalty
        if score > best_score:
            best_score = score
            best = tag

    return best


def to_markdown(node: Any, base_url: str = "") -> str:
    """Convert a BeautifulSoup node tree to simple Markdown."""
    if isinstance(node, NavigableString):
        return clean_text(str(node))
    if isinstance(node, BeautifulSoup):
        return " ".join(filter(None, (to_markdown(c, base_url=base_url) for c in node.children)))
    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()
    if name in {"script", "style", "noscript", "svg", "iframe", "form"}:
        return ""

    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        text = clean_text(" ".join(to_markdown(c, base_url=base_url) for c in node.children))
        return f"{'#' * level} {text}\n\n" if text else ""

    if name == "p":
        text = clean_text(" ".join(to_markdown(c, base_url=base_url) for c in node.children))
        return f"{text}\n\n" if text else ""

    if name == "br":
        return "\n"

    if name in {"ul", "ol"}:
        lines: list[str] = []
        for i, li in enumerate(node.find_all("li", recursive=False), start=1):
            text = clean_text(" ".join(to_markdown(c, base_url=base_url) for c in li.children))
            if text:
                prefix = f"{i}." if name == "ol" else "-"
                lines.append(f"{prefix} {text}")
        return "\n".join(lines) + "\n\n" if lines else ""

    if name == "blockquote":
        text = clean_text(" ".join(to_markdown(c, base_url=base_url) for c in node.children))
        return f"> {text}\n\n" if text else ""

    if name == "a":
        text = clean_text(" ".join(to_markdown(c, base_url=base_url) for c in node.children))
        href = (node.get("href") or "").strip()
        if href and base_url:
            href = urljoin(base_url, href)
        return f"[{text}]({href})" if text and href else text

    if name in {"strong", "b"}:
        text = clean_text(" ".join(to_markdown(c, base_url=base_url) for c in node.children))
        return f"**{text}**" if text else ""

    if name in {"em", "i"}:
        text = clean_text(" ".join(to_markdown(c, base_url=base_url) for c in node.children))
        return f"*{text}*" if text else ""

    if name == "code":
        text = clean_text(node.get_text(" ", strip=True))
        return f"`{text}`" if text else ""

    return " ".join(filter(None, (to_markdown(c, base_url=base_url) for c in node.children)))


def html_to_markdown(html: str, base_url: str = "", use_readability: bool = True) -> str:
    """Extract article-like content and convert it to Markdown."""
    if use_readability and READABILITY_AVAILABLE:
        try:
            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            soup = BeautifulSoup(summary_html, "html.parser")
            markdown = to_markdown(soup.body if soup.body else soup, base_url=base_url)
            markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
            if markdown:
                return markdown
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")
    markdown = to_markdown(pick_main_container(soup), base_url=base_url)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def build_markdown_document(
    *,
    source_url: str,
    resolved_url: str,
    title: str,
    publisher: str,
    published_at: str,
    query: str,
    ingested_at: str,
    locale: str,
    country: str,
    word_count: int,
    section_count: int,
    markdown_body: str,
) -> str:
    """Build a frontmatter-style markdown document."""
    lines = [
        "---",
        f"source_url: {fm_value(source_url)}",
        f"resolved_url: {fm_value(resolved_url)}",
        f"title: {fm_value(title)}",
        f"publisher: {fm_value(publisher)}",
        f"published_at: {fm_value(published_at)}",
        f"query: {fm_value(query)}",
        f"ingested_at: {fm_value(ingested_at)}",
        f"locale: {fm_value(locale)}",
        f"country: {fm_value(country)}",
        f"word_count: {word_count}",
        f"section_count: {section_count}",
        "---",
        "",
        f"# {title}",
        "",
    ]
    body = markdown_body.strip()
    if body:
        lines.append(body)
    return "\n".join(lines).strip() + "\n"


def process_record(record: dict[str, Any], use_readability: bool = True) -> dict[str, Any]:
    """Fetch and transform one RSS record."""
    source_url = str(record.get("link") or "").strip()
    title = str(record.get("title") or "")
    publisher = str(record.get("source") or "")
    published_at = str(record.get("published_at") or "")
    ingested_at = str(record.get("ingested_at") or "")
    query = infer_query(record)
    feed_locale = str(record.get("locale") or "")
    feed_country = str(record.get("country") or "")
    if not feed_locale or not feed_country:
        inferred_locale, inferred_country = infer_locale_country_from_feed_url(str(record.get("feed_url") or ""))
        feed_locale = feed_locale or inferred_locale
        feed_country = feed_country or inferred_country
    dedupe_key = str(record.get("dedupe_key") or "")
    raw_entry = record.get("raw_entry") if isinstance(record.get("raw_entry"), dict) else {}

    expected_host = infer_expected_publisher_host(record)
    final_url, html = resolve_and_fetch_html(source_url, expected_publisher_host=expected_host)
    if is_unresolved_article_url(final_url):
        summary_html = str((raw_entry or {}).get("summary") or "")
        summary_text = clean_text(BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True))
        markdown_body = summary_text or (
            "Source article URL could not be resolved from Google News redirect.\n\n"
            f"Original URL: {source_url}"
        )
    else:
        markdown_body = html_to_markdown(html, base_url=final_url, use_readability=use_readability)
    page_locale, page_country = infer_locale_country_from_html(html)
    locale = page_locale or feed_locale
    country = page_country or feed_country
    word_count = len(re.findall(r"\b\w+\b", markdown_body))
    section_count = len(re.findall(r"(?m)^##\s+", markdown_body))

    document = build_markdown_document(
        source_url=source_url,
        resolved_url=final_url,
        title=title,
        publisher=publisher,
        published_at=published_at,
        query=query,
        ingested_at=ingested_at,
        locale=locale,
        country=country,
        word_count=word_count,
        section_count=section_count,
        markdown_body=markdown_body,
    )

    return {
        "dedupe_key": dedupe_key,
        "source_url": source_url,
        "resolved_url": final_url,
        "title": title,
        "publisher": publisher,
        "published_at": published_at,
        "query": query,
        "ingested_at": ingested_at,
        "locale": locale,
        "country": country,
        "word_count": word_count,
        "section_count": section_count,
        "markdown": document,
        "markdown_body_chars": len(markdown_body),
        "markdown_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    """Run extraction pipeline from RSS LMDB to extracted LMDB and debug JSONL."""
    args = parse_args()
    args.output_lmdb.mkdir(parents=True, exist_ok=True)
    args.debug_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    succeeded = 0
    failed = 0
    total_inserted = 0
    total_updated = 0
    skipped_existing = 0
    existing_keys: set[str] = set()
    if args.only_new:
        existing_keys = load_lmdb_keys(args.output_lmdb)
        print(f"Loaded existing keys: {len(existing_keys)} from {args.output_lmdb}")

    for record in iter_lmdb(args.input_lmdb):
        if args.limit > 0 and processed >= args.limit:
            break

        dedupe_key = str(record.get("dedupe_key") or "")
        if args.only_new and dedupe_key and dedupe_key in existing_keys:
            processed += 1
            skipped_existing += 1
            print(f"[{processed}] skip existing title={str(record.get('title') or '')[:80]}")
            continue

        processed += 1
        try:
            extracted = process_record(record, use_readability=not args.no_readability)
            inserted, updated = save_lmdb(args.output_lmdb, [extracted])
            append_jsonl(args.debug_jsonl, [extracted])
            total_inserted += inserted
            total_updated += updated
            if args.only_new and dedupe_key:
                existing_keys.add(dedupe_key)
            succeeded += 1
            print(
                f"[{processed}] ok  body_chars={extracted['markdown_body_chars']} "
                f"title={extracted['title'][:80]} "
                f"(inserted={inserted}, updated={updated})"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{processed}] ng  title={record.get('title', '')[:80]} err={exc}")

        time.sleep(max(0.0, args.interval))

    print(
        f"Done: processed={processed}, succeeded={succeeded}, failed={failed}, skipped_existing={skipped_existing}, "
        f"lmdb={args.output_lmdb} (inserted={total_inserted}, updated={total_updated}), "
        f"debug_jsonl={args.debug_jsonl}"
    )


if __name__ == "__main__":
    main()
