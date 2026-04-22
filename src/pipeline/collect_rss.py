# %%
import argparse
import hashlib
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TypedDict
from urllib.parse import parse_qs, quote_plus, urlparse

import certifi
import feedparser
import lmdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data"
OUT_PATH = DATA_DIR / "rss_db"
DEBUG_JSONL_PATH = DATA_DIR / "raw_feed_items.jsonl"
DEFAULT_LOG_PATH = DATA_DIR / "logs" / "collect_rss.log"
MAP_SIZE = 128 * 1024 * 1024  # 128MB
REQUEST_INTERVAL_SECONDS = 1.5
CHECKPOINT_PATH = DATA_DIR / "collect_rss_checkpoint.json"
logger = logging.getLogger(__name__)


class FeedItem(TypedDict):
    """RSS item record stored in LMDB and JSONL."""

    feed_url: str
    ingested_at: str
    title: str
    link: str
    published_at: str
    source: str
    locale: str
    country: str
    dedupe_key: str
    raw_entry: dict[str, Any]


class TaskCheckpoint(TypedDict):
    """Checkpoint for resuming task loop."""

    last_completed_task_idx: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Collect Google News RSS and store results into LMDB/JSONL."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed task index in checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Checkpoint JSON path used by --resume.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete checkpoint file before running.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Log file path. Rotated daily and keeps 5 days by default.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    """Configure console and daily rotating file logging.

    Args:
        log_path: Log file path. Parent directories are created automatically.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=5,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def iso_now() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def normalize_title(title: str) -> str:
    """Normalize title for stable dedupe hashing.

    Args:
        title: Raw article title.

    Returns:
        Lower-cased title with collapsed whitespace.
    """
    return " ".join((title or "").strip().split()).lower()


def parse_published(entry: Mapping[str, Any]) -> str:
    """Extract published time from a feed entry.

    Args:
        entry: One feed entry from feedparser.

    Returns:
        ISO 8601 datetime string in UTC, or empty string if unavailable.
    """
    dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if dt_struct:
        dt = datetime(*dt_struct[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return ""


def extract_source(feed: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
    """Extract source title from entry or feed metadata.

    Args:
        feed: Parsed feed metadata.
        entry: One feed entry from feedparser.

    Returns:
        Source name with surrounding whitespace removed.
    """
    src = ""
    if entry.get("source") and isinstance(entry["source"], dict):
        src = entry["source"].get("title") or ""
    return (src or feed.get("title") or "").strip()


def extract_locale_country(feed_url: str) -> tuple[str, str]:
    """Extract locale and country from Google News RSS query params.

    Args:
        feed_url: Google News RSS URL.

    Returns:
        Tuple of `(locale, country)` such as `("en-US", "US")`.
    """
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


def make_dedupe_key(title: str, source: str, published_at: str, link: str) -> str:
    """Build deterministic hash key for deduplication.

    Args:
        title: Article title.
        source: Source name.
        published_at: Published timestamp in ISO 8601.
        link: Article URL.

    Returns:
        SHA-256 hex digest for dedupe.
    """
    base = f"{normalize_title(title)}|{source.strip().lower()}|{published_at[:10]}|{(link or '').strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def fetch_rss_bytes(url: str) -> bytes:
    """Fetch RSS response body as bytes.

    Args:
        url: RSS endpoint URL.

    Returns:
        Raw response bytes.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; trend-to-rule/0.1)"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        return r.read()


def fetch_first_n(feed_url: str, n: int = 5) -> list[FeedItem]:
    """Fetch and normalize first N entries from one RSS feed.

    Args:
        feed_url: RSS endpoint URL.
        n: Maximum number of entries to collect.

    Returns:
        Normalized feed items.
    """
    data = fetch_rss_bytes(feed_url)
    d = feedparser.parse(data)
    locale, country = extract_locale_country(feed_url)

    feed: Mapping[str, Any] = d.feed or {}
    items: list[FeedItem] = []
    for entry in (d.entries or [])[:n]:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        published_at = parse_published(entry)
        source = extract_source(feed, entry)
        ingested_at = iso_now()

        dedupe_key = make_dedupe_key(title, source, published_at, link)

        item: FeedItem = {
            "feed_url": feed_url,
            "ingested_at": ingested_at,
            "title": title,
            "link": link,
            "published_at": published_at or ingested_at,
            "source": source,
            "locale": locale,
            "country": country,
            "dedupe_key": dedupe_key,
            "raw_entry": dict(entry),
        }
        items.append(item)
    return items


def save_lmdb(path: Path, items: list[FeedItem]) -> tuple[int, int]:
    """Write feed items to LMDB.

    Args:
        path: LMDB directory path.
        items: Feed items to store.

    Returns:
        Tuple of `(inserted_count, updated_count)`.
    """
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(path),
        map_size=MAP_SIZE,
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
                key = item["dedupe_key"].encode("utf-8")
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


def append_jsonl(path: Path, items: list[FeedItem]) -> None:
    """Append feed items to JSONL for debugging.

    Args:
        path: JSONL file path.
        items: Feed items to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path) -> TaskCheckpoint:
    """Load task checkpoint.

    Args:
        path: Checkpoint file path.

    Returns:
        Checkpoint dict with last completed task index.
    """
    if not path.is_file():
        return {"last_completed_task_idx": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"last_completed_task_idx": 0}
    last_idx = payload.get("last_completed_task_idx")
    if not isinstance(last_idx, int) or last_idx < 0:
        last_idx = 0
    return {"last_completed_task_idx": last_idx}


def save_checkpoint(path: Path, task_idx: int) -> None:
    """Save task checkpoint.

    Args:
        path: Checkpoint file path.
        task_idx: Last successfully completed task index.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: TaskCheckpoint = {"last_completed_task_idx": int(task_idx)}
    path.write_text(json.dumps(payload, ensure_ascii=False,
                    indent=2), encoding="utf-8")


def main() -> None:
    """Collect multiple RSS queries and persist results."""
    args = parse_args()
    setup_logging(args.log_path)
    logger.info("log_path=%s rotation=daily backup_count=5", args.log_path)
    if args.reset_checkpoint and args.checkpoint_path.exists():
        args.checkpoint_path.unlink()

    current_year = datetime.now(timezone.utc).year
    years = list(range(2021, current_year + 1))

    query_groups = {
        "canonical": [
            "quiet luxury dress code",
            "understated luxury style baseline",
            "minimalist luxury wardrobe",
            "old money aesthetic baseline",
            "timeless fashion essentials",
            "fashion norms understated elegance",
            "classic preppy wardrobe",
            "minimalist fashion normcore",
            "business casual luxury baseline",
            "luxury fashion signaling norms",
            "fashion culture understated elegance",
            "fashion history minimalist luxury",
            "quiet luxury cultural meaning",
            "status signaling in fashion history",
            "classic taste fashion culture",
            "fashion norms wealth signaling",
            "timeless style cultural norms",
            "fashion restraint elegance history",
            "fashion culture understated elegance",
            "fashion history minimalist style",
            "status signaling fashion history",
            "timeless style cultural norms",
            "default wardrobe fashion norms",
            "classic style social signaling",
            "professional dress code baseline",
            "understated elegance cultural meaning",
            # Added and expanded mens-focused queries:
            "men's quiet luxury style",
            "men's minimalist luxury wardrobe",
            "menswear understated elegance",
            "men's fashion classic essentials",
            "timeless menswear style",
            "men's business casual luxury",
            "men's old money aesthetic",
            "men's fashion preppy wardrobe",
            "men's professional dress code",
        ],
        "emerging": [
            f"quiet luxury trend {year}" for year in years
        ] + [
            f"quiet luxury backlash {year}" for year in years
        ] + [
            f"stealth wealth fashion trend {year}" for year in years
        ] + [
            f"logo fashion comeback {year}" for year in years
        ] + [
            f"maximalism comeback fashion {year}" for year in years
        ] + [
            f"mob wife aesthetic trend {year}" for year in years
        ] + [
            f"fashion microtrends {year} luxury" for year in years
        ] + [
            f"Gen Z luxury shift {year}" for year in years
        ] + [
            f"fashion trend adoption {year} luxury" for year in years
        ] + [
            # Mens-focused emerging coverage
            f"men's quiet luxury trend {year}" for year in years
        ] + [
            f"men's minimalist luxury trend {year}" for year in years
        ] + [
            f"menswear understated elegance trend {year}" for year in years
        ] + [
            f"men's old money fashion trend {year}" for year in years
        ],
        "proxy_culture": [
            "luxury fashion market shift",
            "luxury brand strategy",
            "fashion retail luxury slowdown",
            "consumer preference understated luxury",
            "status signaling fashion culture",
            "post pandemic fashion values",
            "fashion aesthetics social media shift",
            # Menswear and culture proxies
            "menswear luxury market shift",
            "men's luxury fashion strategy",
            "men's fashion retail trends",
            "men's consumer preference luxury",
            "men's fashion values culture",
        ],
    }

    locales = [
        {"hl": "en-US", "gl": "US", "ceid": "US:en"},
        {"hl": "en-GB", "gl": "GB", "ceid": "GB:en"},
        {"hl": "fr", "gl": "FR", "ceid": "FR:fr"},
        {"hl": "it", "gl": "IT", "ceid": "IT:it"},
    ]

    total_items = 0
    total_inserted = 0
    total_updated = 0

    total_tasks = sum(len(group_queries)
                      for group_queries in query_groups.values()) * len(locales)
    task_idx = 0
    checkpoint = load_checkpoint(args.checkpoint_path) if args.resume else {
        "last_completed_task_idx": 0}
    resume_from = checkpoint["last_completed_task_idx"]
    if args.resume and resume_from > 0:
        logger.info(
            "resume enabled: skipping tasks <= %s checkpoint=%s",
            resume_from,
            args.checkpoint_path,
        )

    for group_name, group_queries in query_groups.items():
        for query in group_queries:
            for locale_cfg in locales:
                task_idx += 1
                if args.resume and task_idx <= resume_from:
                    continue
                query_encoded = quote_plus(query)
                feed_url = (
                    "https://news.google.com/rss/search?"
                    f"q={query_encoded}"
                    f"&hl={locale_cfg['hl']}"
                    f"&gl={locale_cfg['gl']}"
                    f"&ceid={locale_cfg['ceid']}"
                )

                items = fetch_first_n(feed_url, n=50)
                inserted, updated = save_lmdb(OUT_PATH, items)
                append_jsonl(DEBUG_JSONL_PATH, items)
                save_checkpoint(args.checkpoint_path, task_idx)

                total_items += len(items)
                total_inserted += inserted
                total_updated += updated

                logger.info(
                    "[%s/%s] group=%s query=%s locale=%s items=%s inserted=%s updated=%s",
                    task_idx,
                    total_tasks,
                    group_name,
                    query,
                    locale_cfg["hl"],
                    len(items),
                    inserted,
                    updated,
                )

                for i, item in enumerate(items, 1):
                    logger.info(
                        "[%s/%s] item=%s title=%s link=%s source=%s published_at=%s",
                        task_idx,
                        total_tasks,
                        i,
                        item["title"],
                        item["link"],
                        item["source"],
                        item["published_at"],
                    )

                time.sleep(REQUEST_INTERVAL_SECONDS)

    logger.info(
        "done: total_items=%s output_lmdb=%s inserted=%s updated=%s",
        total_items,
        OUT_PATH,
        total_inserted,
        total_updated,
    )
    logger.info("debug_jsonl=%s", DEBUG_JSONL_PATH)
    if args.checkpoint_path.exists():
        logger.info("checkpoint=%s", args.checkpoint_path)


if __name__ == "__main__":
    main()

# %%
