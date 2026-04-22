import argparse
import hashlib
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import re
import sys
import time
import uuid
import lmdb
import torch
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    from services.chat import infer_attribute
else:
    from services.chat import infer_attribute

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data"
INPUT_LMDB = DATA_DIR / "article_db"
DEFAULT_LOG_PATH = DATA_DIR / "logs" / "embed_articles.log"

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_COLLECTION = "article_markdown_bge_m3"
DEFAULT_QDRANT_URL = "http://localhost:6333"
LMDB_MAP_SIZE = 1024 * 1024 * 1024  # 1GB
REQUEST_INTERVAL_SECONDS = 0.0
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Embed extracted markdown with bge-m3 (dense+sparse) and upsert to Qdrant."
    )
    parser.add_argument("--input-lmdb", type=Path,
                        default=INPUT_LMDB, help="Input extracted LMDB path")
    parser.add_argument("--qdrant-url", type=str,
                        default=DEFAULT_QDRANT_URL, help="Qdrant endpoint URL")
    parser.add_argument("--qdrant-api-key", type=str,
                        default=os.getenv("QDRANT_API_KEY", ""), help="Qdrant API key")
    parser.add_argument("--collection", type=str,
                        default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--model-name", type=str,
                        default=DEFAULT_MODEL_NAME, help="FlagEmbedding model name")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Embedding device. 'auto' prefers mps on Mac, then cuda, then cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for embedding")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of docs (0 = all)")
    parser.add_argument(
        "--interval",
        type=float,
        default=REQUEST_INTERVAL_SECONDS,
        help="Sleep seconds between each record processing",
    )
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate collection")
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Embed only chunks not already present in Qdrant",
    )
    parser.add_argument(
        "--update-payload-only",
        action="store_true",
        help="Update payload only without embedding/upserting vectors",
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

def iter_lmdb(path: Path):
    """Yield JSON records from LMDB values."""
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=True,
        readahead=False,
        subdir=True,
        map_size=LMDB_MAP_SIZE,
    )
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield json.loads(value.decode("utf-8"))
    finally:
        env.close()


def resolve_device(device_arg: str) -> str:
    """Resolve runtime device from CLI arg."""
    if device_arg != "auto":
        return device_arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_qdrant_url(raw_url: str) -> str:
    """Normalize and validate qdrant url."""
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("qdrant url is empty")

    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"invalid qdrant url scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError(f"invalid qdrant url host: {url}")
    if parsed.hostname.startswith(".") or ".." in parsed.hostname:
        raise ValueError(f"invalid qdrant url host: {parsed.hostname}")

    port_part = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port_part}"


def stable_index(token: str) -> int:
    """Map token key to stable positive integer index for sparse vectors."""
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def lexical_weights_to_sparse_vector(weights: dict[Any, float]) -> models.SparseVector:
    """Convert bge-m3 lexical weights to Qdrant sparse vector."""
    merged: dict[int, float] = {}
    for key, value in weights.items():
        if not value:
            continue
        try:
            idx = int(key)
        except (TypeError, ValueError):
            idx = stable_index(str(key))
        merged[idx] = merged.get(idx, 0.0) + float(value)

    if not merged:
        return models.SparseVector(indices=[], values=[])

    pairs = sorted(merged.items(), key=lambda x: x[0])
    indices = [idx for idx, _ in pairs]
    values = [val for _, val in pairs]
    return models.SparseVector(indices=indices, values=values)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    recreate: bool = False,
) -> None:
    """Create collection for hybrid search if missing."""
    vectors_config = {
        "dense": models.VectorParams(
            size=dense_dim,
            distance=models.Distance.COSINE,
        )
    }
    sparse_config = {
        "sparse": models.SparseVectorParams()
    }

    exists = client.collection_exists(collection_name=collection_name)
    if recreate and exists:
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )


def to_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build Qdrant payload from extracted record and frontmatter metadata."""
    markdown = str(record.get("markdown") or "")
    payload: dict[str, Any] = {
        "dedupe_key": record.get("dedupe_key", ""),
        "markdown_body_chars": record.get("markdown_body_chars", 0),
        "markdown_sha256": record.get("markdown_sha256", ""),
        "markdown": markdown,
    }

    # Frontmatter format:
    # ---
    # key: value
    # ---
    if markdown.startswith("---\n"):
        parts = markdown.split("---\n", 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1]
            try:
                frontmatter = yaml.safe_load(frontmatter_text) or {}
                if isinstance(frontmatter, dict):
                    payload.update(frontmatter)
            except yaml.YAMLError:
                # Keep ingestion robust even if a single record has malformed frontmatter.
                pass

    published_dt = parse_datetime_value(
        payload.get("published_at") or record.get("published_at")
    )
    ingested_dt = parse_datetime_value(
        payload.get("ingested_at") or record.get("ingested_at")
    )

    if published_dt is not None:
        payload["published_at"] = published_dt
    if ingested_dt is not None:
        payload["ingested_at"] = ingested_dt

    published_ts = to_unix_ts(published_dt)
    ingested_ts = to_unix_ts(ingested_dt)
    if published_ts is not None:
        payload["published_ts"] = published_ts
    if ingested_ts is not None:
        payload["ingested_ts"] = ingested_ts

    return payload


def parse_datetime_value(value: Any) -> datetime | None:
    """Parse value into timezone-aware UTC datetime.

    Args:
        value: Datetime candidate from frontmatter or record.

    Returns:
        Parsed UTC datetime. Returns None when parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    text = str(value).strip()
    if not text:
        return None

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_unix_ts(value: datetime | None) -> int | None:
    """Convert datetime value to Unix timestamp seconds."""
    if value is None:
        return None
    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def split_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """Parse frontmatter and return (metadata, body)."""
    if not markdown.startswith("---\n"):
        return {}, markdown

    parts = markdown.split("---\n", 2)
    if len(parts) < 3:
        return {}, markdown

    frontmatter_text = parts[1]
    body = parts[2]
    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
        if isinstance(frontmatter, dict):
            return frontmatter, body
    except yaml.YAMLError:
        pass
    return {}, body


def split_by_h2(body_markdown: str) -> list[tuple[str, str]]:
    """Split markdown body by H2 sections.

    Returns:
        List of (section_title, section_markdown).
    """
    body = body_markdown.strip()
    if not body:
        return []

    # Allow leading spaces and optional space after ## because some sources emit " ##Heading".
    matches = list(re.finditer(r"(?m)^\s*##\s*(.+?)\s*$", body))
    if not matches:
        return [("", body)]

    sections: list[tuple[str, str]] = []
    # Preserve intro text that appears before the first H2.
    preface = body[:matches[0].start()].strip()
    if preface:
        sections.append(("", preface))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_title = match.group(1).strip()
        section_text = body[start:end].strip()
        if section_text:
            sections.append((section_title, section_text))
    return sections


def build_chunk_records(record: dict[str, Any], infer_article_metadata: bool = True) -> list[dict[str, Any]]:
    """Build H2-based chunk records from one extracted document."""
    markdown = str(record.get("markdown") or "")
    if not markdown.strip():
        return []

    _, body = split_frontmatter(markdown)

    vertical = "unknown"
    language = ""
    if infer_article_metadata:
        try:
            article_attribute = infer_attribute(input_text=body)
            vertical = str(article_attribute.vertical)
            language = str(article_attribute.language)
        except Exception as err:
            logger.warning("infer_attribute failed: %s", err)

    sections = split_by_h2(body)
    if not sections:
        return []

    base_payload = to_payload(record)
    chunks: list[dict[str, Any]] = []

    for idx, (section_title, section_markdown) in enumerate(sections):
        dedupe_key = str(record.get("dedupe_key") or "")
        # Use UUIDv5 from H2 title (and doc key) for stable deterministic point IDs.
        section_key = section_title or f"section-{idx:04d}"
        uuid_input = f"{dedupe_key}::{section_key}" if dedupe_key else section_key
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, uuid_input))
        text_for_embedding = (
            f"## {section_title}\n\n{section_markdown}"
            if section_title
            else section_markdown
        )

        payload = dict(base_payload)
        payload.update(
            {
                "chunk_index": idx,
                "chunk_id": chunk_id,
                "section_title": section_title,
                "chunk_markdown": section_markdown,
                "vertical": vertical,
                "language": language,
                "locale": str(base_payload.get("locale") or ""),
                "country": str(base_payload.get("country") or ""),
            }
        )
        chunks.append(
            {
                "point_id": chunk_id,
                "text": text_for_embedding,
                "payload": payload,
            }
        )

    return chunks


def embed_records(
    client: QdrantClient,
    model: BGEM3FlagModel,
    collection_name: str,
    chunks: list[dict[str, Any]],
    batch_size: int,
    embedding_model: str,
    embedding_dim: int,
) -> int:
    """Embed chunk records and upsert to Qdrant."""
    upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [str(r.get("text") or "") for r in batch]
        output = model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        points: list[models.PointStruct] = []
        for rec, dense, lexical in zip(batch, dense_vecs, lexical_weights):
            point_id = str(rec.get("point_id") or "")
            if not point_id:
                point_id = hashlib.sha256(str(rec).encode("utf-8")).hexdigest()

            sparse_vector = lexical_weights_to_sparse_vector(lexical or {})
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense.tolist() if hasattr(dense, "tolist") else list(dense),
                        "sparse": sparse_vector,
                    },
                    payload={
                        **dict(rec.get("payload") or {}),
                        "embedding_model": embedding_model,
                        "embedding_dim": embedding_dim,
                    },
                )
            )

        client.upsert(collection_name=collection_name,
                      points=points, wait=True)
        upserted += len(points)
        batch_chars = sum(len(t) for t in texts)
        logger.info(
            "upserted %s/%s chunks (batch_chars=%s, model=%s)",
            upserted,
            len(chunks),
            batch_chars,
            embedding_model,
        )

    return upserted


def get_existing_point_ids(
    client: QdrantClient,
    collection_name: str,
    point_ids: list[str],
) -> set[str]:
    """Return existing point IDs in the target collection."""
    if not point_ids:
        return set()
    if not client.collection_exists(collection_name=collection_name):
        return set()

    existing = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )
    return {str(point.id) for point in existing}


def update_payload_only(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    batch_size: int,
) -> int:
    """Update payload fields only for existing points without re-embedding."""
    updated = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        for rec in batch:
            point_id = str(rec.get("point_id") or "")
            payload = dict(rec.get("payload") or {})
            if not point_id or not payload:
                continue
            client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point_id],
                wait=True,
            )
            updated += 1
    return updated


def main() -> None:
    """Run embedding pipeline from LMDB to Qdrant."""
    args = parse_args()
    setup_logging(args.log_path)
    device = resolve_device(args.device)
    logger.info("log_path=%s rotation=daily backup_count=5", args.log_path)
    logger.info("embedding_device=%s", device)
    logger.info("embedding_model=%s", args.model_name)

    qdrant_url = normalize_qdrant_url(args.qdrant_url)
    logger.info("qdrant_url=%s", qdrant_url)
    client = QdrantClient(
        url=qdrant_url,
        api_key=args.qdrant_api_key or None,
    )

    if args.update_payload_only:
        if not client.collection_exists(collection_name=args.collection):
            logger.warning("collection does not exist: %s", args.collection)
            return
        if args.only_new:
            logger.warning("--only-new is ignored with --update-payload-only")

        processed_records = 0
        updated_records = 0
        updated_chunks = 0
        skipped_missing_chunks = 0
        for rec in iter_lmdb(args.input_lmdb):
            markdown = str(rec.get("markdown") or "").strip()
            if not markdown:
                continue
            if args.limit > 0 and processed_records >= args.limit:
                break
            processed_records += 1

            chunks_for_check = build_chunk_records(rec, infer_article_metadata=False)
            if not chunks_for_check:
                logger.warning("[%s] skip no h2 chunks", processed_records)
                continue

            point_ids = [str(ch.get("point_id") or "") for ch in chunks_for_check]
            existing_ids = get_existing_point_ids(client, args.collection, point_ids)
            chunks = chunks_for_check
            if existing_ids:
                chunks = [ch for ch in chunks if str(ch.get("point_id") or "") in existing_ids]
            skipped_missing_chunks += len(point_ids) - len(chunks)
            if not chunks:
                logger.info("[%s] skip no existing points title=%s",
                            processed_records, str(rec.get("title") or "")[:80])
                continue

            # Infer attributes only when at least one point exists in Qdrant.
            enriched_chunks = build_chunk_records(rec, infer_article_metadata=True)
            enriched_by_id = {str(ch.get("point_id") or ""): ch for ch in enriched_chunks}
            chunks = [
                enriched_by_id[str(ch.get("point_id") or "")]
                for ch in chunks
                if str(ch.get("point_id") or "") in enriched_by_id
            ]
            if not chunks:
                continue

            count = update_payload_only(
                client=client,
                collection_name=args.collection,
                chunks=chunks,
                batch_size=args.batch_size,
            )
            updated_records += 1
            updated_chunks += count
            logger.info(
                "[%s] payload_updated chunks=%s skipped_missing_chunks=%s title=%s",
                processed_records,
                count,
                skipped_missing_chunks,
                str(rec.get("title") or "")[:80],
            )
            if args.interval > 0:
                time.sleep(args.interval)

        logger.info(
            "done_payload_update: collection=%s, records=%s, chunks=%s, skipped_missing_chunks=%s, qdrant=%s",
            args.collection,
            updated_records,
            updated_chunks,
            skipped_missing_chunks,
            qdrant_url,
        )
        return

    try:
        model = BGEM3FlagModel(
            args.model_name,
            use_fp16=False,
            devices=[device],
        )
    except TypeError:
        # Backward compatibility for older FlagEmbedding versions.
        model = BGEM3FlagModel(args.model_name, use_fp16=False)
    dense_dim: int | None = None
    collection_ready = False
    processed_records = 0
    embedded_records = 0
    embedded_chunks = 0
    skipped_existing_records = 0
    skipped_existing_chunks = 0

    for rec in iter_lmdb(args.input_lmdb):
        markdown = str(rec.get("markdown") or "").strip()
        if not markdown:
            continue
        if args.limit > 0 and processed_records >= args.limit:
            break
        processed_records += 1

        chunks_for_check = build_chunk_records(rec, infer_article_metadata=False)
        if not chunks_for_check:
            logger.warning("[%s] skip no h2 chunks", processed_records)
            continue

        if args.only_new:
            point_ids = [str(ch.get("point_id") or "") for ch in chunks_for_check]
            existing_ids = get_existing_point_ids(
                client, args.collection, point_ids)
            chunks = chunks_for_check
            before = len(chunks)
            if existing_ids:
                chunks = [ch for ch in chunks if str(
                    ch.get("point_id") or "") not in existing_ids]
                skipped_existing_chunks += before - len(chunks)
            if not chunks:
                skipped_existing_records += 1
                logger.info("[%s] skip existing chunks title=%s",
                            processed_records, str(rec.get("title") or "")[:80])
                continue
            # Infer attributes only when there are points to embed.
            enriched_chunks = build_chunk_records(rec, infer_article_metadata=True)
            enriched_by_id = {str(ch.get("point_id") or ""): ch for ch in enriched_chunks}
            chunks = [
                enriched_by_id[str(ch.get("point_id") or "")]
                for ch in chunks
                if str(ch.get("point_id") or "") in enriched_by_id
            ]
            if not chunks:
                continue
        else:
            chunks = build_chunk_records(rec, infer_article_metadata=True)
            if not chunks:
                logger.warning("[%s] skip no h2 chunks", processed_records)
                continue

        if not collection_ready:
            probe = model.encode(
                [str(chunks[0].get("text") or "")],
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense_dim = len(probe["dense_vecs"][0])
            ensure_collection(
                client=client,
                collection_name=args.collection,
                dense_dim=dense_dim,
                recreate=args.recreate,
            )
            collection_ready = True

        count = embed_records(
            client=client,
            model=model,
            collection_name=args.collection,
            chunks=chunks,
            batch_size=args.batch_size,
            embedding_model=args.model_name,
            embedding_dim=dense_dim or 0,
        )
        embedded_records += 1
        embedded_chunks += count
        record_chars = int(rec.get("markdown_body_chars") or len(markdown))
        chunk_chars = sum(len(str(ch.get("text") or "")) for ch in chunks)
        logger.info(
            "[%s] upserted_record chunks=%s chars=%s chunk_chars=%s model=%s title=%s",
            processed_records,
            count,
            record_chars,
            chunk_chars,
            args.model_name,
            str(rec.get("title") or "")[:80],
        )
        if args.interval > 0:
            time.sleep(args.interval)

    if not collection_ready:
        logger.warning(
            "no records to embed (skipped_existing_records=%s, skipped_existing_chunks=%s, model=%s)",
            skipped_existing_records,
            skipped_existing_chunks,
            args.model_name,
        )
        return

    logger.info(
        "done: collection=%s, records=%s, chunks=%s, skipped_existing_records=%s, skipped_existing_chunks=%s, qdrant=%s, model=%s",
        args.collection,
        embedded_records,
        embedded_chunks,
        skipped_existing_records,
        skipped_existing_chunks,
        qdrant_url,
        args.model_name,
    )


if __name__ == "__main__":
    main()
