import streamlit as st
from html import escape
import logging
import time
from qdrant_client import models

logger = logging.getLogger(__name__)
EMERGING_DECAY_WINDOW_SECONDS = 180 * 24 * 60 * 60
EMERGING_RECENCY_BOOST_WEIGHT = 0.1


def _scores(points: list[object]) -> list[float]:
    """Extract score list from points."""
    vals: list[float] = []
    for point in points:
        score = getattr(point, "score", None)
        if score is None:
            continue
        try:
            vals.append(float(score))
        except (TypeError, ValueError):
            continue
    return vals


def _published_ts_stats(points: list[object]) -> tuple[int, int | None, int | None]:
    """Return (missing_count, min_ts, max_ts) for payload.published_ts."""
    ts_values: list[int] = []
    missing = 0
    for point in points:
        payload = getattr(point, "payload", None) or {}
        ts = payload.get("published_ts")
        if isinstance(ts, (int, float)):
            ts_values.append(int(ts))
        else:
            missing += 1
    if not ts_values:
        return missing, None, None
    return missing, min(ts_values), max(ts_values)


def retrieve_vector_results_by_queries(
    *,
    canonical_query: str,
    emerging_query: str,
    vector_candidate_k: int,
    mmr_diversity: float = 0.3,
    per_query_top_k: int = 5,
) -> dict[str, list[object]]:
    """Retrieve points per query with hybrid search + per-query MMR."""
    if st.session_state.vector_searcher is None:
        return {"canonical": [], "emerging": []}
    query_by_kind = {
        "canonical": canonical_query.strip(),
        "emerging": emerging_query.strip(),
    }
    result: dict[str, list[object]] = {"canonical": [], "emerging": []}
    now_ts = int(time.time())
    threshold_ts = int(now_ts - EMERGING_DECAY_WINDOW_SECONDS)

    for kind, query_text in query_by_kind.items():
        if not query_text:
            continue
        time_filter = build_time_filter(kind=kind, threshold_ts=threshold_ts)
        points = hybrid_search_with_filter(
            query_text=query_text,
            vector_candidate_k=vector_candidate_k,
            query_filter=time_filter,
        )
        logger.info(
            "retrieval kind=%s stage=hybrid count=%s scores=%s",
            kind,
            len(points),
            [round(s, 6) for s in _scores(points)],
        )
        missing_ts, min_ts, max_ts = _published_ts_stats(points)
        logger.info(
            "retrieval kind=%s stage=hybrid published_ts missing=%s min=%s max=%s threshold=%s",
            kind,
            missing_ts,
            min_ts,
            max_ts,
            threshold_ts,
        )
        deduped: dict[str, object] = {}
        for point in points:
            raw_point_id = getattr(point, "id", None)
            if raw_point_id is None:
                continue
            dedupe_key = str(raw_point_id)
            if dedupe_key and dedupe_key not in deduped:
                deduped[dedupe_key] = raw_point_id
        candidate_ids = list(deduped.values())
        logger.info(
            "retrieval kind=%s stage=dedupe count=%s id_types=%s",
            kind,
            len(candidate_ids),
            sorted({type(pid).__name__ for pid in candidate_ids}) if candidate_ids else [],
        )
        if not candidate_ids:
            continue
        mmr_points = qdrant_mmr_rerank_for_query(
            candidate_ids=candidate_ids,
            query_text=query_text,
            diversity=mmr_diversity,
            top_k=per_query_top_k,
            base_filter=time_filter,
            recency_boost_weight=(EMERGING_RECENCY_BOOST_WEIGHT if kind == "emerging" else 0.0),
            recency_target_ts=(now_ts if kind == "emerging" else None),
            recency_scale_seconds=(EMERGING_DECAY_WINDOW_SECONDS if kind == "emerging" else None),
        )
        logger.info(
            "retrieval kind=%s stage=mmr count=%s scores=%s",
            kind,
            len(mmr_points),
            [round(s, 6) for s in _scores(mmr_points)],
        )
        result[kind] = mmr_points
    return result


def build_time_filter(kind: str, threshold_ts: int) -> models.Filter:
    """Build published_ts time filter by query kind."""
    if kind == "emerging":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="published_ts",
                    range=models.Range(gte=threshold_ts),
                )
            ]
        )
    return models.Filter(
        must=[
            models.FieldCondition(
                key="published_ts",
                range=models.Range(lt=threshold_ts),
            )
        ]
    )


def hybrid_search_with_filter(
    *,
    query_text: str,
    vector_candidate_k: int,
    query_filter: models.Filter,
) -> list[object]:
    """Run hybrid search with a Qdrant filter."""
    if st.session_state.vector_searcher is None:
        return []
    searcher = st.session_state.vector_searcher
    return searcher.hybrid_search_with_filter(
        query_text=query_text,
        candidate_k=vector_candidate_k,
        query_filter=query_filter,
    )


def qdrant_mmr_rerank_for_query(
    *,
    candidate_ids: list[object],
    query_text: str,
    diversity: float,
    top_k: int,
    base_filter: models.Filter,
    recency_boost_weight: float = 0.0,
    recency_target_ts: int | None = None,
    recency_scale_seconds: int | None = None,
) -> list[object]:
    """Rerank candidate IDs with Qdrant MMR for a single query."""
    if st.session_state.vector_searcher is None or not candidate_ids:
        return []
    searcher = st.session_state.vector_searcher
    return searcher.mmr_rerank_by_ids(
        query_text=query_text,
        candidate_ids=candidate_ids,
        diversity=diversity,
        top_k=top_k,
        base_filter=base_filter,
        recency_boost_weight=recency_boost_weight,
        recency_target_ts=recency_target_ts,
        recency_scale_seconds=recency_scale_seconds,
    )


def points_to_prompt_context(
    points: list[object],
    label: str | None = None
) -> str:
    """Build prompt context text from vector search points."""
    retrieved_items: list[str] = []
    for idx, point in enumerate(points, 1):
        payload = point.payload or {}
        title = str(payload.get("title") or "")
        section_title = str(payload.get("section_title") or "")
        published_at = str(payload.get("published_at") or "")
        vertical = str(payload.get("vertical") or "")
        chunk_markdown = str(payload.get("chunk_markdown") or "")
        retrieved_items.append(
            f"{label}[{idx}]\n- title={title}\n- section={section_title}\n- published_at={published_at}\n- vertical={vertical}\n- body={chunk_markdown}"
        )
    return "\n\n".join(retrieved_items)


def points_to_table_rows(points: list[object]) -> list[dict[str, str]]:
    """Convert vector search points into table rows."""
    rows: list[dict[str, str]] = []
    for point in points:
        payload = point.payload or {}
        rows.append(
            {
                "title": str(payload.get("title") or ""),
                "source_url": str(payload.get("source_url") or ""),
                "published_at": str(payload.get("published_at") or ""),
                "chunk_id": str(payload.get("chunk_id") or payload.get("point_id") or point.id or ""),
                "section_title": str(payload.get("section_title") or ""),
            }
        )
    return rows


def build_retrieved_results_html_table(rows: list[dict[str, str]]) -> str:
    """Build HTML table with linked title and retrieval metadata columns."""
    if not rows:
        return ""

    body_rows: list[str] = []
    for row in rows:
        title = escape(row.get("title", ""))
        published_at = escape(row.get("published_at", ""))
        section_title = escape(row.get("section_title", ""))
        source_url = row.get("source_url", "").strip()
        title_cell = (
            f'<a href="{escape(source_url, quote=True)}" target="_blank">{title}</a>'
            if source_url
            else title
        )
        body_rows.append(
            f"<tr><td>{title_cell}</td><td>{published_at}</td><td>{section_title}</td></tr>"
        )

    body_html = "".join(body_rows)
    return (
        "<table>"
        "<thead><tr><th>title</th><th>published_at</th><th>section_title</th></tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )
