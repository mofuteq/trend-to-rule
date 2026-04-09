import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.text_utils import normalize_text_nfkc


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data"

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "article_markdown_bge_m3"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
VERTICAL_BOOSTABLE_VALUES = {"mens", "womens", "unisex"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid (dense+sparse) vector search with optional MMR reranking."
    )
    parser.add_argument("--query", type=str, required=True, help="Search query text")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Qdrant collection")
    parser.add_argument("--top-k", type=int, default=10, help="Final result count")
    parser.add_argument("--candidate-k", type=int, default=40, help="Candidate pool size before MMR")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Embedding model name")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Embedding device. 'auto' prefers mps on Mac, then cuda, then cpu.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=DATA_DIR / "qdrant_data",
        help="Local persistent Qdrant path (prioritized over --qdrant-url)",
    )
    parser.add_argument("--qdrant-url", type=str, default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--qdrant-api-key", type=str, default="", help="Qdrant API key")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR reranking")
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR relevance-diversity tradeoff (0..1, higher = relevance)",
    )
    return parser.parse_args()


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
    """Map lexical token key to stable positive integer index."""
    digest = __import__("hashlib").sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def lexical_weights_to_sparse_vector(weights: dict[Any, float]) -> models.SparseVector:
    """Convert bge-m3 lexical_weights to Qdrant sparse vector."""
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
    return models.SparseVector(
        indices=[i for i, _ in pairs],
        values=[v for _, v in pairs],
    )


def normalize_query_text(text: str) -> str:
    """Normalize query text with Unicode NFKC."""
    return normalize_text_nfkc(text).strip()


def _extract_scores(points: list[Any]) -> list[float]:
    """Extract scores from Qdrant points."""
    scores: list[float] = []
    for p in points:
        score = getattr(p, "score", None)
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    return scores


def _build_score_formula(
    *,
    vertical_match_value: str | None = None,
    vertical_boost_weight: float = 0.0,
    recency_boost_weight: float = 0.0,
    recency_target_ts: int | None = None,
    recency_scale_seconds: int | None = None,
) -> models.FormulaQuery | None:
    """Build a Qdrant formula query for payload-aware score boosts."""
    terms: list[Any] = ["$score"]

    normalized_vertical = (vertical_match_value or "").strip().lower()
    if (
        vertical_boost_weight > 0.0
        and normalized_vertical in VERTICAL_BOOSTABLE_VALUES
    ):
        terms.append(
            models.MultExpression(
                mult=[
                    float(vertical_boost_weight),
                    models.FieldCondition(
                        key="vertical",
                        match=models.MatchValue(value=normalized_vertical),
                    ),
                ]
            )
        )

    if (
        recency_boost_weight > 0.0
        and recency_target_ts is not None
        and recency_scale_seconds is not None
        and recency_scale_seconds > 0
    ):
        terms.append(
            models.MultExpression(
                mult=[
                    float(recency_boost_weight),
                    models.LinDecayExpression(
                        lin_decay=models.DecayParamsExpression(
                            x="published_ts",
                            target=float(recency_target_ts),
                            scale=float(recency_scale_seconds),
                            midpoint=0.5,
                        )
                    ),
                ]
            )
        )

    if len(terms) == 1:
        return None
    return models.FormulaQuery(formula=models.SumExpression(sum=terms))


class HybridVectorSearcher:
    """Hybrid vector searcher using bge-m3 and Qdrant."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        collection: str = DEFAULT_COLLECTION,
        qdrant_path: Path | None = DATA_DIR / "qdrant_data",
        qdrant_url: str = DEFAULT_QDRANT_URL,
        qdrant_api_key: str = "",
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.collection = collection
        self.qdrant_path = qdrant_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self._model: BGEM3FlagModel | None = None
        self._client: QdrantClient | None = None

    def _get_model(self) -> BGEM3FlagModel:
        """Lazy-initialize embedding model."""
        if self._model is None:
            logger.info("device=%s model=%s", self.device, self.model_name)
            try:
                self._model = BGEM3FlagModel(self.model_name, use_fp16=False, devices=[self.device])
            except TypeError:
                self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
        return self._model

    def _get_client(self) -> QdrantClient:
        """Lazy-initialize qdrant client."""
        if self._client is None:
            if self.qdrant_path:
                self.qdrant_path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(self.qdrant_path))
                logger.info("qdrant_mode=local path=%s", self.qdrant_path)
            else:
                qurl = normalize_qdrant_url(self.qdrant_url)
                self._client = QdrantClient(url=qurl, api_key=self.qdrant_api_key or None)
                logger.info("qdrant_mode=remote url=%s", qurl)
        return self._client

    def _encode_query(self, query_text: str) -> tuple[list[float], models.SparseVector]:
        """Encode query into dense and sparse vectors."""
        normalized_query = normalize_query_text(query_text)
        model = self._get_model()
        output = model.encode(
            [normalized_query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        query_dense = output["dense_vecs"][0]
        dense = query_dense.tolist() if hasattr(query_dense, "tolist") else list(query_dense)
        sparse = lexical_weights_to_sparse_vector(output["lexical_weights"][0] or {})
        return dense, sparse

    def hybrid_search_with_filter(
        self,
        *,
        query_text: str,
        candidate_k: int,
        query_filter: models.Filter | None = None,
        vertical_match_value: str | None = None,
        vertical_boost_weight: float = 0.0,
        recency_boost_weight: float = 0.0,
        recency_target_ts: int | None = None,
        recency_scale_seconds: int | None = None,
    ) -> list[Any]:
        """Run hybrid RRF search with optional filter."""
        client = self._get_client()
        query_dense, query_sparse = self._encode_query(query_text)
        prefetch = [
            models.Prefetch(
                query=query_dense,
                using="dense",
                limit=candidate_k,
                filter=query_filter,
            ),
            models.Prefetch(
                query=query_sparse,
                using="sparse",
                limit=candidate_k,
                filter=query_filter,
            ),
        ]
        formula_query = _build_score_formula(
            vertical_match_value=vertical_match_value,
            vertical_boost_weight=vertical_boost_weight,
            recency_boost_weight=recency_boost_weight,
            recency_target_ts=recency_target_ts,
            recency_scale_seconds=recency_scale_seconds,
        )
        resp = client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=formula_query or models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=candidate_k,
            with_payload=True,
        )
        points = list(resp.points)
        scores = _extract_scores(points)
        logger.info(
            "hybrid=rrf filter=%s vertical_boost=%s recency_boost=%s candidate_k=%s results_count=%s scores=%s",
            "on" if query_filter else "off",
            "on" if vertical_boost_weight > 0.0 and vertical_match_value else "off",
            "on" if formula_query and recency_boost_weight > 0.0 else "off",
            candidate_k,
            len(points),
            [round(s, 6) for s in scores],
        )
        return points

    def mmr_rerank_by_ids(
        self,
        *,
        query_text: str,
        candidate_ids: list[object],
        diversity: float,
        top_k: int,
        base_filter: models.Filter | None = None,
        vertical_match_value: str | None = None,
        vertical_boost_weight: float = 0.0,
        recency_boost_weight: float = 0.0,
        recency_target_ts: int | None = None,
        recency_scale_seconds: int | None = None,
    ) -> list[Any]:
        """Run Qdrant MMR rerank constrained by candidate IDs."""
        if not candidate_ids:
            return []
        client = self._get_client()
        query_dense, _ = self._encode_query(query_text)
        diversity = max(0.0, min(1.0, diversity))
        candidate_limit = max(len(candidate_ids), top_k)
        mmr = models.Mmr(diversity=diversity, candidates_limit=candidate_limit)
        base_must = list((base_filter.must if base_filter else []) or [])
        id_filter = models.Filter(
            must=[
                *base_must,
                models.HasIdCondition(has_id=candidate_ids),
            ]
        )

        formula_query = _build_score_formula(
            vertical_match_value=vertical_match_value,
            vertical_boost_weight=vertical_boost_weight,
            recency_boost_weight=recency_boost_weight,
            recency_target_ts=recency_target_ts,
            recency_scale_seconds=recency_scale_seconds,
        )
        use_formula_boost = formula_query is not None
        if use_formula_boost:
            resp = client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=models.NearestQuery(nearest=query_dense, mmr=mmr),
                        using="dense",
                        limit=candidate_limit,
                        filter=id_filter,
                    )
                ],
                query=formula_query,
                limit=top_k,
                with_payload=True,
            )
        else:
            resp = client.query_points(
                collection_name=self.collection,
                query=models.NearestQuery(nearest=query_dense, mmr=mmr),
                using="dense",
                query_filter=id_filter,
                limit=top_k,
                with_payload=True,
            )
        points = list(resp.points)
        scores = _extract_scores(points)
        logger.info(
            "mmr=on diversity=%s top_k=%s candidates=%s vertical_boost=%s recency_boost=%s results_count=%s scores=%s",
            diversity,
            top_k,
            len(candidate_ids),
            "on" if vertical_boost_weight > 0.0 and vertical_match_value else "off",
            "on" if recency_boost_weight > 0.0 else "off",
            len(points),
            [round(s, 6) for s in scores],
        )
        return points

    def mmr_search_with_filter(
        self,
        *,
        query_text: str,
        candidate_k: int,
        diversity: float,
        top_k: int,
        query_filter: models.Filter | None = None,
    ) -> list[Any]:
        """Run Qdrant MMR search over filtered collection space."""
        client = self._get_client()
        query_dense, query_sparse = self._encode_query(query_text)
        diversity = max(0.0, min(1.0, diversity))
        prefetch = [
            models.Prefetch(query=query_dense, using="dense", limit=candidate_k),
            models.Prefetch(query=query_sparse, using="sparse", limit=candidate_k),
        ]
        mmr = models.Mmr(diversity=diversity, candidates_limit=candidate_k)
        resp = client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.NearestQuery(nearest=query_dense, mmr=mmr),
            using="dense",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        points = list(resp.points)
        logger.info(
            "hybrid_prefetch + qdrant_mmr filter=%s top_k=%s candidate_k=%s diversity=%s results_count=%s scores=%s",
            "on" if query_filter else "off",
            top_k,
            candidate_k,
            diversity,
            len(points),
            [round(s, 6) for s in _extract_scores(points)],
        )
        return points

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        candidate_k: int = 40,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
    ) -> list[Any]:
        """Search vectors with hybrid retrieval and optional Qdrant MMR."""
        if not use_mmr:
            logger.info("hybrid=rrf mmr=off top_k=%s", top_k)
            return self.hybrid_search_with_filter(
                query_text=query_text,
                candidate_k=max(top_k, candidate_k),
                query_filter=None,
            )[:top_k]

        points = self.mmr_search_with_filter(
            query_text=query_text,
            candidate_k=max(top_k, candidate_k),
            diversity=max(0.0, min(1.0, 1.0 - mmr_lambda)),
            top_k=top_k,
            query_filter=None,
        )
        return points


def main() -> None:
    """Run hybrid search and optional MMR rerank."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    searcher = HybridVectorSearcher(
        model_name=args.model_name,
        device=args.device,
        collection=args.collection,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
    )
    final_points = searcher.search(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        use_mmr=not args.no_mmr,
        mmr_lambda=args.mmr_lambda,
    )
    logger.info("results_count=%s", len(final_points))

    for i, p in enumerate(final_points, 1):
        payload = p.payload or {}
        title = str(payload.get("title") or "")
        section_title = str(payload.get("section_title") or "")
        chunk_id = str(payload.get("chunk_id") or p.id)
        chunk_markdown = str(payload.get("chunk_markdown") or "")
        score = float(getattr(p, "score", 0.0) or 0.0)
        snippet = chunk_markdown.replace("\n", " ")[:180]
        logger.info(
            "[%s] score=%.6f id=%s title=%s section=%s snippet=%s",
            i,
            score,
            chunk_id,
            title[:100],
            section_title[:100],
            snippet,
        )


if __name__ == "__main__":
    main()
