from __future__ import annotations

from typing import Any, Dict, List, Optional


class HybridRetriever:
    """Combine sparse and dense rankings with reciprocal rank fusion."""

    def __init__(
        self,
        bm25_retriever: Any | None = None,
        dense_retriever: Any | None = None,
        rrf_constant: float = 60,
        bm25_depth: int = 100,
        dense_depth: int = 100,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_constant = float(rrf_constant)
        self.bm25_depth = int(bm25_depth)
        self.dense_depth = int(dense_depth)

    def _get_document_id(self, result: Dict[str, Any]) -> str:
        return str(result.get("document_id") or result.get("corpus_id") or "")

    def retrieve(self, query: str, top_k: Optional[int] = None, query_id: Optional[str] = None, bm25_depth: Optional[int] = None, dense_depth: Optional[int] = None, rrf_constant: Optional[float] = None) -> List[Dict[str, Any]]:
        if self.bm25_retriever is None or self.dense_retriever is None:
            raise ValueError("Hybrid retrieval requires both BM25 and dense retrievers.")
        if top_k is None:
            top_k = self.bm25_depth
        if bm25_depth is None:
            bm25_depth = self.bm25_depth
        if dense_depth is None:
            dense_depth = self.dense_depth
        if rrf_constant is None:
            rrf_constant = self.rrf_constant

        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_depth, query_id=query_id)
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_depth, query_id=query_id)

        bm25_ranks: Dict[str, int] = {}
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = self._get_document_id(result)
            if doc_id:
                bm25_ranks[doc_id] = rank

        dense_ranks: Dict[str, int] = {}
        for rank, result in enumerate(dense_results, start=1):
            doc_id = self._get_document_id(result)
            if doc_id:
                dense_ranks[doc_id] = rank

        fused_scores: Dict[str, float] = {}
        for doc_id, rank in bm25_ranks.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_constant + rank)
        for doc_id, rank in dense_ranks.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_constant + rank)

        ordered = sorted(
            fused_scores.items(),
            key=lambda item: (
                -float(item[1]),
                bm25_ranks.get(item[0], 10**9),
                dense_ranks.get(item[0], 10**9),
                item[0],
            ),
        )

        results: List[Dict[str, Any]] = []
        for final_rank, (doc_id, score) in enumerate(ordered[:max(1, int(top_k))], start=1):
            result = {
                "query_id": str(query_id) if query_id is not None else "",
                "document_id": doc_id,
                "corpus_id": doc_id,
                "rank": final_rank,
                "score": float(score),
                "retrieval_score": float(score),
                "rrf_score": float(score),
                "bm25_rank": bm25_ranks.get(doc_id),
                "dense_rank": dense_ranks.get(doc_id),
                "retriever_name": "hybrid",
                "similarity_score": float(score),
            }
            results.append(result)
        return results
