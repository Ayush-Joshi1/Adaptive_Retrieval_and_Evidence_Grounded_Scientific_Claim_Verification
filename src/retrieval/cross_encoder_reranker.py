from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    _MODEL_CACHE: dict[tuple[str, str | None], Any] = {}
    """Reusable cross-encoder reranker for scientific claim retrieval.

    The reranker keeps the original retrieval metadata intact while adding a
    `cross_encoder_score` and remapped `rank` after reranking. It supports a
    configurable candidate pool and batch inference.
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, batch_size: int = 32, device: str | None = None, max_candidates: Optional[int] = None, strict: bool = False):
        self.model_name = model_name or DEFAULT_CROSS_ENCODER_MODEL
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.max_candidates = None if max_candidates is None else max(1, int(max_candidates))
        self.strict = strict
        self.last_inference_seconds = 0.0
        self._model = self._load_model()

    def _load_model(self):
        cache_key = (self.model_name, self.device)
        if cache_key in CrossEncoderReranker._MODEL_CACHE:
            model = CrossEncoderReranker._MODEL_CACHE[cache_key]
            if self.strict and model is None:
                raise RuntimeError(f"Cross-encoder model could not be loaded: {self.model_name}")
            return model
        if CrossEncoder is None:
            model = None
        else:
            try:
                model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"Failed to load cross-encoder model: {self.model_name}") from exc
                model = None
        CrossEncoderReranker._MODEL_CACHE[cache_key] = model
        if self.strict and model is None:
            raise RuntimeError(f"Cross-encoder model could not be loaded: {self.model_name}")
        return model

    def _coerce_text(self, candidate: Mapping[str, Any] | str) -> str:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, Mapping):
            pieces = []
            if candidate.get("title"):
                pieces.append(str(candidate["title"]))
            if candidate.get("text"):
                pieces.append(str(candidate["text"]))
            if candidate.get("abstract"):
                pieces.append(str(candidate["abstract"]))
            if not pieces and candidate.get("content"):
                pieces.append(str(candidate["content"]))
            if pieces:
                return " ".join(pieces)
        return ""

    def _candidate_documents(self, candidates: Sequence[Mapping[str, Any] | str]) -> List[Mapping[str, Any] | str]:
        if candidates is None:
            return []
        docs = list(candidates)
        if self.max_candidates is not None and self.max_candidates > 0:
            docs = docs[: self.max_candidates]
        return docs

    def _fallback_score(self, claim: str, text: str) -> float:
        claim_tokens = {token.lower() for token in re.findall(r"[a-z0-9]+", claim or "")}
        text_tokens = re.findall(r"[a-z0-9]+", text or "")
        text_token_counts = {}
        for token in text_tokens:
            text_token_counts[token.lower()] = text_token_counts.get(token.lower(), 0) + 1
        overlap = sum(text_token_counts.get(token, 0) for token in claim_tokens)
        if not claim_tokens or not text_tokens:
            return 0.0
        similarity = overlap / max(1, len(claim_tokens) * 2)
        length_bonus = min(len(text_tokens) / 200.0, 1.0)
        return float(similarity + 0.1 * length_bonus)

    def score_pairs(self, query: str, texts: Sequence[str]) -> List[float]:
        if query is None:
            query = ""
        texts = [str(text or "") for text in texts]
        if not texts:
            return []
        if self._model is None:
            if self.strict:
                raise RuntimeError(f"Cross-encoder model is unavailable: {self.model_name}")
            self.last_inference_seconds = 0.0
            return [self._fallback_score(query, text) for text in texts]
        pairs = [(str(query), text) for text in texts]
        inference_start = time.perf_counter()
        try:
            try:
                outputs = self._model.predict(pairs, batch_size=self.batch_size)
            except TypeError:
                outputs = self._model.predict(pairs)
        except Exception as exc:
            if self.strict:
                raise RuntimeError("Cross-encoder inference failed") from exc
            outputs = [self._fallback_score(query, text) for text in texts]
        finally:
            self.last_inference_seconds = time.perf_counter() - inference_start
        scores = []
        for value in outputs:
            try:
                scores.append(float(value))
            except (TypeError, ValueError) as exc:
                if self.strict:
                    raise ValueError("Cross-encoder returned a non-scalar score") from exc
                scores.append(self._fallback_score(query, ""))
        if self.strict and len(scores) != len(texts):
            raise ValueError(f"Cross-encoder returned {len(scores)} scores for {len(texts)} candidates")
        return scores

    def rerank(self, query: str, candidates: Sequence[Mapping[str, Any] | str], top_k: Optional[int] = None, query_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if query is None or str(query).strip() == "":
            return []
        docs = self._candidate_documents(candidates)
        if not docs:
            return []

        selected: List[Dict[str, Any]] = []
        for index, candidate in enumerate(docs):
            if isinstance(candidate, Mapping):
                doc = dict(candidate)
            else:
                doc = {"document_id": str(candidate), "corpus_id": str(candidate), "text": str(candidate), "rank": index + 1}
            doc.setdefault("document_id", doc.get("corpus_id") or f"doc-{index}")
            doc.setdefault("corpus_id", doc.get("document_id") or f"doc-{index}")
            doc.setdefault("rank", index + 1)
            doc.setdefault("retrieval_score", doc.get("score", 0.0))
            doc.setdefault("score", doc.get("retrieval_score", 0.0))
            selected.append(doc)

        texts = [self._coerce_text(doc) for doc in selected]
        scores = self.score_pairs(query, texts)
        scored = []
        for index, (doc, score) in enumerate(zip(selected, scores)):
            original_rank = int(doc.get("rank", index + 1))
            retrieval_score = float(doc.get("retrieval_score") or doc.get("score") or 0.0)
            ranked = dict(doc)
            ranked["query_id"] = str(query_id) if query_id is not None else ""
            ranked["document_id"] = str(doc.get("document_id") or doc.get("corpus_id") or f"doc-{index}")
            ranked["corpus_id"] = str(doc.get("corpus_id") or doc.get("document_id") or ranked["document_id"])
            ranked["original_rank"] = original_rank
            ranked["retrieval_score"] = retrieval_score
            ranked["cross_encoder_score"] = float(score)
            ranked["rerank_score"] = float(score)
            scored.append(ranked)

        scored.sort(key=lambda item: (-float(item["cross_encoder_score"]), int(item.get("original_rank", 10**9))))
        for new_rank, item in enumerate(scored, start=1):
            item["rank"] = new_rank
            item["retriever_name"] = item.get("retriever_name", "cross_encoder")

        limit = len(scored) if top_k is None else max(0, min(len(scored), int(top_k)))
        return scored[:limit]

    def rerank_batch(self, query_doc_pairs: Iterable[tuple[str, Sequence[Mapping[str, Any] | str]]], top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        results: List[List[Dict[str, Any]]] = []
        for query, candidates in query_doc_pairs:
            results.append(self.rerank(query, candidates, top_k=top_k))
        return results
