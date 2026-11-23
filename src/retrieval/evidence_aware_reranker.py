from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _tokenize(text: Any) -> List[str]:
    if text is None:
        return []
    return re.findall(r"[a-z0-9]+", str(text).lower())


class EvidenceAwareReranker:
    """Transparent weighted reranker combining relevance signals.

    Features are normalized to the [0, 1] interval before weighting. The default
    weights preserve the existing project configuration; ablations set every
    omitted signal to zero and renormalize the remaining weights.
    """

    def __init__(self, weights: Optional[Mapping[str, float]] = None, candidate_pool_size: Optional[int] = None):
        default_weights = {
            "cross_encoder": 0.55,
            "lexical_overlap": 0.20,
            "scientific_overlap": 0.15,
            "retrieval_rank": 0.10,
            "retrieval_score": 0.0,
            "evidence_support": 0.0,
            "evidence_refutation": 0.0,
        }
        if weights is None:
            weights = default_weights
            self.weights = {key: float(value) for key, value in default_weights.items()}
        else:
            self.weights = {key: 0.0 for key in default_weights}
            self.weights.update({str(key): float(value) for key, value in weights.items()})
        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("Evidence-aware reranking weights must be non-negative")
        if abs(sum(value for value in self.weights.values() if value > 0.0) - 1.0) > 1e-9:
            # Single-signal and partial ablations are valid; normalize enabled weights at scoring time.
            pass
        self.candidate_pool_size = None if candidate_pool_size is None else max(1, int(candidate_pool_size))

    def _normalize(self, value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 0.0
        return float((value - minimum) / (maximum - minimum))

    def _lexical_overlap(self, query: str, document: str) -> float:
        q_tokens = set(_tokenize(query))
        d_tokens = _tokenize(document)
        if not q_tokens or not d_tokens:
            return 0.0
        overlap = len(q_tokens & set(d_tokens))
        return overlap / max(1, len(q_tokens))

    def _scientific_overlap(self, query: str, document: str) -> float:
        scientific_terms = {
            "cell", "protein", "gene", "mutation", "disease", "clinical", "therapy",
            "treatment", "patient", "model", "virus", "drug", "immune", "rna", "dna",
            "pathway", "biomarker", "genome", "expression", "factor", "signal", "study",
            "outcome", "trial", "evidence", "evidence-based"
        }
        q_terms = set(_tokenize(query)) & scientific_terms
        d_terms = set(_tokenize(document)) & scientific_terms
        union = q_terms | d_terms
        if not union:
            return 0.0
        return len(q_terms & d_terms) / len(union)

    def _normalized_weights(self) -> Dict[str, float]:
        enabled = {
            key: max(0.0, float(value))
            for key, value in self.weights.items()
            if float(value) > 0.0
        }
        total = sum(enabled.values())
        if total <= 0.0:
            return {key: 0.0 for key in self.weights}
        return {key: value / total for key, value in enabled.items()}

    def _score_candidate(self, query: str, candidate: Mapping[str, Any], cross_encoder_score: Optional[float] = None) -> Dict[str, float]:
        text = str(candidate.get("text") or candidate.get("title") or candidate.get("abstract") or "")
        retrieval_score = float(candidate.get("retrieval_score") or candidate.get("score") or 0.0)
        rank = int(candidate.get("rank") or candidate.get("original_rank") or 1)
        lexical_overlap = self._lexical_overlap(query, text)
        scientific_overlap = self._scientific_overlap(query, text)
        cross_value = float(cross_encoder_score if cross_encoder_score is not None else candidate.get("cross_encoder_score") or 0.0)
        normalized_cross = max(0.0, min(1.0, cross_value))
        normalized_retrieval = max(0.0, min(1.0, retrieval_score))
        rank_signal = 1.0 / max(rank, 1)
        weights = self._normalized_weights()
        final_score = sum(
            weights.get(key, 0.0) * value
            for key, value in (
                ("cross_encoder", normalized_cross),
                ("lexical_overlap", lexical_overlap),
                ("scientific_overlap", scientific_overlap),
                ("retrieval_rank", rank_signal),
                ("retrieval_score", normalized_retrieval),
            )
        )
        return {
            "cross_encoder_score": cross_value,
            "lexical_overlap": lexical_overlap,
            "scientific_overlap": scientific_overlap,
            "retrieval_score": retrieval_score,
            "retrieval_rank": rank,
            "rank_signal": rank_signal,
            "final_score": final_score,
        }

    def rerank(self, query: str, candidates: Sequence[Mapping[str, Any]], top_k: Optional[int] = None, cross_encoder_scores: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pool: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, candidate in enumerate(candidates):
            doc = dict(candidate)
            doc_id = str(doc.get("document_id") or doc.get("corpus_id") or "")
            if not doc_id:
                doc_id = f"doc-{index}"
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            doc["document_id"] = doc_id
            doc["corpus_id"] = str(doc.get("corpus_id") or doc_id)
            pool.append(doc)
        if self.candidate_pool_size is not None:
            pool = pool[: self.candidate_pool_size]

        cross_values: List[float] = []
        retrieval_values: List[float] = []
        for index, candidate in enumerate(pool):
            if cross_encoder_scores is not None and index < len(cross_encoder_scores):
                cross_values.append(float(cross_encoder_scores[index]))
            else:
                cross_values.append(float(candidate.get("cross_encoder_score") or 0.0))
            retrieval_values.append(float(candidate.get("retrieval_score") or candidate.get("score") or 0.0))
        cross_min, cross_max = (min(cross_values), max(cross_values)) if cross_values else (0.0, 0.0)
        retrieval_min, retrieval_max = (min(retrieval_values), max(retrieval_values)) if retrieval_values else (0.0, 0.0)

        scored: List[Dict[str, Any]] = []
        for index, candidate in enumerate(pool):
            doc = dict(candidate)
            score = cross_values[index]
            metrics = self._score_candidate(query, doc, cross_encoder_score=cross_values[index])
            metrics["cross_encoder_score"] = score
            metrics["cross_encoder_normalized"] = self._normalize(score, cross_min, cross_max)
            metrics["retrieval_score_normalized"] = self._normalize(
                retrieval_values[index], retrieval_min, retrieval_max
            )
            rank = int(doc.get("rank") or doc.get("original_rank") or index + 1)
            rank_signal = 1.0 if len(pool) <= 1 else 1.0 - (max(0, rank - 1) / max(1, len(pool) - 1))
            rank_signal = max(0.0, min(1.0, rank_signal))
            metrics["retrieval_rank"] = rank
            metrics["rank_signal"] = rank_signal
            weights = self._normalized_weights()
            final_score = (
                weights.get("cross_encoder", 0.0) * metrics["cross_encoder_normalized"]
                + weights.get("lexical_overlap", 0.0) * metrics["lexical_overlap"]
                + weights.get("scientific_overlap", 0.0) * metrics["scientific_overlap"]
                + weights.get("retrieval_rank", 0.0) * rank_signal
                + weights.get("retrieval_score", 0.0) * metrics["retrieval_score_normalized"]
            )
            metrics["final_score"] = float(final_score)
            final = dict(doc)
            final.update(metrics)
            final["evidence_score"] = float(metrics["final_score"])
            final["final_score"] = float(metrics["final_score"])
            final["retriever_name"] = final.get("retriever_name", "evidence_aware")
            scored.append(final)

        scored.sort(
            key=lambda item: (
                -float(item.get("final_score", 0.0)),
                -float(item.get("cross_encoder_score", 0.0)),
                int(item.get("retrieval_rank", item.get("rank", 10**9))),
                str(item.get("document_id", "")),
            )
        )
        for new_rank, item in enumerate(scored, start=1):
            item["rank"] = new_rank
        limit = len(scored) if top_k is None else max(0, min(len(scored), int(top_k)))
        return scored[:limit]

    def rerank_with_evidence(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        evidence_scores: Mapping[str, float],
        top_k: Optional[int] = None,
        cross_encoder_scores: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Integrate document-level evidence strength without changing the base rerank API."""
        ranked = self.rerank(
            query,
            candidates,
            top_k=len(candidates),
            cross_encoder_scores=cross_encoder_scores,
        )
        evidence_weight = float(self.weights.get("evidence_support", 0.0))
        contradiction_weight = float(self.weights.get("evidence_refutation", 0.0))
        for item in ranked:
            document_id = str(item["document_id"])
            evidence_signal = evidence_scores.get(document_id, {})
            if isinstance(evidence_signal, Mapping):
                support = float(evidence_signal.get("support", 0.0))
                refutation = float(evidence_signal.get("refutation", 0.0))
            else:
                support = float(evidence_signal)
                refutation = 0.0
            evidence_bonus = evidence_weight * support + contradiction_weight * refutation
            item["support_evidence_score"] = support
            item["contradicting_evidence_score"] = refutation
            item["evidence_grounding_score"] = evidence_bonus
            item["final_score"] = float(item["final_score"] + evidence_bonus)
        ranked.sort(
            key=lambda item: (
                -float(item["final_score"]),
                -float(item.get("cross_encoder_score", 0.0)),
                int(item.get("retrieval_rank", item.get("rank", 10**9))),
                str(item["document_id"]),
            )
        )
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        limit = len(ranked) if top_k is None else max(0, min(len(ranked), int(top_k)))
        return ranked[:limit]
