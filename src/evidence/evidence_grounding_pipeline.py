from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .citation_validator import CitationValidator
from .claim_evidence_mapper import ClaimEvidenceMapper
from .evidence_selector import EvidenceSelector
from .verdict_classifier import VerdictClassifier


class EvidenceGroundingPipeline:
    """End-to-end CPU-friendly evidence grounding over existing retrieval components."""

    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, Any]],
        retriever: Any = None,
        cross_encoder_reranker: Any = None,
        evidence_aware_reranker: Any = None,
        adaptive_retriever: Any = None,
        evidence_selector: Optional[EvidenceSelector] = None,
        evidence_mapper: Optional[ClaimEvidenceMapper] = None,
        citation_validator: Optional[CitationValidator] = None,
        verdict_classifier: Optional[VerdictClassifier] = None,
        config: Optional[Mapping[str, Any] | str | Path] = None,
    ) -> None:
        self.corpus = {str(document_id): dict(document) for document_id, document in corpus.items()}
        loaded_config: Dict[str, Any] = {}
        if isinstance(config, (str, Path)):
            try:
                import yaml
                loaded_config = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
            except Exception:
                loaded_config = {}
        elif isinstance(config, Mapping):
            loaded_config = dict(config)
        selection_config = loaded_config.get("evidence_selection", {})
        mapping_config = loaded_config.get("claim_evidence_mapping", {})
        verdict_config = loaded_config.get("verdict", {})
        reranking_config = loaded_config.get("evidence_aware_reranking", {})
        self.retriever = retriever
        self.cross_encoder_reranker = cross_encoder_reranker
        self.evidence_aware_reranker = evidence_aware_reranker
        if self.evidence_aware_reranker is not None and isinstance(reranking_config.get("weights"), Mapping):
            for key, value in reranking_config["weights"].items():
                self.evidence_aware_reranker.weights[str(key)] = float(value)
        self.adaptive_retriever = adaptive_retriever
        self.evidence_selector = evidence_selector or EvidenceSelector(
            max_sentences_per_document=int(selection_config.get("max_sentences_per_document", 2)),
            max_total_sentences=int(selection_config.get("max_total_sentences", 6)),
            relevance_threshold=float(selection_config.get("relevance_threshold", 0.25)),
            embedding_batch_size=int(selection_config.get("embedding_batch_size", 64)),
        )
        self.evidence_mapper = evidence_mapper or ClaimEvidenceMapper(
            relevance_threshold=float(mapping_config.get("relevance_threshold", 0.25)),
            overlap_threshold=float(mapping_config.get("overlap_threshold", 0.08)),
            contradiction_marker_threshold=int(mapping_config.get("contradiction_marker_threshold", 1)),
        )
        self.citation_validator = citation_validator or CitationValidator(self.corpus)
        self.verdict_classifier = verdict_classifier or VerdictClassifier(
            confidence_threshold=float(verdict_config.get("confidence_threshold", 0.60)),
            relevance_threshold=float(verdict_config.get("relevance_threshold", 0.25)),
            unresolved_conflict_margin=float(verdict_config.get("unresolved_conflict_margin", 0.15)),
        )

    def _retrieve(self, claim: str, claim_id: str, top_k: int) -> list[dict[str, Any]]:
        active_retriever = self.adaptive_retriever or self.retriever
        if active_retriever is None:
            return []
        candidates = active_retriever.retrieve(claim, top_k=top_k, query_id=claim_id)
        hydrated = []
        seen = set()
        for rank, candidate in enumerate(candidates, start=1):
            if isinstance(candidate, Mapping):
                document_id = str(candidate.get("document_id") or candidate.get("corpus_id") or "")
            else:
                document_id = str(candidate)
            if not document_id or document_id in seen or document_id not in self.corpus:
                continue
            seen.add(document_id)
            source = dict(self.corpus[document_id])
            source.update(candidate if isinstance(candidate, Mapping) else {})
            source["document_id"] = document_id
            source["corpus_id"] = document_id
            source.setdefault("rank", rank)
            source.setdefault("original_rank", rank)
            source.setdefault("retrieval_score", candidate.get("retrieval_score", candidate.get("score", 0.0)) if isinstance(candidate, Mapping) else 0.0)
            hydrated.append(source)
        return hydrated[:top_k]

    def process(
        self,
        claim: str,
        claim_id: str = "",
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        top_k: int = 50,
        retrieval_method: str = "dense",
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        claim = "" if claim is None else str(claim)
        claim_id = str(claim_id)
        retrieve_start = time.perf_counter()
        if candidates is None:
            candidates = self._retrieve(claim, claim_id, top_k)
        else:
            candidates = list(candidates)
        retrieval_latency = time.perf_counter() - retrieve_start

        reranking_metadata: Dict[str, Any] = {
            "retrieval_method": retrieval_method,
            "cross_encoder_used": False,
            "evidence_aware_used": False,
        }
        rerank_start = time.perf_counter()
        ranked = list(candidates)
        if ranked and self.cross_encoder_reranker is not None:
            ranked = self.cross_encoder_reranker.rerank(
                claim,
                ranked,
                top_k=len(ranked),
                query_id=claim_id,
            )
            reranking_metadata["cross_encoder_used"] = True

        reranking_latency = time.perf_counter() - rerank_start

        if ranked and self.evidence_aware_reranker is not None:
            reranking_metadata["evidence_aware_used"] = True
            cross_scores = [float(row.get("cross_encoder_score", 0.0)) for row in ranked]
            evidence_rerank_start = time.perf_counter()
            ranked = self.evidence_aware_reranker.rerank_with_evidence(
                claim,
                ranked,
                {},
                top_k=len(ranked),
                cross_encoder_scores=cross_scores,
            )
            reranking_latency += time.perf_counter() - evidence_rerank_start

        evidence_start = time.perf_counter()
        evidence_items = self.evidence_selector.select(claim, ranked)
        evidence_latency = time.perf_counter() - evidence_start

        retrieval_scores = {
            str(row.get("document_id") or row.get("corpus_id")):
            float(row.get("retrieval_score", row.get("score", 0.0)))
            for row in ranked
        }
        source_metadata = {
            str(row.get("document_id") or row.get("corpus_id")): {
                "title": row.get("title"),
                "corpus_id": str(row.get("document_id") or row.get("corpus_id")),
                "text": row.get("text"),
            }
            for row in ranked
        }
        mapping_start = time.perf_counter()
        mappings = self.evidence_mapper.map(
            claim_id,
            claim,
            evidence_items,
            retrieval_scores=retrieval_scores,
            source_metadata=source_metadata,
        )
        mapping_latency = time.perf_counter() - mapping_start

        citation_start = time.perf_counter()
        citation_validation = self.citation_validator.validate(mappings)
        citation_latency = time.perf_counter() - citation_start

        evidence_scores: Dict[str, Dict[str, float]] = {}
        for mapping in mappings:
            document_id = str(mapping["document_id"])
            evidence_scores.setdefault(document_id, {"support": 0.0, "refutation": 0.0})
            relationship = str(mapping.get("relationship"))
            relevance = float(mapping.get("evidence_relevance_score", 0.0))
            if relationship == "SUPPORTS":
                evidence_scores[document_id]["support"] = max(evidence_scores[document_id]["support"], relevance)
            elif relationship == "REFUTES":
                evidence_scores[document_id]["refutation"] = max(evidence_scores[document_id]["refutation"], relevance)
        if ranked and self.evidence_aware_reranker is not None:
            reranking_metadata["evidence_aware_used"] = True
            cross_scores = [float(row.get("cross_encoder_score", 0.0)) for row in ranked]
            evidence_rerank_start = time.perf_counter()
            ranked = self.evidence_aware_reranker.rerank_with_evidence(
                claim,
                ranked,
                evidence_scores,
                top_k=len(ranked),
                cross_encoder_scores=cross_scores,
            )
            reranking_latency += time.perf_counter() - evidence_rerank_start

        verdict_result = self.verdict_classifier.classify(mappings, citation_validation)

        source_document_ids = sorted({
            str(row.get("document_id") or row.get("corpus_id"))
            for row in ranked
            if row.get("document_id") or row.get("corpus_id")
        })
        return {
            "claim_id": claim_id,
            "claim_text": claim,
            "final_verdict": verdict_result["verdict"],
            "confidence": verdict_result["confidence"],
            "evidence_mappings": [dict(item) for item in mappings],
            "supporting_evidence": verdict_result["supporting_evidence"],
            "contradicting_evidence": verdict_result["contradicting_evidence"],
            "neutral_evidence": verdict_result["neutral_evidence"],
            "source_document_ids": source_document_ids,
            "citation_validation": citation_validation,
            "abstention_reason": verdict_result["abstention_reason"],
            "retrieval_metadata": {
                "candidate_count": len(ranked),
                "candidate_document_ids": [
                    str(row.get("document_id") or row.get("corpus_id"))
                    for row in ranked
                ],
                "retrieval_method": retrieval_method,
                "adaptive": self.adaptive_retriever is not None,
                "adaptive_decision": getattr(self.adaptive_retriever, "last_decision", {}) if self.adaptive_retriever is not None else {},
            },
            "reranking_metadata": reranking_metadata,
            "mapping_latency_seconds": mapping_latency,
            "retrieval_latency_seconds": retrieval_latency,
            "reranking_latency_seconds": reranking_latency,
            "evidence_selection_latency_seconds": evidence_latency,
            "citation_validation_latency_seconds": citation_latency,
            "processing_latency_seconds": time.perf_counter() - start,
        }
