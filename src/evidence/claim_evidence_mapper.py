from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


CONTRADICTION_MARKERS = {
    "no", "not", "none", "neither", "without", "absence", "fail", "fails", "failed",
    "contradict", "contrary", "opposite", "decrease", "decreased", "decline",
    "inhibit", "inhibits", "inhibited", "reduce", "reduces", "reduced",
    "worse", "lower", "lowered", "decreases",
}
SUPPORT_MARKERS = {
    "support", "supports", "associated", "demonstrates", "show", "shows",
    "observed", "increase", "increases", "increased", "improves", "improved",
    "promotes", "induces", "higher", "positive", "confirms",
}
ANTONYM_PAIRS = (
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("improve", "worsen"),
    ("improved", "worsened"),
    ("higher", "lower"),
    ("activate", "inhibit"),
    ("activated", "inhibited"),
    ("promotes", "suppresses"),
    ("positive", "negative"),
    ("improves", "decreases"),
    ("improve", "decrease"),
    ("increases", "decreases"),
)


def _tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


class ClaimEvidenceMapper:
    """Rule-based baseline mapping of claim/evidence relevance to relationship classes."""

    def __init__(
        self,
        relevance_threshold: float = 0.25,
        overlap_threshold: float = 0.08,
        contradiction_marker_threshold: int = 1,
    ) -> None:
        self.relevance_threshold = float(relevance_threshold)
        self.overlap_threshold = float(overlap_threshold)
        self.contradiction_marker_threshold = max(1, int(contradiction_marker_threshold))

    def classify(self, claim: str, evidence_sentence: str, relevance_score: float) -> str:
        claim_tokens = _tokens(claim)
        evidence_tokens = _tokens(evidence_sentence)
        relevance_score = float(relevance_score)
        if relevance_score < self.relevance_threshold:
            return "INSUFFICIENT"
        if not claim_tokens or not evidence_tokens:
            return "INSUFFICIENT"
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens | evidence_tokens))
        if overlap < self.overlap_threshold:
            return "NEUTRAL"

        claim_negated = bool(claim_tokens & {"no", "not", "without", "absence", "none"})
        evidence_negated = bool(evidence_tokens & {"no", "not", "without", "absence", "none"})
        contradiction_markers = len(evidence_tokens & CONTRADICTION_MARKERS)

        if claim_negated != evidence_negated and contradiction_markers >= 1:
            return "REFUTES"

        for positive, negative in ANTONYM_PAIRS:
            if positive in claim_tokens and negative in evidence_tokens:
                return "REFUTES"
            if negative in claim_tokens and positive in evidence_tokens:
                return "REFUTES"

        if contradiction_markers >= self.contradiction_marker_threshold:
            return "REFUTES"

        support_markers = evidence_tokens & SUPPORT_MARKERS
        if support_markers:
            return "SUPPORTS"
        return "NEUTRAL"

    def map(
        self,
        claim_id: str,
        claim: str,
        evidence_items: Sequence[Mapping[str, Any]],
        retrieval_scores: Optional[Mapping[str, float]] = None,
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        retrieval_scores = retrieval_scores or {}
        source_metadata = source_metadata or {}
        mappings: List[Dict[str, Any]] = []
        for evidence in evidence_items:
            document_id = str(evidence.get("document_id") or evidence.get("corpus_id") or "")
            if not document_id:
                continue
            sentence = str(evidence.get("evidence_sentence") or evidence.get("sentence") or "")
            score = float(evidence.get("evidence_relevance_score", evidence.get("relevance_score", 0.0)))
            relationship = self.classify(claim, sentence, score)
            mappings.append({
                "claim_id": str(claim_id),
                "claim_text": claim,
                "document_id": document_id,
                "evidence_sentence": sentence,
                "sentence_index": int(evidence.get("sentence_index", 0)),
                "evidence_relevance_score": score,
                "retrieval_score": float(retrieval_scores.get(document_id, evidence.get("retrieval_score", 0.0))),
                "relationship": relationship,
                "source_metadata": dict(source_metadata.get(document_id, evidence.get("source_metadata", {}))),
                "start_char": int(evidence.get("start_char", 0)),
                "end_char": int(evidence.get("end_char", 0)),
                "source_document_ids": list(evidence.get("source_document_ids", [document_id])),
                "source_traceability": list(evidence.get("source_traceability", [])),
            })
        return mappings
