from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


class VerdictClassifier:
    """Configurable abstaining verdict classifier for local evidence baselines."""

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        relevance_threshold: float = 0.25,
        unresolved_conflict_margin: float = 0.15,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.relevance_threshold = float(relevance_threshold)
        self.unresolved_conflict_margin = float(unresolved_conflict_margin)

    def classify(
        self,
        mappings: Sequence[Mapping[str, Any]],
        citation_validation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        valid_keys = {
            (str(item["document_id"]), int(item["sentence_index"]))
            for item in citation_validation.get("valid", [])
        }
        decisive = []
        supporting = []
        contradicting = []
        neutral = []
        invalid_decisive = []

        for mapping in mappings:
            relationship = str(mapping.get("relationship", "INSUFFICIENT")).upper()
            relevance = float(mapping.get("evidence_relevance_score", 0.0))
            key = (str(mapping.get("document_id")), int(mapping.get("sentence_index", 0)))
            citation_valid = key in valid_keys
            item = dict(mapping)
            item["citation_valid"] = citation_valid
            if relationship == "SUPPORTS" or relationship == "REFUTES":
                if not citation_valid:
                    invalid_decisive.append(item)
                    continue
                if relevance >= self.relevance_threshold:
                    decisive.append(item)
            if relationship == "SUPPORTS" and relevance >= self.relevance_threshold:
                supporting.append(item)
            elif relationship == "REFUTES" and relevance >= self.relevance_threshold:
                contradicting.append(item)
            else:
                neutral.append(item)

        def strength(items: Sequence[Mapping[str, Any]]) -> float:
            if not items:
                return 0.0
            unique_documents = {str(item["document_id"]) for item in items}
            mean_relevance = sum(float(item.get("evidence_relevance_score", 0.0)) for item in items) / len(items)
            diversity_bonus = min(0.15, 0.05 * max(0, len(unique_documents) - 1))
            return min(1.0, mean_relevance * 0.85 + diversity_bonus)

        support_strength = strength(supporting)
        refute_strength = strength(contradicting)
        margin = abs(support_strength - refute_strength)

        if not decisive:
            if invalid_decisive:
                return self._result("NOT_ENOUGH_INFO", 0.0, "decisive_citations_failed_validation", supporting, contradicting, neutral)
            return self._result("NOT_ENOUGH_INFO", 0.0, "no_decisive_citations", supporting, contradicting, neutral)
        if support_strength > 0.0 and refute_strength > 0.0:
            if margin < self.unresolved_conflict_margin:
                return self._result("NOT_ENOUGH_INFO", min(support_strength, refute_strength), "conflicting_evidence_unresolved", supporting, contradicting, neutral)

        if support_strength > refute_strength:
            confidence = min(1.0, support_strength)
            verdict = "SUPPORTS"
        elif refute_strength > support_strength:
            confidence = min(1.0, refute_strength)
            verdict = "REFUTES"
        else:
            return self._result("NOT_ENOUGH_INFO", 0.0, "evidence_strength_tied", supporting, contradicting, neutral)

        if confidence < self.confidence_threshold:
            return self._result("NOT_ENOUGH_INFO", confidence, "confidence_below_threshold", supporting, contradicting, neutral)
        return self._result(verdict, confidence, None, supporting, contradicting, neutral)

    @staticmethod
    def _result(
        verdict: str,
        confidence: float,
        abstention_reason: str | None,
        supporting: Sequence[Mapping[str, Any]],
        contradicting: Sequence[Mapping[str, Any]],
        neutral: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "verdict": verdict,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "abstention_reason": abstention_reason,
            "supporting_evidence": [dict(item) for item in supporting],
            "contradicting_evidence": [dict(item) for item in contradicting],
            "neutral_evidence": [dict(item) for item in neutral],
        }
