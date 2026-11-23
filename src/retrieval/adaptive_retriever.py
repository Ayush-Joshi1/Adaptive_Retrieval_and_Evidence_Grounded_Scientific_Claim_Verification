from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class ScoreGapAdaptiveStrategy:
    """Use the gap between the top two retrieval scores to decide depth."""

    def __init__(self, threshold: float = 0.10, min_depth: int = 10, max_depth: int = 50, fallback_depth: int = 20):
        self.threshold = float(threshold)
        self.min_depth = max(1, int(min_depth))
        self.max_depth = max(self.min_depth, int(max_depth))
        self.fallback_depth = max(self.min_depth, int(fallback_depth))

    def normalize_scores(self, scores: Sequence[float]) -> List[float]:
        values = [float(score) for score in scores if score is not None]
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if math.isclose(maximum, minimum):
            return [0.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def choose_depth(self, scores: Sequence[float]) -> int:
        values = [float(score) for score in scores if score is not None]
        if not values:
            return self.min_depth
        if len(values) == 1:
            return self.min_depth
        normalized = self.normalize_scores(values)
        top_1 = normalized[0] if normalized else 0.0
        top_2 = normalized[1] if len(normalized) > 1 else 0.0
        gap = top_1 - top_2
        if gap >= self.threshold:
            return self.min_depth
        return self.fallback_depth

    def should_stop(self, scores: Sequence[float]) -> bool:
        values = [float(score) for score in scores if score is not None]
        if len(values) < 2:
            return True
        normalized = self.normalize_scores(values)
        return (normalized[0] - normalized[1]) >= self.threshold


class ConfidenceAdaptiveStrategy:
    """Use a confidence-like statistic derived from score distribution."""

    def __init__(self, threshold: float = 0.25, min_depth: int = 10, max_depth: int = 50, fallback_depth: int = 20):
        self.threshold = float(threshold)
        self.min_depth = max(1, int(min_depth))
        self.max_depth = max(self.min_depth, int(max_depth))
        self.fallback_depth = max(self.min_depth, int(fallback_depth))

    def normalize_scores(self, scores: Sequence[float]) -> List[float]:
        values = [float(score) for score in scores if score is not None]
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if math.isclose(maximum, minimum):
            return [0.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def choose_depth(self, scores: Sequence[float]) -> int:
        values = [float(score) for score in scores if score is not None]
        if len(values) < 2:
            return self.min_depth
        normalized = self.normalize_scores(values)
        mean_value = sum(normalized) / len(normalized)
        variance = sum((value - mean_value) ** 2 for value in normalized) / len(normalized)
        std_dev = math.sqrt(variance)
        top_first = normalized[0]
        confidence = (top_first - mean_value) / (std_dev + 1e-9)
        if confidence >= self.threshold:
            return self.min_depth
        return self.fallback_depth

    def should_stop(self, scores: Sequence[float]) -> bool:
        values = [float(score) for score in scores if score is not None]
        if len(values) < 2:
            return True
        normalized = self.normalize_scores(values)
        mean_value = sum(normalized) / len(normalized)
        variance = sum((value - mean_value) ** 2 for value in normalized) / len(normalized)
        std_dev = math.sqrt(variance)
        confidence = (normalized[0] - mean_value) / (std_dev + 1e-9)
        return confidence >= self.threshold


class AdaptiveRetriever:
    """Retriever wrapper selecting a depth based on the score distribution."""

    def __init__(self, base_retriever: Any, strategy: str = "score_gap", max_depth: int = 50, min_depth: int = 10, threshold: float = 0.10, initial_depth: Optional[int] = None):
        self.base_retriever = base_retriever
        self.max_depth = max(1, int(max_depth))
        self.min_depth = max(1, int(min_depth))
        self.initial_depth = self.min_depth if initial_depth is None else max(1, min(int(initial_depth), self.max_depth))
        self.strategy = self._normalize_strategy(strategy)
        self.threshold = float(threshold)
        self._strategy_impl = self._build_strategy()
        self.last_decision: Dict[str, Any] = {}

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        if strategy is None:
            return "score_gap"
        normalized = str(strategy).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "score_gap": "score_gap",
            "scoregap": "score_gap",
            "gap": "score_gap",
            "confidence": "confidence",
            "confidence_based": "confidence",
            "confidencebased": "confidence",
            "fixed": "score_gap",
        }
        return aliases.get(normalized, normalized)

    def _build_strategy(self):
        fallback_depth = min(self.max_depth, max(self.min_depth + 1, self.max_depth // 2))
        if self.strategy in {"confidence", "confidence_based", "confidencebased"}:
            return ConfidenceAdaptiveStrategy(threshold=self.threshold, min_depth=self.min_depth, max_depth=self.max_depth, fallback_depth=fallback_depth)
        return ScoreGapAdaptiveStrategy(threshold=self.threshold, min_depth=self.min_depth, max_depth=self.max_depth, fallback_depth=fallback_depth)

    def _scores_from_results(self, results: Sequence[Mapping[str, Any]]) -> List[float]:
        scores = []
        for item in results:
            score = item.get("retrieval_score")
            if score is None:
                score = item.get("score")
            if score is None:
                continue
            scores.append(float(score))
        return scores

    def choose_depth(self, results: Sequence[Mapping[str, Any]]) -> int:
        scores = self._scores_from_results(results)
        chosen = self._strategy_impl.choose_depth(scores)
        return min(max(1, int(chosen)), self.max_depth)

    def retrieve(self, query: str, top_k: Optional[int] = None, query_id: Optional[str] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        if self.base_retriever is None:
            return []
        requested_depth = self.max_depth if top_k is None else min(self.max_depth, max(1, int(top_k)))
        initial_depth = min(self.initial_depth, requested_depth)
        initial_results = self.base_retriever.retrieve(query, top_k=initial_depth, query_id=query_id, **kwargs)
        if not initial_results:
            return []
        initial_scores = self._scores_from_results(initial_results)
        expansion_stages = 1
        candidate_results = list(initial_results)
        stopping_signals: Dict[str, float] = {}

        if hasattr(self._strategy_impl, "should_stop"):
            stopped_initial = bool(self._strategy_impl.should_stop(initial_scores))
        else:
            stopped_initial = self.choose_depth(initial_results) <= initial_depth
        stopping_signals["initial"] = float(initial_scores[0] - initial_scores[1]) if len(initial_scores) >= 2 else 0.0
        if stopped_initial or requested_depth <= initial_depth:
            final_depth = min(requested_depth, len(candidate_results))
            self.last_decision = {
                "stopped_early": True,
                "final_depth": final_depth,
                "expansion_stages": expansion_stages,
                "stopping_signals": stopping_signals,
            }
            return list(candidate_results[:final_depth])

        if requested_depth <= 20 and requested_depth > initial_depth:
            candidate_results = list(
                self.base_retriever.retrieve(
                    query,
                    top_k=requested_depth,
                    query_id=query_id,
                    **kwargs,
                )
            )
            expansion_stages += 1
            final_depth = min(requested_depth, len(candidate_results))
            self.last_decision = {
                "stopped_early": True,
                "final_depth": final_depth,
                "expansion_stages": expansion_stages,
                "stopping_signals": stopping_signals,
            }
            return list(candidate_results[:final_depth])

        depth_20 = min(20, requested_depth)
        if len(candidate_results) < depth_20:
            candidate_results = list(self.base_retriever.retrieve(query, top_k=depth_20, query_id=query_id, **kwargs))
            expansion_stages += 1
        scores_20 = self._scores_from_results(candidate_results)
        stopped_at_20 = bool(self._strategy_impl.should_stop(scores_20)) if hasattr(self._strategy_impl, "should_stop") else False
        if stopped_at_20 or requested_depth <= 20:
            final_depth = min(requested_depth, len(candidate_results))
            self.last_decision = {
                "stopped_early": final_depth < requested_depth,
                "final_depth": final_depth,
                "expansion_stages": expansion_stages,
                "stopping_signals": stopping_signals,
            }
            return list(candidate_results[:final_depth])

        if len(candidate_results) < requested_depth:
            candidate_results = list(self.base_retriever.retrieve(query, top_k=requested_depth, query_id=query_id, **kwargs))
            expansion_stages += 1
        self.last_decision = {
            "stopped_early": False,
            "final_depth": min(requested_depth, len(candidate_results)),
            "expansion_stages": expansion_stages,
            "stopping_signals": stopping_signals,
        }
        return list(candidate_results[:requested_depth])
