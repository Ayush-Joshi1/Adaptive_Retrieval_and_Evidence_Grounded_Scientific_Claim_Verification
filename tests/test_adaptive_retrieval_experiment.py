import json

import pytest

from experiments.run_adaptive_retrieval_experiment import _adaptive_candidates, _valid_checkpoint
from src.retrieval.adaptive_retriever import AdaptiveRetriever


class CountingRetriever:
    def __init__(self, scores=None):
        self.calls = []
        self.scores = scores

    def retrieve(self, query, top_k=None, query_id=None, **kwargs):
        depth = int(top_k)
        self.calls.append(depth)
        values = self.scores or [0.5] * depth
        return [
            {
                "query_id": query_id,
                "document_id": f"doc-{index}",
                "corpus_id": f"doc-{index}",
                "rank": index + 1,
                "score": float(values[index]) if index < len(values) else 0.5,
                "retrieval_score": float(values[index]) if index < len(values) else 0.5,
            }
            for index in range(depth)
        ]


def test_adaptive_candidates_expand_only_after_stopping_signal_fails():
    base = CountingRetriever()
    adaptive = AdaptiveRetriever(
        base,
        strategy="score_gap",
        min_depth=10,
        max_depth=50,
        threshold=0.1,
        initial_depth=10,
    )

    candidates, decision, latency, _ = _adaptive_candidates(
        "dense", base, adaptive, "claim", "q1", {}
    )

    assert base.calls == [10, 20, 50]
    assert len(candidates) == 50
    assert decision["stage_depths"] == [10, 20, 50]
    assert decision["number_of_expansion_stages"] == 2
    assert [row["new_candidate_count"] for row in decision["stage_records"]] == [10, 10, 30]
    assert decision["candidate_ids"] == [f"doc-{index}" for index in range(50)]
    assert decision["stopping_signals"]["after_10"]["should_stop"] is False
    assert latency >= 0


def test_adaptive_candidates_stop_at_ten_without_requesting_more():
    base = CountingRetriever([1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02])
    adaptive = AdaptiveRetriever(
        base,
        strategy="score_gap",
        min_depth=10,
        max_depth=50,
        threshold=0.1,
        initial_depth=10,
    )

    candidates, decision, _, _ = _adaptive_candidates(
        "dense", base, adaptive, "claim", "q1", {}
    )

    assert base.calls == [10]
    assert len(candidates) == decision["final_candidate_depth"] == 10
    assert decision["stopped_at_10"] is True
    assert decision["number_of_expansion_stages"] == 0


def test_adaptive_candidates_reject_duplicate_ids_on_expansion():
    class DuplicateRetriever(CountingRetriever):
        def retrieve(self, query, top_k=None, query_id=None, **kwargs):
            depth = int(top_k)
            self.calls.append(depth)
            ids = [f"doc-{index}" for index in range(depth)]
            if depth == 20:
                ids[-1] = ids[0]
            return [
                {
                    "query_id": query_id,
                    "document_id": doc_id,
                    "corpus_id": doc_id,
                    "rank": index + 1,
                    "score": 0.5,
                    "retrieval_score": 0.5,
                }
                for index, doc_id in enumerate(ids)
            ]

    base = DuplicateRetriever()
    adaptive = AdaptiveRetriever(
        base,
        strategy="score_gap",
        min_depth=10,
        max_depth=50,
        threshold=0.1,
        initial_depth=10,
    )

    with pytest.raises(AssertionError, match="Duplicate candidates at adaptive depth 20"):
        _adaptive_candidates("dense", base, adaptive, "claim", "q1", {})
    assert base.calls == [10, 20]


def test_adaptive_candidates_accept_hybrid_prefix_reordering_without_duplicates():
    class ReorderingRetriever(CountingRetriever):
        def retrieve(self, query, top_k=None, query_id=None, **kwargs):
            depth = int(top_k)
            self.calls.append(depth)
            ids = [f"doc-{index}" for index in range(depth)]
            if depth == 20:
                ids = ids[10:] + ids[:10]
            elif depth == 50:
                ids = ids[20:] + ids[:20]
            return [
                {
                    "query_id": query_id,
                    "document_id": doc_id,
                    "corpus_id": doc_id,
                    "rank": index + 1,
                    "score": 0.5,
                    "retrieval_score": 0.5,
                }
                for index, doc_id in enumerate(ids)
            ]

    base = ReorderingRetriever()
    adaptive = AdaptiveRetriever(
        base,
        strategy="score_gap",
        min_depth=10,
        max_depth=50,
        threshold=0.1,
        initial_depth=10,
    )

    candidates, decision, _, _ = _adaptive_candidates(
        "hybrid",
        base,
        adaptive,
        "claim",
        "q1",
        {f"doc-{index}": {"title": "", "text": ""} for index in range(50)},
    )

    assert base.calls == [10, 20, 50]
    assert len({row["document_id"] for row in candidates}) == 50
    assert decision["number_unique_candidates_seen_across_stages"] == 50
    assert decision["stage_records"][1]["added_candidate_ids"] == [
        f"doc-{index}" for index in range(10, 20)
    ]


def test_checkpoint_requires_each_unique_configuration(tmp_path):
    configs = [
        f"{method}_fixed_top_{depth}"
        for method in ("dense", "hybrid")
        for depth in (10, 20, 50)
    ] + [
        f"{method}_{strategy}_adaptive"
        for method in ("dense", "hybrid")
        for strategy in ("score_gap", "confidence")
    ]
    fields = ("metric_rows", "ranking_rows", "decision_rows", "performance_rows")
    checkpoint = {
        "query_id": "q1",
        **{
            field: [
                {
                    "query_id": "q1",
                    "retrieval_method": "dense" if config.startswith("dense") else "hybrid",
                    "configuration": config,
                }
                for config in configs
            ]
            for field in fields
        },
    }
    path = tmp_path / "q1.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")

    assert _valid_checkpoint(path, "q1") == checkpoint

    checkpoint["decision_rows"][-1] = dict(checkpoint["decision_rows"][0])
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate or missing configuration"):
        _valid_checkpoint(path, "q1")
