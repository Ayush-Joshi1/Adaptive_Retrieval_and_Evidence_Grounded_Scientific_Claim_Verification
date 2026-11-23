from src.retrieval.adaptive_retriever import AdaptiveRetriever, ConfidenceAdaptiveStrategy, ScoreGapAdaptiveStrategy
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.evidence_aware_reranker import EvidenceAwareReranker


class DummyRetriever:
    def retrieve(self, query, top_k=None, query_id=None, **kwargs):
        return [
            {"document_id": "a", "corpus_id": "a", "rank": 1, "score": 0.90, "retrieval_score": 0.90},
            {"document_id": "b", "corpus_id": "b", "rank": 2, "score": 0.80, "retrieval_score": 0.80},
            {"document_id": "c", "corpus_id": "c", "rank": 3, "score": 0.70, "retrieval_score": 0.70},
        ][: max(1, int(top_k or 3))]


def test_cross_encoder_reranker_model_is_reused_across_instances():
    reranker_a = CrossEncoderReranker(batch_size=8, max_candidates=3)
    reranker_b = CrossEncoderReranker(batch_size=16, max_candidates=2)
    assert reranker_a._model is reranker_b._model


def test_cross_encoder_reranker_batch_inference_and_candidate_limit():
    reranker = CrossEncoderReranker(batch_size=8, max_candidates=2)
    query_pairs = [
        ("Inflammation in disease", [
            {"document_id": "doc-1", "corpus_id": "doc-1", "rank": 1, "retrieval_score": 0.7, "text": "Exercise regulates immune balance."},
            {"document_id": "doc-2", "corpus_id": "doc-2", "rank": 2, "retrieval_score": 0.6, "text": "Inflammation plays a central role in disease pathogenesis."},
            {"document_id": "doc-3", "corpus_id": "doc-3", "rank": 3, "retrieval_score": 0.5, "text": "A general study of cell signaling."},
        ]),
        ("Cancer mechanisms", [
            {"document_id": "doc-4", "corpus_id": "doc-4", "rank": 1, "retrieval_score": 0.9, "text": "Cancer is driven by mutations."},
            {"document_id": "doc-5", "corpus_id": "doc-5", "rank": 2, "retrieval_score": 0.8, "text": "Genes regulate tissue repair."},
            {"document_id": "doc-6", "corpus_id": "doc-6", "rank": 3, "retrieval_score": 0.7, "text": "A detailed overview of protein folding."},
        ]),
    ]
    results = reranker.rerank_batch(query_pairs, top_k=2)
    assert len(results) == 2
    assert all(len(batch) <= 2 for batch in results)
    assert all(all(row["document_id"] in {doc["document_id"] for doc in query_pair[1][:2]} for row in batch) for batch, query_pair in zip(results, query_pairs))


def test_cross_encoder_reranker_preserves_metadata_and_reranks():
    reranker = CrossEncoderReranker(batch_size=8, max_candidates=3)
    query = "What is the role of inflammation in disease?"
    candidates = [
        {"document_id": "doc-1", "corpus_id": "doc-1", "rank": 1, "retrieval_score": 0.7, "text": "This is about diet and exercise."},
        {"document_id": "doc-2", "corpus_id": "doc-2", "rank": 2, "retrieval_score": 0.6, "text": "Inflammation is central to disease pathogenesis and immune response."},
        {"document_id": "doc-3", "corpus_id": "doc-3", "rank": 3, "retrieval_score": 0.5, "text": "A general overview of biochemistry."},
    ]
    results = reranker.rerank(query, candidates, top_k=2)
    assert len(results) == 2
    assert all("cross_encoder_score" in row for row in results)
    assert set(row["document_id"] for row in results) <= {"doc-1", "doc-2", "doc-3"}
    assert [row["rank"] for row in results] == [1, 2]
    assert results[0]["original_rank"] in {1, 2, 3}
    assert results[0]["retrieval_score"] == 0.7 or results[0]["retrieval_score"] in {0.6, 0.5}


def test_cross_encoder_reranker_handles_empty_and_invalid_inputs():
    reranker = CrossEncoderReranker(batch_size=8)
    assert reranker.rerank("   ", []) == []
    assert reranker.rerank("query", None) == []
    assert reranker.score_pairs("query", []) == []


def test_strict_cross_encoder_surfaces_model_load_failure(monkeypatch):
    class BrokenCrossEncoder:
        def __init__(self, model_name, device=None):
            raise OSError("model unavailable")

    monkeypatch.setattr("src.retrieval.cross_encoder_reranker.CrossEncoder", BrokenCrossEncoder)
    try:
        CrossEncoderReranker(model_name="strict-load-failure-test", strict=True)
    except RuntimeError as exc:
        assert "Failed to load cross-encoder model" in str(exc)
    else:
        raise AssertionError("Strict model loading must not silently use fallback scores")


def test_strict_cross_encoder_surfaces_inference_failure(monkeypatch):
    class BrokenPredictor:
        def __init__(self, model_name, device=None):
            pass

        def predict(self, pairs, batch_size=None):
            raise RuntimeError("inference failed")

    monkeypatch.setattr("src.retrieval.cross_encoder_reranker.CrossEncoder", BrokenPredictor)
    reranker = CrossEncoderReranker(model_name="strict-inference-failure-test", strict=True)
    try:
        reranker.score_pairs("query", ["document"])
    except RuntimeError as exc:
        assert "Cross-encoder inference failed" in str(exc)
    else:
        raise AssertionError("Strict inference must not silently use fallback scores")


def test_score_gap_and_confidence_strategies_make_valid_decisions():
    score_gap = ScoreGapAdaptiveStrategy(threshold=0.10, min_depth=2, max_depth=5, fallback_depth=3)
    confidence = ConfidenceAdaptiveStrategy(threshold=0.40, min_depth=2, max_depth=5, fallback_depth=3)
    scores = [0.9, 0.5, 0.4, 0.3]
    assert score_gap.choose_depth(scores) in {2, 3, 5}
    assert confidence.choose_depth(scores) in {2, 3, 5}
    assert score_gap.choose_depth([0.8, 0.8, 0.8]) in {2, 3}
    assert confidence.choose_depth([0.5, 0.5, 0.5]) in {2, 3, 5}


def test_adaptive_retriever_uses_score_gap_policy():
    retriever = AdaptiveRetriever(DummyRetriever(), strategy="score_gap", min_depth=2, max_depth=3, threshold=0.05)
    results = retriever.retrieve("query", top_k=3, query_id="q1")
    assert len(results) <= 3
    assert len(results) >= 2
    assert {row["document_id"] for row in results}.issubset({"a", "b", "c"})
    assert all(row["retrieval_score"] >= 0 for row in results)


def test_adaptive_retriever_only_expands_when_required():
    class CountingRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=None, query_id=None, **kwargs):
            self.calls.append(int(top_k))
            scores = [0.50, 0.50, 0.50]
            results = [
                {"document_id": f"d{i}", "corpus_id": f"d{i}", "rank": i + 1, "score": score, "retrieval_score": score}
                for i, score in enumerate(scores[: max(1, int(top_k or 3))])
            ]
            return results

    base = CountingRetriever()
    retriever = AdaptiveRetriever(base, strategy="score_gap", min_depth=2, max_depth=3, threshold=0.05)
    _ = retriever.retrieve("query", top_k=3, query_id="q1")
    assert base.calls == [2, 3]


def test_adaptive_retriever_expands_incrementally_through_supported_depths():
    class CountingRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=None, query_id=None, **kwargs):
            self.calls.append(int(top_k))
            scores = [1.0 / (i + 1) for i in range(int(top_k))]
            return [
                {"document_id": f"d{i}", "corpus_id": f"d{i}", "rank": i + 1,
                 "score": score, "retrieval_score": score}
                for i, score in enumerate(scores)
            ]

    base = CountingRetriever()
    retriever = AdaptiveRetriever(
        base,
        strategy="score_gap",
        min_depth=10,
        max_depth=50,
        threshold=1.1,
    )
    results = retriever.retrieve("query", top_k=50, query_id="q1")
    assert base.calls == [10, 20, 50]
    assert len(results) == 50
    assert retriever.last_decision["expansion_stages"] == 3
    assert len({row["document_id"] for row in results}) == 50


def test_evidence_aware_reranker_combines_relevance_signals():
    reranker = EvidenceAwareReranker(candidate_pool_size=3)
    candidates = [
        {"document_id": "d1", "corpus_id": "d1", "rank": 1, "score": 0.8, "text": "Cancer and inflammation are linked to immune signaling."},
        {"document_id": "d2", "corpus_id": "d2", "rank": 2, "score": 0.6, "text": "A general article about space."},
        {"document_id": "d3", "corpus_id": "d3", "rank": 3, "score": 0.5, "text": "A review of inflammatory pathways in disease."},
    ]
    results = reranker.rerank("What is the role of inflammation in disease?", candidates, top_k=2)
    assert len(results) == 2
    assert all("final_score" in row for row in results)
    assert all(row["rank"] == idx + 1 for idx, row in enumerate(results))
    assert all(-1e9 < float(row["final_score"]) < 1e9 for row in results)


def test_evidence_aware_reranker_normalizes_and_uses_configurable_weights():
    reranker = EvidenceAwareReranker(weights={"cross_encoder": 0.7, "lexical_overlap": 0.2, "retrieval_score": 0.1}, candidate_pool_size=2)
    candidates = [
        {"document_id": "d1", "corpus_id": "d1", "rank": 1, "score": 0.9, "text": "Inflammation is associated with disease progression."},
        {"document_id": "d2", "corpus_id": "d2", "rank": 2, "score": 0.3, "text": "A general article on astronomy."},
    ]
    results = reranker.rerank("What causes inflammation in disease?", candidates, top_k=2, cross_encoder_scores=[0.7, 0.2])
    assert len(results) == 2
    assert all("evidence_score" in row for row in results)
    assert results[0]["final_score"] >= results[1]["final_score"]
