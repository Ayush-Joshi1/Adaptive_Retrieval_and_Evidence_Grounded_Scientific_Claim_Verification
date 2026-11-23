from src.evaluation.retrieval_metrics import compute_retrieval_metrics, hit_at_k, precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k


def test_hit_and_recall_metrics_for_binary_qrels():
    retrieved = ["d2", "d1", "d3"]
    qrels = {"d1": 1.0, "d3": 1.0}
    metrics = compute_retrieval_metrics(retrieved, qrels, ks=[1, 2, 3])
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@2"] == 1.0
    assert metrics["precision@3"] == 2 / 3
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 1 / 2


def test_empty_retrieval_output_returns_zeroes():
    qrels = {"d1": 1.0, "d2": 1.0}
    metrics = compute_retrieval_metrics([], qrels, ks=[1, 3, 5])
    assert metrics["hit@1"] == 0.0
    assert metrics["precision@3"] == 0.0
    assert metrics["recall@3"] == 0.0
    assert metrics["ndcg@5"] == 0.0


def test_queries_without_qrels_are_treated_as_zero_recall():
    retrieved = ["x1", "x2"]
    qrels = {}
    assert hit_at_k(retrieved, qrels, 1) == 0.0
    assert precision_at_k(retrieved, qrels, 2) == 0.0
    assert recall_at_k(retrieved, qrels, 2) == 0.0
    assert mrr_at_k(retrieved, qrels, 2) == 0.0
    assert ndcg_at_k(retrieved, qrels, 2) == 0.0
