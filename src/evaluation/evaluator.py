from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from .classification_metrics import classification_report
from .retrieval_metrics import compute_retrieval_metrics, mean_metric_by_query


def evaluate_retrieval_queries(
    retrieval_results: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Mapping[str, float] | Iterable[str]],
    ks: Sequence[int],
) -> Dict[str, object]:
    per_query: Dict[str, Dict[str, float]] = {}
    for query_id, retrieved in retrieval_results.items():
        qrels = qrels_by_query.get(query_id, {})
        per_query[str(query_id)] = compute_retrieval_metrics(retrieved, qrels, ks)
    summary: Dict[str, float] = {}
    for metric_name in [f"{prefix}@{k}" for k in ks for prefix in ("hit", "precision", "recall", "mrr", "map", "ndcg")]:
        values = [float(item.get(metric_name, 0.0)) for item in per_query.values()]
        summary[metric_name] = sum(values) / float(len(values)) if values else 0.0
    return {"per_query": per_query, "summary": summary}


def evaluate_classification_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    return classification_report(y_true, y_pred)
