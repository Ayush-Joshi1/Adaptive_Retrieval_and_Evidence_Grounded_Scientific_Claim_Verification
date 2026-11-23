from __future__ import annotations

from math import log2
from typing import Dict, Iterable, List, Mapping, Sequence, Set


def _as_relevance_map(qrels: Mapping[str, float] | Iterable[str] | Set[str]) -> Dict[str, float]:
    if isinstance(qrels, Mapping):
        return {str(doc_id): float(score) for doc_id, score in qrels.items()}
    if isinstance(qrels, (set, list, tuple)):
        return {str(doc_id): 1.0 for doc_id in qrels}
    raise TypeError("qrels must be a mapping or an iterable of relevant document IDs")


def hit_at_k(retrieved: Sequence[str], qrels: Mapping[str, float] | Iterable[str] | Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {doc_id for doc_id, score in _as_relevance_map(qrels).items() if float(score) > 0}
    top_k = [str(doc_id) for doc_id in retrieved[:k]]
    return 1.0 if any(doc_id in relevant for doc_id in top_k) else 0.0


def precision_at_k(retrieved: Sequence[str], qrels: Mapping[str, float] | Iterable[str] | Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = _as_relevance_map(qrels)
    top_k = [str(doc_id) for doc_id in retrieved[:k]]
    retrieved_relevant = sum(1 for doc_id in top_k if float(relevant.get(doc_id, 0.0)) > 0)
    return retrieved_relevant / float(k)


def recall_at_k(retrieved: Sequence[str], qrels: Mapping[str, float] | Iterable[str] | Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = _as_relevance_map(qrels)
    total_relevant = sum(1 for score in relevant.values() if float(score) > 0)
    if total_relevant == 0:
        return 0.0
    top_k = [str(doc_id) for doc_id in retrieved[:k]]
    retrieved_relevant = sum(1 for doc_id in top_k if float(relevant.get(doc_id, 0.0)) > 0)
    return retrieved_relevant / float(total_relevant)


def mrr_at_k(retrieved: Sequence[str], qrels: Mapping[str, float] | Iterable[str] | Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {doc_id for doc_id, score in _as_relevance_map(qrels).items() if float(score) > 0}
    top_k = [str(doc_id) for doc_id in retrieved[:k]]
    for position, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            return 1.0 / float(position)
    return 0.0


def average_precision_at_k(retrieved: Sequence[str], qrels: Mapping[str, float] | Iterable[str] | Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = _as_relevance_map(qrels)
    total_relevant = sum(1 for score in relevant.values() if float(score) > 0)
    if total_relevant == 0:
        return 0.0
    top_k = [str(doc_id) for doc_id in retrieved[:k]]
    precision_sum = 0.0
    hits = 0
    for position, doc_id in enumerate(top_k, start=1):
        if float(relevant.get(doc_id, 0.0)) > 0:
            hits += 1
            precision_sum += hits / float(position)
    return precision_sum / float(min(total_relevant, k))


def ndcg_at_k(retrieved: Sequence[str], qrels: Mapping[str, float] | Iterable[str] | Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel_map = _as_relevance_map(qrels)
    top_k = [str(doc_id) for doc_id in retrieved[:k]]
    dcg = 0.0
    for index, doc_id in enumerate(top_k, start=1):
        rel = max(float(rel_map.get(doc_id, 0.0)), 0.0)
        if rel <= 0:
            continue
        dcg += ((2 ** rel) - 1.0) / log2(index + 1)

    ideal_scores = sorted(rel_map.values(), reverse=True)[:k]
    idcg = 0.0
    for index, rel in enumerate(ideal_scores, start=1):
        if rel <= 0:
            continue
        idcg += ((2 ** float(rel)) - 1.0) / log2(index + 1)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(retrieved: Sequence[str], qrels: Mapping[str, float] | Iterable[str] | Set[str], ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(retrieved, qrels, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved, qrels, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, qrels, k)
        metrics[f"mrr@{k}"] = mrr_at_k(retrieved, qrels, k)
        metrics[f"map@{k}"] = average_precision_at_k(retrieved, qrels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, qrels, k)
    return metrics


def mean_metric_by_query(results: Sequence[Dict[str, float]], metric_name: str) -> float:
    if not results:
        return 0.0
    return sum(float(item.get(metric_name, 0.0)) for item in results) / float(len(results))
