from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.statistical_tests import holm_bonferroni_adjust, mcnemar_test, paired_bootstrap_ci
from src.experiments.artifact_manager import ArtifactManager
from src.experiments.run_manager import unique_run_dir
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import Retriever

RAW_DEV_CLAIMS_PATH = ROOT / "data" / "processed" / "raw_scifact" / "data" / "claims_dev.jsonl"
K_VALUES = [1, 3, 5, 10, 20]
SENSITIVITY_CONSTANTS = [10, 30, 60, 100]
SENSITIVITY_K_VALUES = [5, 10, 20]


def load_dev_queries():
    rows = []
    with RAW_DEV_CLAIMS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                rows.append({
                    "query_id": str(record["id"]),
                    "claim_text": record.get("claim") or "",
                    "ground_truth_label": record.get("label"),
                })
    return rows


def load_qrels_by_query():
    qrels = defaultdict(dict)
    dataset = load_dataset("BeIR/scifact-qrels")
    for row in dataset["test"]:
        qid = str(row["query-id"])
        qrels[qid][str(row["corpus-id"])] = float(row["score"])
    return dict(qrels)


def metric_rows_for_method(rows, method_name, qrels_by_query):
    per_query = []
    for row in rows:
        query_id = str(row["query_id"])
        qrels = qrels_by_query.get(query_id, {})
        metrics = compute_retrieval_metrics(row["retrieved"], qrels, K_VALUES)
        metrics["query_id"] = query_id
        metrics["method"] = method_name
        metrics["retrieved_count"] = len(row["retrieved"])
        per_query.append(metrics)
    return per_query


def summarize_metrics(metric_rows):
    summary = {}
    for metric_name in [
        *[f"hit@{k}" for k in K_VALUES],
        *[f"precision@{k}" for k in K_VALUES],
        *[f"recall@{k}" for k in K_VALUES],
        *[f"mrr@{k}" for k in K_VALUES],
        *[f"map@{k}" for k in K_VALUES],
        *[f"ndcg@{k}" for k in K_VALUES],
    ]:
        values = [float(row.get(metric_name, 0.0)) for row in metric_rows]
        summary[metric_name] = sum(values) / len(values) if values else 0.0
    return summary


def measure_latency(func, *, repeats=5):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return {
        "mean": sum(timings) / len(timings),
        "median": median(timings),
        "p95": sorted(timings)[max(0, int(round(0.95 * len(timings))) - 1)],
        "queries_per_second": 1.0 / (sum(timings) / len(timings)),
    }


def paired_metric_analysis(method_a, method_b, metrics_map, metric_names, *, bootstrap_iterations=10000, seed=42):
    results = []
    metric_p_values = []
    for metric_name in metric_names:
        a_values = [float(metrics_map[method_a][q_id][metric_name]) for q_id in sorted(metrics_map[method_a])]
        b_values = [float(metrics_map[method_b][q_id][metric_name]) for q_id in sorted(metrics_map[method_b])]
        if len(a_values) != len(b_values):
            raise ValueError(f"Mismatched query count for {metric_name}: {len(a_values)} vs {len(b_values)}")
        bootstrap = paired_bootstrap_ci(a_values, b_values, n_bootstrap=bootstrap_iterations, confidence_level=0.95, seed=seed)
        if metric_name.startswith("hit@"):
            p_result = mcnemar_test([int(v > 0.5) for v in a_values], [int(v > 0.5) for v in b_values])
            p_value = float(p_result["p_value"])
            test_name = "mcnemar"
        else:
            tt = ttest_rel(np.asarray(a_values, dtype=float), np.asarray(b_values, dtype=float))
            p_value = float(tt.pvalue)
            test_name = "paired_t_test"
        metric_p_values.append((metric_name, p_value, test_name))
        results.append({
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric_name,
            "mean_difference": float(np.mean(np.asarray(a_values, dtype=float) - np.asarray(b_values, dtype=float))),
            "ci95": bootstrap["ci95"],
            "score_a_mean": float(np.mean(np.asarray(a_values, dtype=float))),
            "score_b_mean": float(np.mean(np.asarray(b_values, dtype=float))),
            "p_value": p_value,
            "test": test_name,
            "sample_size": len(a_values),
        })
    adjusted = holm_bonferroni_adjust([p for _, p, _ in metric_p_values])
    for item, adj in zip(results, adjusted):
        item["p_value_adjusted"] = float(adj)
    return results


def build_run_summary(per_query_metrics):
    summary = {}
    for method_name in sorted({row["method"] for row in per_query_metrics}):
        rows = [row for row in per_query_metrics if row["method"] == method_name]
        summary[method_name] = summarize_metrics(rows)
    return summary


def main():
    queries = load_dev_queries()
    qrels_by_query = load_qrels_by_query()
    run_dir = unique_run_dir(ROOT / "results" / "runs")
    artifact_manager = ArtifactManager(run_dir)

    bm25_start = time.perf_counter()
    bm25 = BM25Retriever(corpus_path=ROOT / "data" / "processed" / "corpus.jsonl", candidate_depth=100)
    bm25_build_time = time.perf_counter() - bm25_start

    dense_start = time.perf_counter()
    dense = Retriever(rebuild=False)
    dense_load_time = time.perf_counter() - dense_start

    def run_bm25_query_loop():
        rows = []
        for q in queries:
            qid = str(q["query_id"])
            hits = bm25.retrieve(q["claim_text"], top_k=20, query_id=qid)
            rows.append({"query_id": qid, "retrieved": [row["document_id"] for row in hits]})
        return rows

    def run_dense_query_loop():
        rows = []
        for q in queries:
            qid = str(q["query_id"])
            hits = dense.retrieve(q["claim_text"], top_k=20, query_id=qid)
            rows.append({"query_id": qid, "retrieved": [row["document_id"] for row in hits]})
        return rows

    # warm-up to isolate steady-state latency from model initialization and caching effects
    for _ in range(3):
        bm25.retrieve(queries[0]["claim_text"], top_k=20, query_id=queries[0]["query_id"])
        dense.retrieve(queries[0]["claim_text"], top_k=20, query_id=queries[0]["query_id"])

    bm25_results = run_bm25_query_loop()
    dense_results = run_dense_query_loop()

    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_constant=60, bm25_depth=100, dense_depth=100)

    def run_hybrid_query_loop():
        rows = []
        for q in queries:
            qid = str(q["query_id"])
            hits = hybrid.retrieve(q["claim_text"], top_k=20, query_id=qid, bm25_depth=100, dense_depth=100, rrf_constant=60)
            rows.append({"query_id": qid, "retrieved": [row["document_id"] for row in hits]})
        return rows

    for _ in range(3):
        hybrid.retrieve(queries[0]["claim_text"], top_k=20, query_id=queries[0]["query_id"], bm25_depth=100, dense_depth=100, rrf_constant=60)

    hybrid_results = run_hybrid_query_loop()

    bm25_metrics = metric_rows_for_method(bm25_results, "bm25", qrels_by_query)
    dense_metrics = metric_rows_for_method(dense_results, "dense", qrels_by_query)
    hybrid_metrics = metric_rows_for_method(hybrid_results, "hybrid", qrels_by_query)
    all_per_query_metrics = bm25_metrics + dense_metrics + hybrid_metrics

    fieldnames = ["method", "query_id", "retrieved_count"] + [f"hit@{k}" for k in K_VALUES] + [f"precision@{k}" for k in K_VALUES] + [f"recall@{k}" for k in K_VALUES] + [f"mrr@{k}" for k in K_VALUES] + [f"map@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES]
    with artifact_manager.raw_path("per_query_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_per_query_metrics:
            writer.writerow({key: row.get(key, 0.0) for key in fieldnames})

    summary = build_run_summary(all_per_query_metrics)
    artifact_manager.write_json("aggregate_metrics.json", summary)

    def profile_method(method_label, method_runner):
        for _ in range(3):
            method_runner(queries[0])
        tracemalloc.start()
        gc.collect()
        start = time.perf_counter()
        timings = []
        for q in queries:
            loop_start = time.perf_counter()
            method_runner(q)
            timings.append(time.perf_counter() - loop_start)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "mean_latency_seconds": sum(timings) / len(timings),
            "median_latency_seconds": median(timings),
            "p95_latency_seconds": sorted(timings)[max(0, min(len(timings) - 1, int(math.ceil(0.95 * len(timings))) - 1))],
            "qps": len(timings) / sum(timings),
            "peak_memory_mb": peak / (1024 * 1024),
            "total_query_time_seconds": elapsed,
            "measurements": timings,
        }

    bm25_latency = profile_method("bm25", lambda q: bm25.retrieve(q["claim_text"], top_k=20, query_id=str(q["query_id"])))
    dense_latency = profile_method("dense", lambda q: dense.retrieve(q["claim_text"], top_k=20, query_id=str(q["query_id"])))
    hybrid_latency = profile_method("hybrid", lambda q: hybrid.retrieve(q["claim_text"], top_k=20, query_id=str(q["query_id"]), bm25_depth=100, dense_depth=100, rrf_constant=60))

    dense_index_path = ROOT / "data" / "index" / "faiss_index_flat_ip.index"
    performance = {
        "bm25": {
            "index_build_time_seconds": bm25_build_time,
            "index_storage_size_bytes": 0,
            "mean_latency_seconds": bm25_latency["mean_latency_seconds"],
            "median_latency_seconds": bm25_latency["median_latency_seconds"],
            "p95_latency_seconds": bm25_latency["p95_latency_seconds"],
            "qps": bm25_latency["qps"],
            "peak_memory_mb": bm25_latency["peak_memory_mb"],
            "query_count": len(queries),
            "warmup_queries": 3,
        },
        "dense": {
            "index_build_time_seconds": dense_load_time,
            "index_storage_size_bytes": dense_index_path.stat().st_size if dense_index_path.exists() else 0,
            "mean_latency_seconds": dense_latency["mean_latency_seconds"],
            "median_latency_seconds": dense_latency["median_latency_seconds"],
            "p95_latency_seconds": dense_latency["p95_latency_seconds"],
            "qps": dense_latency["qps"],
            "peak_memory_mb": dense_latency["peak_memory_mb"],
            "query_count": len(queries),
            "warmup_queries": 3,
        },
        "hybrid": {
            "index_build_time_seconds": bm25_build_time + dense_load_time,
            "index_storage_size_bytes": 0,
            "mean_latency_seconds": hybrid_latency["mean_latency_seconds"],
            "median_latency_seconds": hybrid_latency["median_latency_seconds"],
            "p95_latency_seconds": hybrid_latency["p95_latency_seconds"],
            "qps": hybrid_latency["qps"],
            "peak_memory_mb": hybrid_latency["peak_memory_mb"],
            "query_count": len(queries),
            "warmup_queries": 3,
        },
    }
    artifact_manager.write_json("performance_metrics.json", performance)

    metadata = {
        "run_id": run_dir.name,
        "dataset": "BEIR SciFact / official SciFact dev claims",
        "split": "official_dev_for_tuning",
        "query_count": len(queries),
        "corpus_count": 5183,
        "k_values": K_VALUES,
        "candidate_depths": {"bm25": 100, "dense": 100, "hybrid": {"bm25": 100, "dense": 100}},
        "rrf_constant": 60,
        "retrievers": {
            "bm25": {"k1": 1.5, "b": 0.75, "candidate_depth": 100, "tokenization": "lowercase, simple regex"},
            "dense": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "index": "faiss.IndexFlatIP"},
            "hybrid": {"rrf_constant": 60, "candidate_depths": {"bm25": 100, "dense": 100}},
        },
        "qrels_source": "BeIR/scifact-qrels test split",
        "raw_claims_source": str(RAW_DEV_CLAIMS_PATH),
        "run_dir": str(run_dir),
    }
    artifact_manager.write_json("metadata.json", metadata)

    sensitivity_rows = []
    for c_value in SENSITIVITY_CONSTANTS:
        for k in SENSITIVITY_K_VALUES:
            per_method = []
            hybrid_candidate = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_constant=c_value, bm25_depth=100, dense_depth=100)
            for q in queries:
                qid = str(q["query_id"])
                hits = hybrid_candidate.retrieve(q["claim_text"], top_k=k, query_id=qid, bm25_depth=100, dense_depth=100, rrf_constant=c_value)
                per_method.append({"query_id": qid, "retrieved": [row["document_id"] for row in hits]})
            metrics = metric_rows_for_method(per_method, "hybrid", qrels_by_query)
            row = {"rrf_constant": c_value, "k": k}
            for metric_name in [
                *[f"hit@{k}" for k in [k]],
                *[f"precision@{k}" for k in [k]],
                *[f"recall@{k}" for k in [k]],
                *[f"mrr@{k}" for k in [k]],
                *[f"map@{k}" for k in [k]],
                *[f"ndcg@{k}" for k in [k]],
            ]:
                row[metric_name] = float(sum(m.get(metric_name, 0.0) for m in metrics) / len(metrics)) if metrics else 0.0
            sensitivity_rows.append(row)
    with artifact_manager.raw_path("rrf_sensitivity.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["rrf_constant", "k", *[f"hit@{k}" for k in [5, 10, 20]], *[f"precision@{k}" for k in [5, 10, 20]], *[f"recall@{k}" for k in [5, 10, 20]], *[f"mrr@{k}" for k in [5, 10, 20]], *[f"map@{k}" for k in [5, 10, 20]], *[f"ndcg@{k}" for k in [5, 10, 20]]]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sensitivity_rows:
            writer.writerow({key: row.get(key, 0.0) for key in fieldnames})

    metrics_map = {
        "bm25": {row["query_id"]: {k: float(row.get(k, 0.0)) for k in [f"hit@{k}" for k in K_VALUES] + [f"precision@{k}" for k in K_VALUES] + [f"recall@{k}" for k in K_VALUES] + [f"mrr@{k}" for k in K_VALUES] + [f"map@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES]} for row in bm25_metrics},
        "dense": {row["query_id"]: {k: float(row.get(k, 0.0)) for k in [f"hit@{k}" for k in K_VALUES] + [f"precision@{k}" for k in K_VALUES] + [f"recall@{k}" for k in K_VALUES] + [f"mrr@{k}" for k in K_VALUES] + [f"map@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES]} for row in dense_metrics},
        "hybrid": {row["query_id"]: {k: float(row.get(k, 0.0)) for k in [f"hit@{k}" for k in K_VALUES] + [f"precision@{k}" for k in K_VALUES] + [f"recall@{k}" for k in K_VALUES] + [f"mrr@{k}" for k in K_VALUES] + [f"map@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES]} for row in hybrid_metrics},
    }
    comparisons = []
    for pair in [("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid")]:
        metric_names = ["hit@10", "precision@10", "recall@10", "mrr@10", "map@10", "ndcg@10"]
        comparisons.extend(paired_metric_analysis(pair[0], pair[1], metrics_map, metric_names))
    artifact_manager.write_json("statistical_tests.json", {"pairs": comparisons, "sample_size": len(queries), "method": "paired bootstrap + paired t-test / McNemar", "correction": "Holm-Bonferroni"})

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def save_metric_barplot(metric_key, out_name):
        labels = ["bm25", "dense", "hybrid"]
        values = [summary[m][metric_key] for m in labels]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color=["steelblue", "darkorange", "forestgreen"])
        ax.set_title(f"{metric_key.upper()} by retriever")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        for label, value in zip(labels, values):
            ax.text(label, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout(); fig.savefig(figures_dir / out_name, dpi=200); plt.close(fig)

    for key in ["hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "mrr@10", "ndcg@10"]:
        save_metric_barplot(key, f"{key.replace('@', '_')}_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["bm25", "dense", "hybrid"], [performance[m]["mean_latency_seconds"] for m in ["bm25", "dense", "hybrid"]], color=["steelblue", "darkorange", "forestgreen"])
    ax.set_title("Mean query latency by retriever")
    ax.set_ylabel("Seconds")
    fig.tight_layout(); fig.savefig(figures_dir / "latency_comparison.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    c_values = SENSITIVITY_CONSTANTS
    hit_values = {}
    for row in sensitivity_rows:
        if row["k"] == 10:
            hit_values[int(row["rrf_constant"])] = float(row["hit@10"])
    ax.plot(c_values, [hit_values[c] for c in c_values], marker="o")
    ax.set_title("RRF sensitivity: Hit@10")
    ax.set_xlabel("RRF constant c")
    ax.set_ylabel("Hit@10")
    ax.set_xticks(c_values)
    fig.tight_layout(); fig.savefig(figures_dir / "rrf_sensitivity.png", dpi=200); plt.close(fig)

    print(f"run_dir={run_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
