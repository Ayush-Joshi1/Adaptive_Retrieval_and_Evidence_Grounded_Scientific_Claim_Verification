from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEV_CLAIMS_PATH = ROOT / "data" / "processed" / "raw_scifact" / "data" / "claims_dev.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
RUN_ROOT = ROOT / "results" / "runs"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOLS = (10, 20, 50)
CUTOFFS = (1, 3, 5, 10)
BOOTSTRAP_ITERATIONS = 10_000
SEED = 42
BATCH_SIZE = 16
TORCH_THREADS = 2
RRF_CONSTANT = 60
RETRIEVAL_DEPTH = 100
SMOKE_QUERY_COUNT = 3
PRACTICAL_DIFFERENCE_THRESHOLD = 0.01
ACTIVE_RUN_DIR: Path | None = None
RUN_LOG: Any = None

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_manager import unique_run_dir


def _read_dev_queries() -> list[dict[str, str]]:
    if not DEV_CLAIMS_PATH.exists():
        raise FileNotFoundError(f"Official development claims were not found: {DEV_CLAIMS_PATH}")
    rows = []
    with DEV_CLAIMS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append({"query_id": str(record["id"]), "claim_text": str(record.get("claim") or "")})
    query_ids = [row["query_id"] for row in rows]
    if len(rows) != 300 or len(set(query_ids)) != 300:
        raise ValueError(f"Expected 300 unique official dev queries; found {len(rows)} rows and {len(set(query_ids))} IDs")
    return rows


def _load_dev_qrels(query_ids: set[str]) -> dict[str, dict[str, float]]:
    from datasets import load_dataset

    qrels_split = load_dataset("BeIR/scifact-qrels", split="test")
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in qrels_split:
        query_id = str(row["query-id"])
        if query_id not in query_ids:
            raise ValueError(f"BEIR qrels contain a query outside official SciFact dev: {query_id}")
        qrels[query_id][str(row["corpus-id"])] = float(row["score"])
    if set(qrels) != query_ids:
        missing = sorted(query_ids - set(qrels))
        raise ValueError(f"Dev qrels are missing {len(missing)} of the 300 official dev query IDs")
    return dict(qrels)


def _sha256_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        raise FileNotFoundError(f"Protected canonical run is missing: {root}")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _memory_snapshot() -> dict[str, int | str | None]:
    if os.name != "nt":
        return {"status": "unavailable", "peak_working_set_bytes": None, "working_set_bytes": None}

    import ctypes
    from ctypes import wintypes

    size_t = ctypes.c_size_t

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", size_t),
            ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t),
            ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t),
            ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t),
            ("PeakPagefileUsage", size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process_handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    succeeded = get_process_memory_info(process_handle, ctypes.byref(counters), counters.cb)
    if not succeeded:
        return {"status": "unavailable", "peak_working_set_bytes": None, "working_set_bytes": None}
    return {
        "status": "measured via Windows GetProcessMemoryInfo",
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "working_set_bytes": int(counters.WorkingSetSize),
        "pagefile_usage_bytes": int(counters.PagefileUsage),
    }


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hydrate_candidates(candidates: list[dict[str, Any]], corpus_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hydrated = []
    for candidate in candidates:
        document_id = str(candidate.get("document_id") or candidate.get("corpus_id") or "")
        document = corpus_by_id.get(document_id)
        if document is None:
            raise KeyError(f"Retrieved document is absent from the shared corpus: {document_id}")
        hydrated.append({**candidate, "document_id": document_id, "corpus_id": document_id, "title": document.get("title"), "text": document.get("text")})
    return hydrated


def _validate_ranking(candidates: list[dict[str, Any]], ranked: list[dict[str, Any]], query_id: str) -> None:
    expected_ids = [str(candidate.get("document_id") or candidate.get("corpus_id")) for candidate in candidates]
    actual_ids = [str(candidate["document_id"]) for candidate in ranked]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise AssertionError(f"Cross-encoder changed candidate IDs for query {query_id}")
    if any(int(item["rank"]) != position for position, item in enumerate(ranked, start=1)):
        raise AssertionError(f"Cross-encoder returned invalid ranks for query {query_id}")
    if any(float(ranked[index]["cross_encoder_score"]) < float(ranked[index + 1]["cross_encoder_score"]) for index in range(len(ranked) - 1)):
        raise AssertionError(f"Cross-encoder scores are not in descending rank order for query {query_id}")
    if any(str(item.get("query_id")) != query_id for item in ranked):
        raise AssertionError(f"Cross-encoder changed query IDs for query {query_id}")


def _retrieved_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["document_id"]) for row in rows]


def _ranked_subset(scored_candidates: list[dict[str, Any]], pool_size: int) -> list[dict[str, Any]]:
    subset = [row for row in scored_candidates if int(row["original_rank"]) <= pool_size]
    subset.sort(key=lambda row: (-float(row["cross_encoder_score"]), int(row["original_rank"])))
    return [{**row, "rank": rank} for rank, row in enumerate(subset, start=1)]


def _metric_names() -> list[str]:
    return [
        f"{metric}@{cutoff}"
        for cutoff in CUTOFFS
        for metric in ("hit", "precision", "recall", "mrr", "map", "ndcg")
    ]


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(__import__("math").ceil(0.95 * len(ordered))) - 1))
    total = sum(values)
    return {
        "mean_seconds": mean(values),
        "median_seconds": median(values),
        "p95_seconds": ordered[p95_index],
        "throughput_queries_per_second": len(values) / total if total else 0.0,
        "total_seconds": total,
        "query_count": len(values),
    }


def _ranked_record(row: dict[str, Any]) -> dict[str, Any]:
    record = {
        "document_id": str(row["document_id"]),
        "rank": int(row["rank"]),
        "original_rank": int(row.get("original_rank", row.get("rank", 0))),
        "retrieval_score": float(row.get("retrieval_score", row.get("score", 0.0)) or 0.0),
    }
    if "cross_encoder_score" in row:
        record["cross_encoder_score"] = float(row["cross_encoder_score"])
    return record


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _log(message: str) -> None:
    print(message, flush=True)
    if RUN_LOG is not None:
        RUN_LOG.write(message + "\n")
        RUN_LOG.flush()


def _valid_query_checkpoint(path: Path, query_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    expected_configurations = {
        f"{method}_pool_{pool}_{condition}"
        for method in ("dense", "hybrid")
        for pool in CANDIDATE_POOLS
        for condition in ("baseline", "reranked")
    }
    if checkpoint.get("query_id") != query_id:
        raise ValueError(f"Checkpoint query ID mismatch in {path}")
    for field in ("metrics", "rankings", "performance"):
        rows = checkpoint.get(field)
        if not isinstance(rows, list) or len(rows) != len(expected_configurations):
            raise ValueError(f"Incomplete {field} checkpoint for query {query_id}")
        keys = {
            f"{row.get('base_method')}_pool_{row.get('candidate_pool_size')}_{row.get('condition')}"
            for row in rows
        }
        if keys != expected_configurations:
            raise ValueError(f"Invalid or duplicate {field} configurations for query {query_id}")
    ranking_lookup = {
        (row["base_method"], row["candidate_pool_size"], row["condition"]): row["rankings"]
        for row in checkpoint["rankings"]
    }
    for method in ("dense", "hybrid"):
        for pool_size in CANDIDATE_POOLS:
            baseline_ids = [
                row["document_id"]
                for row in ranking_lookup[(method, pool_size, "baseline")]
            ]
            reranked = ranking_lookup[(method, pool_size, "reranked")]
            reranked_ids = [row["document_id"] for row in reranked]
            if len(baseline_ids) != len(set(baseline_ids)) or set(baseline_ids) != set(reranked_ids):
                raise ValueError(f"Checkpoint candidate IDs differ for query {query_id}, {method}, pool {pool_size}")
    return checkpoint


def _smoke_test(
    queries: list[dict[str, str]],
    dense: Any,
    hybrid: Any,
    reranker: Any,
    corpus_by_id: dict[str, dict[str, Any]],
    memory: dict[str, Any],
) -> dict[str, Any]:
    smoke_rows = []
    query_count = min(SMOKE_QUERY_COUNT, len(queries))
    for query in queries[:query_count]:
        for method, retriever in (("dense", dense), ("hybrid", hybrid)):
            query_id, claim = query["query_id"], query["claim_text"]
            retrieve_start = time.perf_counter()
            candidates = retriever.retrieve(claim, top_k=max(CANDIDATE_POOLS), query_id=query_id)
            retrieval_seconds = time.perf_counter() - retrieve_start
            hydration_start = time.perf_counter()
            if method == "hybrid":
                candidates = _hydrate_candidates(candidates, corpus_by_id)
            hydration_seconds = time.perf_counter() - hydration_start
            rerank_start = time.perf_counter()
            ranked = reranker.rerank(claim, candidates, top_k=max(CANDIDATE_POOLS), query_id=query_id)
            rerank_seconds = time.perf_counter() - rerank_start
            _validate_ranking(candidates, ranked, query_id)
            if method == "hybrid" and any(not (row.get("text") or row.get("title")) for row in candidates):
                raise AssertionError(f"Hybrid candidates were not hydrated with document text for query {query_id}")
            smoke_rows.append({
                "query_id": query_id,
                "method": method,
                "candidate_count": len(candidates),
                "retrieval_seconds": retrieval_seconds,
                "candidate_hydration_seconds": hydration_seconds,
                "reranking_seconds_pool_50": rerank_seconds,
                "cross_encoder_inference_seconds_pool_50": reranker.last_inference_seconds,
                "ranking_valid": True,
            })
    rerank_times = [row["reranking_seconds_pool_50"] for row in smoke_rows]
    per_query_system_mean = mean(rerank_times) if rerank_times else 0.0
    retrieval_mean = mean(row["retrieval_seconds"] for row in smoke_rows) if smoke_rows else 0.0
    hydration_mean = mean(row["candidate_hydration_seconds"] for row in smoke_rows) if smoke_rows else 0.0
    estimated_full_reranking_seconds = per_query_system_mean * len(queries) * 2
    estimated_full_end_to_end_seconds = (
        (per_query_system_mean + retrieval_mean + hydration_mean) * len(queries) * 2
    )
    return {
        "status": "passed",
        "purpose": "technical smoke test only; no smoke metrics are included in final evaluation",
        "query_count": query_count,
        "system_query_count": len(smoke_rows),
        "model_name": MODEL_NAME,
        "device": "cpu",
        "batch_size": BATCH_SIZE,
        "model_load_seconds": reranker.model_load_seconds,
        "torch_threads": TORCH_THREADS,
        "candidate_scoring": "50 candidates per query/system, batched by the cross-encoder",
        "ranking_validation": "candidate IDs preserved, scores descending, contiguous ranks, query IDs preserved",
        "measurements": smoke_rows,
        "mean_pool_50_reranking_seconds_per_query_system": per_query_system_mean,
        "mean_pool_50_inference_seconds_per_query_system": mean(
            row["cross_encoder_inference_seconds_pool_50"] for row in smoke_rows
        ) if smoke_rows else 0.0,
        "mean_retrieval_seconds_per_query_system": retrieval_mean,
        "estimated_full_end_to_end_minutes_with_score_reuse": estimated_full_end_to_end_seconds / 60.0,
        "estimated_full_run_reranking_minutes_with_50_pool_score_reuse": estimated_full_reranking_seconds / 60.0,
        "peak_memory_after_smoke": memory(),
    }


def _evaluate(
    run_dir: Path,
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, float]],
    dense: Any,
    hybrid: Any,
    reranker: Any,
    corpus_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.evaluation.retrieval_metrics import compute_retrieval_metrics

    metric_fields = _metric_names()
    checkpoint_dir = run_dir / "checkpoints" / "queries"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    metrics_path = run_dir / "per_query_metrics.csv"
    rankings_path = run_dir / "per_query_rankings.jsonl"
    performance_path = run_dir / "query_performance.csv"
    for query_index, query in enumerate(queries, start=1):
        query_id, claim = query["query_id"], query["claim_text"]
        checkpoint_path = checkpoint_dir / f"{query_id}.json"
        checkpoint = _valid_query_checkpoint(checkpoint_path, query_id)
        if checkpoint is None:
            query_qrels = qrels.get(query_id, {})
            metric_rows: list[dict[str, Any]] = []
            ranking_rows: list[dict[str, Any]] = []
            performance_rows: list[dict[str, Any]] = []
            for method, retriever in (("dense", dense), ("hybrid", hybrid)):
                retrieval_start = time.perf_counter()
                candidates = retriever.retrieve(claim, top_k=max(CANDIDATE_POOLS), query_id=query_id)
                retrieval_seconds = time.perf_counter() - retrieval_start
                if len(candidates) != len(_retrieved_ids(candidates)):
                    raise AssertionError(f"{method} returned duplicate candidates for query {query_id}")

                hydration_start = time.perf_counter()
                model_candidates = _hydrate_candidates(candidates, corpus_by_id) if method == "hybrid" else candidates
                hydration_seconds = time.perf_counter() - hydration_start
                rerank_start = time.perf_counter()
                scored = reranker.rerank(
                    claim, model_candidates, top_k=max(CANDIDATE_POOLS), query_id=query_id
                )
                reranking_seconds = time.perf_counter() - rerank_start
                inference_seconds = float(reranker.last_inference_seconds)
                _validate_ranking(model_candidates, scored, query_id)

                for pool_size in CANDIDATE_POOLS:
                    baseline = candidates[:pool_size]
                    postprocess_start = time.perf_counter()
                    reranked = _ranked_subset(scored, pool_size)
                    postprocess_seconds = time.perf_counter() - postprocess_start
                    if len(reranked) != len(baseline):
                        raise AssertionError(f"Candidate pool {pool_size} mismatch for {method}, query {query_id}")
                    if set(_retrieved_ids(reranked)) != set(_retrieved_ids(baseline)):
                        raise AssertionError(f"Reranking changed candidate IDs for {method}, query {query_id}")
                    for condition, ranked in (("baseline", baseline), ("reranked", reranked)):
                        metrics = compute_retrieval_metrics(_retrieved_ids(ranked), query_qrels, CUTOFFS)
                        metric_rows.append({
                            "query_id": query_id,
                            "base_method": method,
                            "candidate_pool_size": pool_size,
                            "condition": condition,
                            **metrics,
                        })
                        ranking_rows.append({
                            "query_id": query_id,
                            "base_method": method,
                            "candidate_pool_size": pool_size,
                            "condition": condition,
                            "rankings": [_ranked_record(row) for row in ranked],
                        })
                        condition_inference = inference_seconds if condition == "reranked" else 0.0
                        condition_reranking = reranking_seconds if condition == "reranked" else 0.0
                        condition_hydration = hydration_seconds if condition == "reranked" else 0.0
                        condition_postprocess = postprocess_seconds if condition == "reranked" else 0.0
                        end_to_end = (
                            retrieval_seconds
                            + condition_hydration
                            + condition_reranking
                            + condition_postprocess
                        )
                        performance_rows.append({
                            "query_id": query_id,
                            "base_method": method,
                            "candidate_pool_size": pool_size,
                            "condition": condition,
                            "retrieval_latency_seconds": retrieval_seconds,
                            "candidate_hydration_latency_seconds": condition_hydration,
                            "cross_encoder_inference_seconds": condition_inference,
                            "reranking_latency_seconds": condition_reranking,
                            "ranking_postprocess_latency_seconds": condition_postprocess,
                            "end_to_end_latency_seconds": end_to_end,
                            "scored_pool_size": len(candidates) if condition == "reranked" else 0,
                            "scores_reused_across_pool_sizes": condition == "reranked",
                        })
            checkpoint = {
                "query_id": query_id,
                "metrics": metric_rows,
                "rankings": ranking_rows,
                "performance": performance_rows,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write_json(checkpoint_path, checkpoint)
        checkpoints[query_id] = checkpoint
        if query_index % 10 == 0 or query_index == len(queries):
            elapsed = time.perf_counter() - started
            _log(
                f"evaluated={query_index}/{len(queries)} "
                f"elapsed_minutes={elapsed / 60:.1f} "
                f"estimated_remaining_minutes={(elapsed / query_index) * (len(queries) - query_index) / 60:.1f}"
            )

    metric_rows = [row for query in queries for row in checkpoints[query["query_id"]]["metrics"]]
    ranking_rows = [row for query in queries for row in checkpoints[query["query_id"]]["rankings"]]
    performance_rows = [row for query in queries for row in checkpoints[query["query_id"]]["performance"]]
    _atomic_write_json(run_dir / "per_query_metrics.json", metric_rows)
    _atomic_write_json(run_dir / "per_query_rankings.json", ranking_rows)
    metric_columns = ["query_id", "base_method", "candidate_pool_size", "condition", *metric_fields]
    performance_columns = [
        "query_id", "base_method", "candidate_pool_size", "condition",
        "retrieval_latency_seconds", "candidate_hydration_latency_seconds",
        "cross_encoder_inference_seconds", "reranking_latency_seconds",
        "ranking_postprocess_latency_seconds", "end_to_end_latency_seconds",
        "scored_pool_size", "scores_reused_across_pool_sizes",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows(metric_rows)
    with rankings_path.open("w", encoding="utf-8") as rankings_file:
        for row in ranking_rows:
            rankings_file.write(json.dumps(row, sort_keys=True) + "\n")
    with performance_path.open("w", encoding="utf-8", newline="") as performance_file:
        writer = csv.DictWriter(performance_file, fieldnames=performance_columns)
        writer.writeheader()
        writer.writerows(performance_rows)

    metric_lookup: dict[tuple[str, int, str, str], dict[str, float]] = {}
    latency_lookup: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        key = (row["base_method"], row["candidate_pool_size"], row["condition"], row["query_id"])
        metric_lookup[key] = {metric: float(row[metric]) for metric in metric_fields}
    for row in performance_rows:
        latency_lookup[(row["base_method"], row["candidate_pool_size"], row["condition"])].append(
            float(row["end_to_end_latency_seconds"])
        )
    aggregates: dict[str, Any] = {}
    for method in ("dense", "hybrid"):
        for pool_size in CANDIDATE_POOLS:
            for condition in ("baseline", "reranked"):
                rows = [
                    metric_lookup[(method, pool_size, condition, query["query_id"])]
                    for query in queries
                ]
                aggregates[f"{method}_pool_{pool_size}_{condition}"] = {
                    metric: mean(float(row[metric]) for row in rows)
                    for metric in metric_fields
                }

    performance = {}
    for method in ("dense", "hybrid"):
        for pool_size in CANDIDATE_POOLS:
            for condition in ("baseline", "reranked"):
                key = (method, pool_size, condition)
                performance[f"{method}_pool_{pool_size}_{condition}"] = _latency_summary(latency_lookup[key])

    elapsed_seconds = time.perf_counter() - started
    performance["evaluation_wall_time_seconds"] = elapsed_seconds
    performance["query_count"] = len(queries)
    performance["latency_methodology"] = {
        "retrieval": "one retrieval at depth 50 per query and base system; exact prefixes are cached and reused for all candidate pools",
        "cross_encoder_inference": "model predict time for one batched 50-candidate scoring pass per query and base system",
        "reranking": "end-to-end reranker call time, including preprocessing, model inference, sorting, and rank assignment",
        "end_to_end": "retrieval plus (for reranked conditions) candidate hydration and the measured 50-candidate reranker pass",
        "pool_specific_latency_caveat": "The 10- and 20-candidate reranked outputs reuse the same 50-candidate score pass; timings are not separate smaller-pool inference runs.",
    }
    performance["retrieval_latency_summary"] = _latency_summary([
        float(row["retrieval_latency_seconds"])
        for row in performance_rows
        if row["condition"] == "baseline" and row["candidate_pool_size"] == max(CANDIDATE_POOLS)
    ])
    performance["cross_encoder_inference_latency_summary"] = _latency_summary([
        float(row["cross_encoder_inference_seconds"])
        for row in performance_rows
        if row["condition"] == "reranked" and row["candidate_pool_size"] == max(CANDIDATE_POOLS)
    ])
    performance["reranking_latency_summary"] = _latency_summary([
        float(row["reranking_latency_seconds"])
        for row in performance_rows
        if row["condition"] == "reranked" and row["candidate_pool_size"] == max(CANDIDATE_POOLS)
    ])
    return aggregates, performance


def _paired_statistical_analysis(
    queries: list[dict[str, str]],
    aggregate_rows_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import binomtest, wilcoxon
    from src.evaluation.statistical_tests import holm_bonferroni_adjust, paired_bootstrap_ci

    with aggregate_rows_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        metric_lookup = {
            (row["base_method"], int(row["candidate_pool_size"]), row["condition"], row["query_id"]): {
                name: float(row[name]) for name in _metric_names()
            }
            for row in reader
        }

    comparisons = []
    for method in ("dense", "hybrid"):
        for pool_size in CANDIDATE_POOLS:
            for metric in _metric_names():
                baseline_scores = np.asarray([
                    metric_lookup[(method, pool_size, "baseline", query["query_id"])][metric]
                    for query in queries
                ], dtype=float)
                reranked_scores = np.asarray([
                    metric_lookup[(method, pool_size, "reranked", query["query_id"])][metric]
                    for query in queries
                ], dtype=float)
                differences = reranked_scores - baseline_scores
                bootstrap = paired_bootstrap_ci(
                    reranked_scores.tolist(),
                    baseline_scores.tolist(),
                    n_bootstrap=BOOTSTRAP_ITERATIONS,
                    confidence_level=0.95,
                    seed=SEED,
                )
                if metric.startswith("hit@"):
                    baseline_hit = baseline_scores > 0.5
                    reranked_hit = reranked_scores > 0.5
                    a_b = int((baseline_hit & ~reranked_hit).sum())
                    b_a = int((~baseline_hit & reranked_hit).sum())
                    discordant = a_b + b_a
                    p_value = float(binomtest(b_a, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
                    test_name = "exact McNemar test"
                elif np.all(differences == 0):
                    p_value = 1.0
                    test_name = "Wilcoxon signed-rank test"
                else:
                    p_value = float(wilcoxon(reranked_scores, baseline_scores, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
                    test_name = "Wilcoxon signed-rank test"
                difference_mean = float(differences.mean())
                difference_std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
                practical = abs(difference_mean) >= PRACTICAL_DIFFERENCE_THRESHOLD
                direction = "improvement" if difference_mean > 0 else "decline" if difference_mean < 0 else "no change"
                comparisons.append({
                    "base_method": method,
                    "comparison": f"{method} vs {method}_reranked",
                    "candidate_pool_size": pool_size,
                    "metric": metric,
                    "baseline_mean": float(baseline_scores.mean()),
                    "reranked_mean": float(reranked_scores.mean()),
                    "mean_difference_reranked_minus_baseline": difference_mean,
                    "paired_cohens_dz": difference_mean / difference_std if difference_std else None,
                    "bootstrap_ci95": bootstrap["ci95"],
                    "bootstrap_resamples": BOOTSTRAP_ITERATIONS,
                    "test": test_name,
                    "p_value": p_value,
                    "sample_size": len(queries),
                    "baseline_only_hits": a_b if metric.startswith("hit@") else None,
                    "reranked_only_hits": b_a if metric.startswith("hit@") else None,
                    "practical_difference_threshold": PRACTICAL_DIFFERENCE_THRESHOLD,
                    "practically_significant": practical,
                    "direction": direction,
                })
    adjusted = holm_bonferroni_adjust([item["p_value"] for item in comparisons])
    for item, adjusted_p in zip(comparisons, adjusted):
        item["p_value_holm_adjusted"] = float(adjusted_p)
        item["statistically_significant_after_holm"] = adjusted_p < 0.05
        if item["statistically_significant_after_holm"] and item["practically_significant"]:
            item["interpretation"] = f"statistically and practically significant {item['direction']}"
        elif item["statistically_significant_after_holm"]:
            item["interpretation"] = f"statistically significant but below the practical threshold ({item['direction']})"
        elif item["practically_significant"]:
            item["interpretation"] = f"practical-sized {item['direction']} without corrected statistical significance"
        else:
            item["interpretation"] = "no statistically or practically significant difference"
    return {
        "pairs": comparisons,
        "sample_size": len(queries),
        "bootstrap_resamples_per_metric": BOOTSTRAP_ITERATIONS,
        "confidence_level": 0.95,
        "paired_tests": {
            "hit": "exact two-sided McNemar test",
            "other_metrics": "two-sided Wilcoxon signed-rank test",
        },
        "holm_bonferroni_family": "all metrics, cutoffs, candidate pools, and both base systems (144 tests)",
        "correction": "Holm-Bonferroni",
        "practical_significance": {
            "absolute_mean_difference_threshold": PRACTICAL_DIFFERENCE_THRESHOLD,
            "definition": "at least a 0.01 absolute mean metric difference",
        },
    }


def _generate_plots(run_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run_dir / "aggregate_metrics.json").open("r", encoding="utf-8") as handle:
        aggregates = json.load(handle)
    with (run_dir / "performance_metrics.json").open("r", encoding="utf-8") as handle:
        performance = json.load(handle)
    with (run_dir / "statistical_tests.json").open("r", encoding="utf-8") as handle:
        statistics = json.load(handle)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for method, title in (("dense", "Dense retrieval"), ("hybrid", "Hybrid retrieval")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, metric in zip(axes, ("hit", "ndcg")):
            for pool_size in CANDIDATE_POOLS:
                for condition, style in (("baseline", "-o"), ("reranked", "--o")):
                    values = [
                        aggregates[f"{method}_pool_{pool_size}_{condition}"][f"{metric}@{cutoff}"]
                        for cutoff in CUTOFFS
                    ]
                    axis.plot(CUTOFFS, values, style, label=f"pool {pool_size} {condition}")
            axis.set_title(f"{metric.upper()}@K")
            axis.set_xlabel("Evaluation cutoff K")
            axis.set_xticks(CUTOFFS)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Mean development-set score")
        axes[1].legend(fontsize=7, loc="best")
        fig.suptitle(f"{title}: baseline vs cross-encoder reranking")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{method}_vs_reranking.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ("hit@10", "ndcg@10")):
        for method, condition, label in (
            ("dense", "baseline", "Dense"),
            ("dense", "reranked", "Dense + reranking"),
            ("hybrid", "baseline", "Hybrid"),
            ("hybrid", "reranked", "Hybrid + reranking"),
        ):
            axis.plot(
                CANDIDATE_POOLS,
                [aggregates[f"{method}_pool_{pool}_{condition}"][metric] for pool in CANDIDATE_POOLS],
                marker="o",
                label=label,
            )
        axis.set_title(metric)
        axis.set_xlabel("Candidate pool size")
        axis.set_xticks(CANDIDATE_POOLS)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean development-set score")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Candidate pool size comparison")
    fig.tight_layout()
    fig.savefig(figures_dir / "candidate_pool_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method, condition, label, marker in (
        ("dense", "baseline", "Dense", "o"),
        ("dense", "reranked", "Dense + reranking", "o"),
        ("hybrid", "baseline", "Hybrid", "s"),
        ("hybrid", "reranked", "Hybrid + reranking", "s"),
    ):
        for pool_size in CANDIDATE_POOLS:
            key = f"{method}_pool_{pool_size}_{condition}"
            axis.scatter(
                performance[key]["mean_seconds"],
                aggregates[key]["ndcg@10"],
                marker=marker,
                label=f"{label}, pool {pool_size}",
            )
    axis.set_xlabel("Mean end-to-end latency (seconds/query)")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Retrieval quality vs end-to-end CPU latency")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_vs_latency.png", dpi=180)
    plt.close(fig)

    selected = [item for item in statistics["pairs"] if item["metric"] == "ndcg@10"]
    labels = [f"{item['base_method']} pool {item['candidate_pool_size']}" for item in selected]
    differences = [item["mean_difference_reranked_minus_baseline"] for item in selected]
    lower = [item["bootstrap_ci95"][0] for item in selected]
    upper = [item["bootstrap_ci95"][1] for item in selected]
    positions = list(range(len(selected)))
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(
        differences,
        positions,
        xerr=[
            [difference - low for difference, low in zip(differences, lower)],
            [high - difference for difference, high in zip(differences, upper)],
        ],
        fmt="o",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("nDCG@10 mean difference (reranked - baseline)")
    axis.set_title("Paired bootstrap 95% confidence intervals")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "statistical_confidence_intervals.png", dpi=180)
    plt.close(fig)
    return [
        "dense_vs_reranking.png",
        "hybrid_vs_reranking.png",
        "candidate_pool_comparison.png",
        "quality_vs_latency.png",
        "statistical_confidence_intervals.png",
    ]


def _run_focused_tests(run_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--ignore=tests/test_dataset_validation.py",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    validation = {
        "command": command,
        "exit_code": result.returncode,
        "status": "passed" if result.returncode == 0 else "failed",
        "test_pass_count": int(result.stdout.split(" passed", 1)[0].split()[-1]) if " passed" in result.stdout else None,
        "suite_scope": "All repository tests except tests/test_dataset_validation.py",
        "excluded_test_file": "tests/test_dataset_validation.py",
        "exclusion_reason": "That test loads official claims_test.jsonl, which this experiment is prohibited from accessing.",
        "official_test_split_accessed": False,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (run_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    return validation


def _validate_phase2_baselines(metrics_path: Path) -> dict[str, Any]:
    phase2_path = RUN_ROOT / "run_0003" / "per_query_metrics.csv"
    expected: dict[tuple[str, str], dict[str, float]] = {}
    with phase2_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in {"dense", "hybrid"}:
                expected[(row["method"], row["query_id"])] = {
                    metric: float(row[metric]) for metric in _metric_names()
                }

    actual: dict[tuple[str, str], dict[str, float]] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["base_method"] in {"dense", "hybrid"}
                and int(row["candidate_pool_size"]) == 20
                and row["condition"] == "baseline"
            ):
                actual[(row["base_method"], row["query_id"])] = {
                    metric: float(row[metric]) for metric in _metric_names()
                }
    if len(expected) != 600 or actual.keys() != expected.keys():
        raise AssertionError(
            f"Phase 2 baseline query alignment failed: expected {len(expected)} method-query rows, found {len(actual)}"
        )
    max_abs_difference = 0.0
    for key, expected_metrics in expected.items():
        for metric, expected_value in expected_metrics.items():
            difference = abs(actual[key][metric] - expected_value)
            max_abs_difference = max(max_abs_difference, difference)
            if difference > 1e-12:
                raise AssertionError(
                    f"Phase 2 baseline mismatch for {key}, {metric}: "
                    f"{actual[key][metric]} vs {expected_value}"
                )
    return {
        "status": "passed",
        "phase2_run": "results/runs/run_0003",
        "comparison": "dense and hybrid per-query baseline metrics at pool 20",
        "query_count_per_method": 300,
        "metric_count": len(_metric_names()),
        "max_absolute_difference": max_abs_difference,
    }


def _write_experiment_report(
    run_dir: Path,
    aggregates: dict[str, Any],
    performance: dict[str, Any],
    statistics: dict[str, Any],
    smoke: dict[str, Any],
    validation: dict[str, Any],
    runtime_seconds: float,
) -> Path:
    report_path = ROOT / "docs" / "PHASE3_REPORT.md"
    metric_names = _metric_names()
    lines = [
        "# Phase 3.2 — Experiment A: Cross-Encoder Reranking",
        "",
        "> This report covers Experiment A only. Phase 3 is not complete; Experiments B and C have not been run.",
        "",
        "## Protocol and integrity",
        "",
        f"- Run: `{run_dir.name}` (`{run_dir.relative_to(ROOT).as_posix()}/`).",
        "- Dataset and split: all 300 claims from the official SciFact development split (`claims_dev.jsonl`); every configuration uses identical query IDs and qrels.",
        "- Qrels: `BeIR/scifact-qrels` `test` split, previously aligned to the official SciFact development set in the Phase 2 manifest; this run verified 300 qrel query IDs, 339 relevance judgments, and no query IDs outside official dev.",
        "- The official SciFact test claims file was not opened. No generative model or additional pretrained model was used.",
        f"- Reranker: `{MODEL_NAME}`, CPU, batch size {BATCH_SIZE}; dense retrieval uses the Phase 2 embedding model and hybrid uses RRF constant {RRF_CONSTANT}.",
        f"- Candidate pools: {', '.join(map(str, CANDIDATE_POOLS))}; evaluation cutoffs: {', '.join(map(str, CUTOFFS))}.",
        "- Candidate scores for the maximum pool of 50 were computed once per query and retrieval system, then reused for pool sizes 10 and 20. Cross-encoder query-document scores are independent of other candidates, so this preserves the exact rankings while avoiding redundant inference.",
        "- Historical `results/runs/run_0001` through `run_0003` trees were SHA-256 checked before and after the run.",
        "- Dense and hybrid baseline per-query metrics at pool 20 were compared with the Phase 2 canonical run and matched at full precision.",
        "",
        "## CPU smoke test",
        "",
        f"- Status: {smoke['status']}; {smoke['query_count']} dev queries, both retrieval systems, 50 candidates each.",
        f"- Cross-encoder load time: {smoke['model_load_seconds']:.4f} s.",
        f"- Mean reranking latency at pool 50: {smoke['mean_pool_50_reranking_seconds_per_query_system']:.4f} s/query-system.",
        f"- Mean cross-encoder inference at pool 50: {smoke['mean_pool_50_inference_seconds_per_query_system']:.4f} s/query-system.",
        f"- Estimated full retrieval/reranking runtime: {smoke['estimated_full_end_to_end_minutes_with_score_reuse']:.1f} min based on smoke retrieval and reranking measurements.",
        f"- Estimated full reranking time with score reuse: {smoke['estimated_full_run_reranking_minutes_with_50_pool_score_reuse']:.2f} min.",
        "- Smoke-test measurements were used only for technical validation and runtime estimation; they are not included in the final metrics.",
        "",
        "## Aggregate retrieval metrics",
        "",
        "All values are means over the same 300 development queries. The machine-readable artifact retains full-precision values.",
        "",
    ]
    for pool_size in CANDIDATE_POOLS:
        lines.extend([
            f"### Candidate pool {pool_size}",
            "",
            "| Metric | Dense | Dense + reranking | Hybrid | Hybrid + reranking |",
            "|---|---:|---:|---:|---:|",
        ])
        for metric in metric_names:
            values = [
                aggregates[f"{method}_pool_{pool_size}_{condition}"][metric]
                for method, condition in (
                    ("dense", "baseline"), ("dense", "reranked"),
                    ("hybrid", "baseline"), ("hybrid", "reranked"),
                )
            ]
            lines.append(f"| {metric} | " + " | ".join(f"{value:.9f}" for value in values) + " |")
        lines.append("")

    lines.extend([
        "## Paired statistical analysis",
        "",
        f"- Pairwise comparisons: dense vs dense + reranking, and hybrid vs hybrid + reranking, for each candidate pool.",
        f"- Paired bootstrap: {BOOTSTRAP_ITERATIONS:,} resamples, 95% confidence intervals, seed {SEED}.",
        "- Tests: exact two-sided McNemar for Hit@K and two-sided Wilcoxon signed-rank for the other metrics.",
        "- Holm-Bonferroni correction across the full family of 144 tests (all metrics, cutoffs, pools, and both base systems).",
        f"- Practical significance threshold: absolute mean metric difference ≥ {PRACTICAL_DIFFERENCE_THRESHOLD:.2f}. Positive differences favor reranking.",
        "",
        "| System | Pool | Metric | Difference | 95% bootstrap CI | Holm-adjusted p | Interpretation |",
        "|---|---:|---|---:|---|---:|---|",
    ])
    for item in statistics["pairs"]:
        if not item["metric"].endswith("@10"):
            continue
        low, high = item["bootstrap_ci95"]
        lines.append(
            f"| {item['base_method']} | {item['candidate_pool_size']} | {item['metric']} | "
            f"{item['mean_difference_reranked_minus_baseline']:.9f} | "
            f"[{low:.9f}, {high:.9f}] | {item['p_value_holm_adjusted']:.9g} | "
            f"{item['interpretation']} |"
        )
    lines.extend([
        "",
        "The complete 144-test output, paired tests, confidence intervals, effect sizes, and practical-significance classifications are in `statistical_tests.json`.",
        "",
        "## CPU performance",
        "",
        f"- Cross-encoder model load: {performance['cross_encoder_model_load_seconds']:.4f} s.",
        f"- Dense index load: {performance['dense_index_load_seconds']:.4f} s; BM25 index build: {performance['bm25_index_build_seconds']:.4f} s.",
        "- Query latency uses one cached retrieval at depth 50 per query and system; candidate prefixes are reused for all three pool sizes.",
        "- Pool 10/20 reranking latency includes the shared 50-candidate scoring pass; it is not represented as a separate smaller-pool inference timing.",
        "",
        "| Stage | Mean (s) | Median (s) | P95 (s) | Throughput (query-systems/s) |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in (
        ("Initial retrieval at depth 50", "retrieval_latency_summary"),
        ("Cross-encoder inference", "cross_encoder_inference_latency_summary"),
        ("Reranking (end-to-end call)", "reranking_latency_summary"),
    ):
        item = performance[key]
        lines.append(
            f"| {label} | {item['mean_seconds']:.6f} | {item['median_seconds']:.6f} | "
            f"{item['p95_seconds']:.6f} | {item['throughput_queries_per_second']:.6f} |"
        )
    lines.extend([
        "",
        "| System | Pool | Condition | Mean (s) | Median (s) | P95 (s) | Throughput (q/s) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for method in ("dense", "hybrid"):
        for pool_size in CANDIDATE_POOLS:
            for condition in ("baseline", "reranked"):
                item = performance[f"{method}_pool_{pool_size}_{condition}"]
                lines.append(
                    f"| {method} | {pool_size} | {condition} | {item['mean_seconds']:.6f} | "
                    f"{item['median_seconds']:.6f} | {item['p95_seconds']:.6f} | "
                    f"{item['throughput_queries_per_second']:.6f} |"
                )
    memory = performance["peak_memory"]
    lines.extend([
        "",
        f"- Peak process working set: {memory['peak_working_set_bytes']} bytes; current working set at measurement: {memory['working_set_bytes']} bytes ({memory['status']}).",
        f"- Total run wall time, including artifact generation and validation: {runtime_seconds / 60:.2f} minutes.",
        f"- Test suite excluding the sole official-test-file reader: **{validation['status']}**, {validation['test_pass_count']} passed (exit code {validation['exit_code']}).",
        "- `tests/test_dataset_validation.py` was excluded because it reads the official SciFact test claims; running it would violate the no-test-split-access rule.",
        "",
        "## Phase 2 comparison and limitations",
        "",
        "- Dense and hybrid baseline metrics at pool 20 are compared query-by-query with canonical `results/runs/run_0003`; validation requires equality within 1e-12.",
        "- This is an official development-set experiment, not final held-out evaluation. Development qrels and results must not be interpreted as test-set performance.",
        "- CPU timings are machine- and cache-dependent. Reranking scores are measured at pool 50 once and reused for smaller prefixes; no separate 10/20 inference latency is claimed.",
        "",
        "## Reproducibility",
        "",
        "From the repository root, run `\\.venv\\Scripts\\python.exe experiments\\run_experiment_a.py` to allocate a fresh unique run. To resume an interrupted run, pass `--resume <run_id>`; validated per-query checkpoints are reused.",
        "",
        "## Figures",
        "",
        *[f"- `results/runs/{run_dir.name}/figures/{name}`" for name in (
            "dense_vs_reranking.png",
            "hybrid_vs_reranking.png",
            "candidate_pool_comparison.png",
            "quality_vs_latency.png",
            "statistical_confidence_intervals.png",
        )],
        "",
        "## Artifacts",
        "",
        f"- Metadata/config: `results/runs/{run_dir.name}/metadata.json`, `config.json`",
        f"- Smoke test: `results/runs/{run_dir.name}/smoke_test.json`",
        f"- Per-query rankings: `results/runs/{run_dir.name}/per_query_rankings.json` (plus JSONL)",
        f"- Per-query metrics: `results/runs/{run_dir.name}/per_query_metrics.json` (plus CSV)",
        f"- Aggregate metrics: `results/runs/{run_dir.name}/aggregate_metrics.json`",
        f"- Statistical results: `results/runs/{run_dir.name}/statistical_tests.json`",
        f"- Phase 2 baseline reproduction check: `results/runs/{run_dir.name}/baseline_reproduction_check.json`",
        f"- Query performance: `results/runs/{run_dir.name}/query_performance.csv`",
        f"- Performance summary: `results/runs/{run_dir.name}/performance_metrics.json`",
        f"- Environment metadata: `results/runs/{run_dir.name}/environment_metadata.json`",
        f"- Test output: `results/runs/{run_dir.name}/validation.json`",
        f"- Plots: `results/runs/{run_dir.name}/figures/`",
        "",
        "## Scope",
        "",
        "Only Experiment A was executed. No Phase 3 completion claim is made; Experiments B and C remain unstarted.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_configuration_outputs(run_dir: Path) -> None:
    with (run_dir / "per_query_metrics.json").open("r", encoding="utf-8") as handle:
        metric_rows = json.load(handle)
    with (run_dir / "per_query_rankings.json").open("r", encoding="utf-8") as handle:
        ranking_rows = json.load(handle)
    metrics_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rankings_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        key = f"{row['base_method']}_pool_{row['candidate_pool_size']}_{row['condition']}"
        metrics_by_config[key].append(row)
    for row in ranking_rows:
        key = f"{row['base_method']}_pool_{row['candidate_pool_size']}_{row['condition']}"
        rankings_by_config[key].append(row)
    for key in sorted(metrics_by_config):
        output_dir = run_dir / "configuration_outputs" / key
        _atomic_write_json(output_dir / "per_query_metrics.json", metrics_by_config[key])
        _atomic_write_json(output_dir / "per_query_rankings.json", rankings_by_config[key])


def _validate_experiment_artifacts(
    run_dir: Path,
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, Any]:
    from src.evaluation.retrieval_metrics import compute_retrieval_metrics

    with (run_dir / "per_query_metrics.json").open("r", encoding="utf-8") as handle:
        metric_rows = json.load(handle)
    with (run_dir / "per_query_rankings.json").open("r", encoding="utf-8") as handle:
        ranking_rows = json.load(handle)
    with (run_dir / "aggregate_metrics.json").open("r", encoding="utf-8") as handle:
        aggregates = json.load(handle)
    expected_ids = {row["query_id"] for row in queries}
    expected_configurations = {
        (method, pool, condition)
        for method in ("dense", "hybrid")
        for pool in CANDIDATE_POOLS
        for condition in ("baseline", "reranked")
    }

    def keyed_rows(rows: list[dict[str, Any]], artifact: str) -> dict[tuple[str, int, str, str], dict[str, Any]]:
        keyed = {}
        for row in rows:
            key = (
                str(row["base_method"]),
                int(row["candidate_pool_size"]),
                str(row["condition"]),
                str(row["query_id"]),
            )
            if key in keyed:
                raise AssertionError(f"Duplicate {artifact} record: {key}")
            keyed[key] = row
        return keyed

    metrics = keyed_rows(metric_rows, "metric")
    rankings = keyed_rows(ranking_rows, "ranking")
    expected_keys = {
        (method, pool, condition, query_id)
        for method, pool, condition in expected_configurations
        for query_id in expected_ids
    }
    if metrics.keys() != expected_keys or rankings.keys() != expected_keys:
        raise AssertionError(
            f"Expected {len(expected_keys)} complete query/configuration results; "
            f"found {len(metrics)} metrics and {len(rankings)} rankings"
        )
    max_metric_error = 0.0
    for key in expected_keys:
        metric_row = metrics[key]
        ranking = rankings[key]["rankings"]
        document_ids = [str(row["document_id"]) for row in ranking]
        if len(document_ids) != len(set(document_ids)):
            raise AssertionError(f"Duplicate candidate IDs in ranking {key}")
        if [int(row["rank"]) for row in ranking] != list(range(1, len(ranking) + 1)):
            raise AssertionError(f"Invalid rank sequence in {key}")
        method, pool, condition, query_id = key
        if condition == "reranked" and any(int(row["original_rank"]) > pool for row in ranking):
            raise AssertionError(f"Reranking introduced a candidate outside pool {pool}: {key}")
        actual = compute_retrieval_metrics(document_ids, qrels[query_id], CUTOFFS)
        for metric, value in actual.items():
            difference = abs(float(metric_row[metric]) - float(value))
            max_metric_error = max(max_metric_error, difference)
            if difference > 1e-12:
                raise AssertionError(f"Per-query metric does not match ranking for {key}, {metric}")

    max_aggregate_error = 0.0
    for method, pool, condition in expected_configurations:
        key = f"{method}_pool_{pool}_{condition}"
        for metric in _metric_names():
            values = [
                float(metrics[(method, pool, condition, query_id)][metric])
                for query_id in sorted(expected_ids)
            ]
            difference = abs(float(aggregates[key][metric]) - mean(values))
            max_aggregate_error = max(max_aggregate_error, difference)
            if difference > 1e-12:
                raise AssertionError(f"Aggregate metric mismatch for {key}, {metric}")
        for query_id in expected_ids:
            baseline_ids = [
                row["document_id"]
                for row in rankings[(method, pool, "baseline", query_id)]["rankings"]
            ]
            reranked_ids = [
                row["document_id"]
                for row in rankings[(method, pool, "reranked", query_id)]["rankings"]
            ]
            if set(baseline_ids) != set(reranked_ids):
                raise AssertionError(f"Candidate IDs changed during reranking for {(method, pool, query_id)}")

    expected_figures = {
        "dense_vs_reranking.png",
        "hybrid_vs_reranking.png",
        "candidate_pool_comparison.png",
        "quality_vs_latency.png",
        "statistical_confidence_intervals.png",
    }
    if not all((run_dir / "figures" / name).is_file() for name in expected_figures):
        raise AssertionError("One or more required figures are missing")
    return {
        "query_count_per_configuration": len(expected_ids),
        "configuration_count": len(expected_configurations),
        "per_query_metric_rows": len(metrics),
        "per_query_ranking_rows": len(rankings),
        "duplicate_query_results": False,
        "missing_query_results": False,
        "candidate_ids_preserved": True,
        "all_metrics_match_rankings": True,
        "all_aggregates_match_per_query_metrics": True,
        "max_metric_absolute_error": max_metric_error,
        "max_aggregate_absolute_error": max_aggregate_error,
        "required_figures": sorted(expected_figures),
        "official_test_split_accessed": False,
    }


def main(resume_run_id: str | None = None) -> None:
    global ACTIVE_RUN_DIR, RUN_LOG
    run_started = time.perf_counter()
    protected_runs = [RUN_ROOT / f"run_{run_number:04d}" for run_number in (1, 2, 3)]
    protected_hashes_before = {path.name: _sha256_tree(path) for path in protected_runs}
    queries = _read_dev_queries()
    query_ids = {row["query_id"] for row in queries}
    qrels = _load_dev_qrels(query_ids)
    qrels_document_count = sum(len(documents) for documents in qrels.values())

    if resume_run_id:
        if Path(resume_run_id).name != resume_run_id or not resume_run_id.startswith("run_"):
            raise ValueError("--resume must be a run directory name such as run_0006")
        run_dir = RUN_ROOT / resume_run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing run directory: {run_dir}")
        existing_config = run_dir / "config.json"
        if not existing_config.is_file():
            raise ValueError(f"Cannot resume run without its config.json: {run_dir}")
        with existing_config.open("r", encoding="utf-8") as handle:
            previous_config = json.load(handle)
        if previous_config.get("experiment") != "Phase 3.2 Experiment A":
            raise ValueError(f"Run is not a Phase 3.2 Experiment A run: {run_dir}")
    else:
        run_dir = unique_run_dir(RUN_ROOT, prefix="run")
    ACTIVE_RUN_DIR = run_dir
    RUN_LOG = (run_dir / "experiment.log").open("a", encoding="utf-8")
    config = {
        "experiment": "Phase 3.2 Experiment A",
        "objective": "Evaluate cross-encoder reranking against dense and hybrid retrieval baselines",
        "dataset": "SciFact official development claims",
        "query_file": DEV_CLAIMS_PATH.relative_to(ROOT).as_posix(),
        "query_count": len(queries),
        "qrels_source": "BeIR/scifact-qrels test qrels, which the verified project manifest aligns to official SciFact development claims",
        "qrels_query_count": len(qrels),
        "qrels_document_judgments": qrels_document_count,
        "qrels_query_ids_must_be_subset_of_official_dev_ids": True,
        "official_test_claim_file_opened": False,
        "candidate_pools": list(CANDIDATE_POOLS),
        "evaluation_cutoffs": list(CUTOFFS),
        "systems": {
            "dense": {"model": "sentence-transformers/all-MiniLM-L6-v2", "index": "existing FAISS index"},
            "hybrid": {"method": "reciprocal rank fusion", "rrf_constant": RRF_CONSTANT, "bm25_depth": RETRIEVAL_DEPTH, "dense_depth": RETRIEVAL_DEPTH},
        },
        "reranker": {"model": MODEL_NAME, "device": "cpu", "batch_size": BATCH_SIZE, "strict_model_and_inference": True},
        "scoring_optimization": "Score the maximum 50 candidates once per query/system; reuse each independent query-document score for candidate prefixes 10 and 20.",
        "statistics": {
            "bootstrap_resamples": BOOTSTRAP_ITERATIONS,
            "confidence_level": 0.95,
            "seed": SEED,
            "paired_tests": {"hit": "exact McNemar", "other_metrics": "Wilcoxon signed-rank"},
            "holm_family_size": 144,
            "practical_difference_threshold": PRACTICAL_DIFFERENCE_THRESHOLD,
        },
        "run_id": run_dir.name,
        "run_directory": str(run_dir),
        "resumed": bool(resume_run_id),
        "status": "running",
    }
    _atomic_write_json(run_dir / "config.json", config)
    metadata = {
        "run_id": run_dir.name,
        "experiment": "Phase 3.2 Experiment A",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_split": "official SciFact development",
        "query_count": len(queries),
        "qrels_query_count": len(qrels),
        "official_test_claim_file_opened": False,
        "protected_phase2_hashes_before": protected_hashes_before,
    }
    _atomic_write_json(run_dir / "metadata.json", metadata)
    _log(f"run_id={run_dir.name}")

    import torch

    torch.set_num_threads(TORCH_THREADS)
    framework_import_start = time.perf_counter()
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.retriever import Retriever
    from src.retrieval import embedder as dense_embedder

    retrieval_framework_import_seconds = time.perf_counter() - framework_import_start
    if dense_embedder._MODEL is None:
        raise RuntimeError("The Phase 2 dense embedding model is unavailable; refusing fallback retrieval")
    dense_init_start = time.perf_counter()
    dense = Retriever(rebuild=False)
    dense_index_load_seconds = time.perf_counter() - dense_init_start
    bm25_init_start = time.perf_counter()
    bm25 = BM25Retriever(corpus_path=CORPUS_PATH, candidate_depth=RETRIEVAL_DEPTH)
    bm25_index_build_seconds = time.perf_counter() - bm25_init_start
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        rrf_constant=RRF_CONSTANT,
        bm25_depth=RETRIEVAL_DEPTH,
        dense_depth=RETRIEVAL_DEPTH,
    )

    model_load_start = time.perf_counter()
    reranker = CrossEncoderReranker(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device="cpu",
        max_candidates=max(CANDIDATE_POOLS),
        strict=True,
    )
    reranker.model_load_seconds = time.perf_counter() - model_load_start
    if reranker._model is None:
        raise RuntimeError("The required pretrained cross-encoder is unavailable; refusing fallback scoring")
    memory_snapshots = {"after_retriever_and_model_initialization": _memory_snapshot()}
    smoke = _smoke_test(queries, dense, hybrid, reranker, dense.corpus_by_id, _memory_snapshot)
    memory_snapshots["after_cpu_smoke_test"] = smoke["peak_memory_after_smoke"]
    _atomic_write_json(run_dir / "smoke_test.json", smoke)
    estimated_minutes = smoke["estimated_full_run_reranking_minutes_with_50_pool_score_reuse"]
    _log(
        f"smoke_test=passed queries={smoke['query_count']} "
        f"cross_encoder_load_seconds={reranker.model_load_seconds:.2f} "
        f"estimated_full_reranking_minutes={estimated_minutes:.1f} "
        f"estimated_full_end_to_end_minutes={smoke['estimated_full_end_to_end_minutes_with_score_reuse']:.1f}"
    )

    aggregates, performance = _evaluate(
        run_dir, queries, qrels, dense, hybrid, reranker, dense.corpus_by_id
    )
    memory_snapshots["after_full_evaluation"] = _memory_snapshot()
    _atomic_write_json(run_dir / "aggregate_metrics.json", aggregates)

    baseline_reproduction = _validate_phase2_baselines(run_dir / "per_query_metrics.csv")
    _atomic_write_json(run_dir / "baseline_reproduction_check.json", baseline_reproduction)
    statistics = _paired_statistical_analysis(queries, run_dir / "per_query_metrics.csv")
    _atomic_write_json(run_dir / "statistical_tests.json", statistics)
    performance.update({
        "cross_encoder_model_load_seconds": reranker.model_load_seconds,
        "retrieval_framework_import_seconds": retrieval_framework_import_seconds,
        "dense_index_load_seconds": dense_index_load_seconds,
        "bm25_index_build_seconds": bm25_index_build_seconds,
        "peak_memory": _memory_snapshot(),
        "memory_measurement_scope": "Windows process peak working set and pagefile usage; includes Python, retrieval indexes, and cross-encoder model",
    })
    _atomic_write_json(run_dir / "performance_metrics.json", performance)
    generated_figures = _generate_plots(run_dir)
    validation = _run_focused_tests(run_dir)
    artifact_validation = _validate_experiment_artifacts(run_dir, queries, qrels)
    _atomic_write_json(run_dir / "artifact_validation.json", artifact_validation)
    _write_configuration_outputs(run_dir)
    protected_hashes_after = {path.name: _sha256_tree(path) for path in protected_runs}
    if protected_hashes_after != protected_hashes_before:
        raise RuntimeError("Protected historical runs changed during Experiment A")

    total_runtime_seconds = time.perf_counter() - run_started
    performance["total_run_wall_time_seconds"] = total_runtime_seconds
    _atomic_write_json(run_dir / "performance_metrics.json", performance)
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "datasets_version": _package_version("datasets"),
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "faiss_version": _package_version("faiss-cpu"),
        "git_commit": _git_commit(),
        "device": "CPU",
        "dense_embedding_model_loaded": True,
        "memory_snapshots": memory_snapshots,
        "protected_phase2_runs": ["results/runs/run_0001", "results/runs/run_0002", "results/runs/run_0003"],
        "protected_phase2_hashes_before": protected_hashes_before,
        "protected_phase2_hashes_after": protected_hashes_after,
        "test_suite_status": validation["status"],
        "test_suite_scope": validation["suite_scope"],
    }
    _atomic_write_json(run_dir / "environment_metadata.json", environment)
    metadata.update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if validation["status"] == "passed" else "completed_with_test_failure",
        "python_version": environment["python_version"],
        "platform": environment["platform"],
        "git_commit": environment["git_commit"],
        "artifact_validation": artifact_validation,
        "protected_phase2_hashes_after": protected_hashes_after,
    })
    _atomic_write_json(run_dir / "metadata.json", metadata)
    report_path = _write_experiment_report(
        run_dir, aggregates, performance, statistics, smoke, validation, total_runtime_seconds
    )
    config["status"] = "completed" if validation["status"] == "passed" else "completed_with_test_failure"
    config["focused_test_status"] = validation["status"]
    _atomic_write_json(run_dir / "config.json", config)
    _atomic_write_json(run_dir / "run_status.json", {
        "run_id": run_dir.name,
        "status": "completed" if validation["status"] == "passed" else "completed_with_test_failure",
        "run_wall_time_seconds": total_runtime_seconds,
        "generated_figures": generated_figures,
        "documentation_report": str(report_path.relative_to(ROOT).as_posix()),
        "test_suite_status": validation["status"],
        "test_suite_scope": validation["suite_scope"],
        "phase2_baseline_reproduction": baseline_reproduction["status"],
        "historical_runs_unchanged": True,
        "protected_phase2_hashes_after": protected_hashes_after,
    })
    _log(f"run_id={run_dir.name}")
    _log(f"run_directory={run_dir}")
    _log(f"total_runtime_minutes={total_runtime_seconds / 60:.2f}")
    _log(f"tests={validation['status']} ({validation['test_pass_count']} passed)")
    _log(f"documentation_report={report_path}")
    _log(json.dumps(aggregates, sort_keys=True))
    if validation["status"] != "passed":
        raise RuntimeError("Experiment artifacts were saved, but the safe test suite failed")
    RUN_LOG.close()
    RUN_LOG = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3.2 Experiment A.")
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume a validated run checkpoint, e.g. run_0006")
    arguments = parser.parse_args()
    try:
        main(resume_run_id=arguments.resume)
    except BaseException as exc:
        if ACTIVE_RUN_DIR is not None:
            _atomic_write_json(ACTIVE_RUN_DIR / "run_status.json", {
                "run_id": ACTIVE_RUN_DIR.name,
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        if RUN_LOG is not None:
            RUN_LOG.write(f"run_failed={type(exc).__name__}: {exc}\n")
            RUN_LOG.write(traceback.format_exc())
            RUN_LOG.close()
        raise
