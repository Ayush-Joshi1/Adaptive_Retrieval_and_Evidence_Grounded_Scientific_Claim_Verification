from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEV_CLAIMS_PATH = ROOT / "data" / "processed" / "raw_scifact" / "data" / "claims_dev.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
RUN_ROOT = ROOT / "results" / "runs"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 16
TORCH_THREADS = 2
RRF_CONSTANT = 60
BOOTSTRAP_ITERATIONS = 10_000
SEED = 42
FIXED_DEPTHS = (10, 20, 50)
ADAPTIVE_STRATEGIES = ("score_gap", "confidence")
SCORE_GAP_THRESHOLD = 0.10
CONFIDENCE_THRESHOLD = 0.25
ACTIVE_RUN_DIR: Path | None = None
RUN_LOG: Any = None

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_manager import unique_run_dir


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _log(message: str) -> None:
    print(message, flush=True)
    if RUN_LOG is not None:
        RUN_LOG.write(message + "\n")
        RUN_LOG.flush()


def _read_queries() -> list[dict[str, str]]:
    rows = []
    with DEV_CLAIMS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({"query_id": str(row["id"]), "claim_text": str(row.get("claim") or "")})
    ids = [row["query_id"] for row in rows]
    if len(rows) != 300 or len(set(ids)) != 300:
        raise ValueError("Expected exactly 300 unique official development claims")
    return rows


def _load_qrels(query_ids: set[str]) -> dict[str, dict[str, float]]:
    from datasets import load_dataset

    qrels = {}
    split = load_dataset("BeIR/scifact-qrels", split="test")
    for row in split:
        qid = str(row["query-id"])
        if qid not in query_ids:
            raise ValueError(f"Qrels contained a query outside official development set: {qid}")
        qrels.setdefault(qid, {})[str(row["corpus-id"])] = float(row["score"])
    if set(qrels) != query_ids:
        raise ValueError("Development qrels do not cover all 300 development queries")
    return qrels


def _sha256_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _memory_snapshot() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "unavailable", "peak_working_set_bytes": None, "working_set_bytes": None}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    ok = get_memory(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return {"status": "unavailable", "peak_working_set_bytes": None, "working_set_bytes": None}
    return {
        "status": "measured via Windows GetProcessMemoryInfo",
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "working_set_bytes": int(counters.WorkingSetSize),
        "pagefile_usage_bytes": int(counters.PagefileUsage),
    }


def _retrieve(method: str, retriever: Any, claim: str, query_id: str, depth: int) -> list[dict[str, Any]]:
    if method == "hybrid":
        return retriever.retrieve(
            claim,
            top_k=depth,
            query_id=query_id,
            bm25_depth=depth,
            dense_depth=depth,
        )
    return retriever.retrieve(claim, top_k=depth, query_id=query_id)


def _hydrate(method: str, candidates: list[dict[str, Any]], corpus: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if method == "dense":
        return list(candidates)
    hydrated = []
    for candidate in candidates:
        cid = str(candidate["document_id"])
        document = corpus[cid]
        hydrated.append({**candidate, "title": document.get("title"), "text": document.get("text")})
    return hydrated


def _latency(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"mean_seconds": 0.0, "median_seconds": 0.0, "p95_seconds": 0.0,
                "throughput_queries_per_second": 0.0, "total_seconds": 0.0, "query_count": 0}
    p95 = ordered[min(len(ordered) - 1, int(__import__("math").ceil(0.95 * len(ordered))) - 1)]
    total = sum(values)
    return {
        "mean_seconds": mean(values),
        "median_seconds": median(values),
        "p95_seconds": p95,
        "throughput_queries_per_second": len(values) / total if total else 0.0,
        "total_seconds": total,
        "query_count": len(values),
    }


def _metric_names() -> list[str]:
    return ["hit@10", "recall@10", "mrr@10", "map@10", "ndcg@10"]


def _adaptive_candidates(
    method: str,
    retriever: Any,
    adaptive: Any,
    claim: str,
    query_id: str,
    corpus: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], float, float]:
    retrieval_start = time.perf_counter()
    candidate_results: list[dict[str, Any]] = []
    stage_depths: list[int] = []
    stage_records: list[dict[str, Any]] = []
    stopping_signals: dict[str, Any] = {}
    seen_candidate_ids: set[str] = set()

    def retrieve_stage(depth: int) -> tuple[list[dict[str, Any]], list[float]]:
        stage_start = time.perf_counter()
        rows = _retrieve(method, retriever, claim, query_id, depth)
        stage_seconds = time.perf_counter() - stage_start
        ids = [str(row["document_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate candidates at adaptive depth {depth}: {query_id}")
        added_ids = [candidate_id for candidate_id in ids if candidate_id not in seen_candidate_ids]
        scores = [float(row.get("retrieval_score", row.get("score", 0.0))) for row in rows]
        stage_records.append({
            "requested_depth": depth,
            "backend_result_count": len(rows),
            "new_candidate_count": len(added_ids),
            "candidate_ids": ids,
            "added_candidate_ids": added_ids,
            "retrieval_scores": scores,
            "retrieval_latency_seconds": stage_seconds,
        })
        seen_candidate_ids.update(ids)
        stage_depths.append(depth)
        return rows, scores

    def stopping_record(scores: list[float]) -> dict[str, Any]:
        strategy = adaptive._strategy_impl
        normalized_scores = strategy.normalize_scores(scores)
        return {
            "threshold": adaptive.threshold,
            "should_stop": bool(strategy.should_stop(scores)),
            "retrieval_scores": scores,
            "normalized_scores": normalized_scores,
        }

    candidate_results, scores_10 = retrieve_stage(10)
    signal_10 = stopping_record(scores_10)
    stopping_signals["after_10"] = signal_10
    if not signal_10["should_stop"]:
        candidate_results, scores_20 = retrieve_stage(20)
        signal_20 = stopping_record(scores_20)
        stopping_signals["after_20"] = signal_20
        if not signal_20["should_stop"]:
            candidate_results, _ = retrieve_stage(50)

    final_depth = len(candidate_results)
    retrieval_seconds = time.perf_counter() - retrieval_start
    hydrated = _hydrate(method, candidate_results, corpus)
    decision = {
        "initial_candidate_depth": 10,
        "final_candidate_depth": final_depth,
        "number_of_expansion_stages": max(0, len(stage_depths) - 1),
        "number_of_retrieval_stages": len(stage_depths),
        "number_unique_candidates_seen_across_stages": len(seen_candidate_ids),
        "unique_candidate_ids_seen_across_stages": sorted(seen_candidate_ids),
        "stage_depths": stage_depths,
        "stopped_at_10": len(stage_depths) == 1,
        "stopped_at_20": len(stage_depths) == 2,
        "stopped_at_50": len(stage_depths) == 3,
        "stopping_signals": stopping_signals,
        "stage_records": stage_records,
        "candidate_retrieval_scores": {
            str(row["document_id"]): float(row.get("retrieval_score", row.get("score", 0.0)))
            for row in candidate_results
        },
        "candidate_ids": [str(row["document_id"]) for row in candidate_results],
    }
    return hydrated, decision, retrieval_seconds, sum(
        float(stage["retrieval_latency_seconds"]) for stage in stage_records
    )


def _valid_checkpoint(path: Path, query_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("query_id") != query_id:
        raise ValueError(f"Checkpoint query mismatch: {path}")
    expected_configurations = {
        f"{method}_fixed_top_{depth}"
        for method in ("dense", "hybrid")
        for depth in FIXED_DEPTHS
    } | {
        f"{method}_{strategy}_adaptive"
        for method in ("dense", "hybrid")
        for strategy in ADAPTIVE_STRATEGIES
    }
    fields = ("metric_rows", "ranking_rows", "decision_rows", "performance_rows")
    if any(not isinstance(checkpoint.get(field), list) for field in fields):
        raise ValueError(f"Incomplete checkpoint: {path}")
    if any(len(checkpoint[field]) != len(expected_configurations) for field in fields):
        raise ValueError(f"Checkpoint does not contain all 10 configurations: {path}")
    for field in fields:
        keys = [
            (row.get("query_id"), row.get("retrieval_method"), row.get("configuration"))
            for row in checkpoint[field]
        ]
        if len(set(keys)) != len(keys) or {
            key[2] for key in keys if key[0] == query_id
        } != expected_configurations:
            raise ValueError(f"Duplicate or missing configuration in checkpoint {field}: {path}")
    return checkpoint


def _smoke_test(
    run_dir: Path,
    queries: list[dict[str, str]],
    dense: Any,
    hybrid: Any,
    reranker: Any,
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    from src.retrieval.adaptive_retriever import AdaptiveRetriever

    smoke_queries = queries[:3]
    for query in smoke_queries:
        for method, retriever in (("dense", dense), ("hybrid", hybrid)):
            for depth in FIXED_DEPTHS:
                retrieval_start = time.perf_counter()
                candidates = _hydrate(
                    method,
                    _retrieve(method, retriever, query["claim_text"], query["query_id"], depth),
                    corpus,
                )
                retrieval_seconds = time.perf_counter() - retrieval_start
                configuration = f"{method}_fixed_top_{depth}"
                decision_depth = depth
                start = time.perf_counter()
                ranked = reranker.rerank(
                    query["claim_text"], candidates, top_k=None, query_id=query["query_id"]
                )
                rerank_seconds = time.perf_counter() - start
                _validate_smoke_ranking(ranked, candidates, query["query_id"], configuration)
                rows.append({
                    "query_id": query["query_id"],
                    "method": method,
                    "configuration": configuration,
                    "final_candidate_depth": decision_depth,
                    "retrieval_seconds": retrieval_seconds,
                    "reranking_seconds": rerank_seconds,
                    "cross_encoder_inference_seconds": reranker.last_inference_seconds,
                    "candidate_count": len(candidates),
                })
            for strategy, threshold in (
                ("score_gap", SCORE_GAP_THRESHOLD),
                ("confidence", CONFIDENCE_THRESHOLD),
            ):
                adaptive = AdaptiveRetriever(
                    retriever,
                    strategy=strategy,
                    min_depth=10,
                    max_depth=50,
                    threshold=threshold,
                    initial_depth=10,
                )
                candidates, decision, retrieval_seconds, _ = _adaptive_candidates(
                    method, retriever, adaptive, query["claim_text"], query["query_id"], corpus
                )
                configuration = f"{method}_{strategy}_adaptive"
                start = time.perf_counter()
                ranked = reranker.rerank(
                    query["claim_text"], candidates, top_k=None, query_id=query["query_id"]
                )
                rerank_seconds = time.perf_counter() - start
                _validate_smoke_ranking(ranked, candidates, query["query_id"], configuration)
                rows.append({
                    "query_id": query["query_id"],
                    "method": method,
                    "configuration": configuration,
                    "final_candidate_depth": decision["final_candidate_depth"],
                    "retrieval_seconds": retrieval_seconds,
                    "reranking_seconds": rerank_seconds,
                    "cross_encoder_inference_seconds": reranker.last_inference_seconds,
                    "candidate_count": len(candidates),
                })
        _log(f"smoke_queries={len({row['query_id'] for row in rows})}/{len(smoke_queries)}")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["configuration"], []).append(
            row["retrieval_seconds"] + row["reranking_seconds"]
        )
    estimate_seconds = sum(mean(values) * len(queries) for values in grouped.values())
    result = {
        "status": "passed",
        "query_count": len(smoke_queries),
        "system_query_count": len(rows),
        "batch_size": BATCH_SIZE,
        "model_name": MODEL_NAME,
        "measurements": rows,
        "configuration_count": len(grouped),
        "estimated_full_end_to_end_minutes": estimate_seconds / 60.0,
        "estimate_method": "sum of each configuration's measured smoke mean latency multiplied by 300 queries",
    }
    _atomic_json(run_dir / "smoke_test.json", result)
    return result


def _validate_smoke_ranking(
    ranked: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    query_id: str,
    configuration: str,
) -> None:
    ranked_ids = [str(row["document_id"]) for row in ranked]
    candidate_ids = [str(row["document_id"]) for row in candidates]
    if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != set(candidate_ids):
        raise AssertionError(f"Smoke ranking changed candidate IDs: {query_id}, {configuration}")


def _evaluate(
    run_dir: Path,
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, float]],
    dense: Any,
    hybrid: Any,
    reranker: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from src.evaluation.retrieval_metrics import compute_retrieval_metrics
    from src.retrieval.adaptive_retriever import AdaptiveRetriever

    checkpoint_dir = run_dir / "checkpoints" / "queries"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    corpus = dense.corpus_by_id
    metric_rows = []
    ranking_rows = []
    decision_rows = []
    performance_rows = []
    started = time.perf_counter()

    for index, query in enumerate(queries, start=1):
        query_id = query["query_id"]
        claim = query["claim_text"]
        checkpoint_path = checkpoint_dir / f"{query_id}.json"
        checkpoint = _valid_checkpoint(checkpoint_path, query_id)
        if checkpoint is None:
            query_metrics = []
            query_rankings = []
            query_decisions = []
            query_performance = []

            for method, retriever in (("dense", dense), ("hybrid", hybrid)):
                for depth in FIXED_DEPTHS:
                    retrieval_start = time.perf_counter()
                    candidates = _retrieve(method, retriever, claim, query_id, depth)
                    retrieval_seconds = time.perf_counter() - retrieval_start
                    candidates = _hydrate(method, candidates, corpus)
                    rerank_start = time.perf_counter()
                    ranked = reranker.rerank(claim, candidates, top_k=None, query_id=query_id)
                    rerank_seconds = time.perf_counter() - rerank_start
                    ranked_ids = [str(row["document_id"]) for row in ranked]
                    candidate_ids = [str(row["document_id"]) for row in candidates]
                    if len(ranked_ids) != len(set(ranked_ids)) or not set(ranked_ids).issubset(set(candidate_ids)):
                        raise AssertionError(f"Fixed reranking changed candidate IDs: {query_id}, {method}, {depth}")
                    metrics = compute_retrieval_metrics(ranked_ids[:10], qrels[query_id], (10,))
                    query_metrics.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": f"{method}_fixed_top_{depth}",
                        "adaptive_strategy": "fixed",
                        **metrics,
                    })
                    query_rankings.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": f"{method}_fixed_top_{depth}",
                        "rankings": [
                            {
                                "document_id": str(row["document_id"]),
                                "rank": int(row["rank"]),
                                "original_rank": int(row["original_rank"]),
                                "retrieval_score": float(row["retrieval_score"]),
                                "cross_encoder_score": float(row["cross_encoder_score"]),
                            }
                            for row in ranked
                        ],
                    })
                    query_decisions.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": f"{method}_fixed_top_{depth}",
                        "adaptive_strategy": "fixed",
                        "initial_candidate_depth": depth,
                        "final_candidate_depth": depth,
                        "number_of_expansion_stages": 0,
                        "stage_depths": [depth],
                        "number_of_retrieval_stages": 1,
                        "stopped_early": False,
                        "stopping_depth": depth,
                        "number_candidates_retrieved": len(candidates),
                        "number_candidates_processed": len(candidates),
                        "number_candidates_cross_encoder_scored": len(candidates),
                        "number_candidate_results_returned_total": len(candidates),
                        "number_unique_candidates_seen_across_stages": len(candidate_ids),
                        "unique_candidate_ids_seen_across_stages": candidate_ids,
                        "candidate_retrieval_scores": {
                            str(row["document_id"]): float(row.get("retrieval_score", row.get("score", 0.0)))
                            for row in candidates
                        },
                        "candidate_ids": candidate_ids,
                        "stage_records": [{
                            "requested_depth": depth,
                            "backend_result_count": len(candidates),
                            "new_candidate_count": len(candidates),
                            "candidate_ids": candidate_ids,
                            "added_candidate_ids": candidate_ids,
                            "retrieval_scores": [
                                float(row.get("retrieval_score", row.get("score", 0.0)))
                                for row in candidates
                            ],
                            "retrieval_latency_seconds": retrieval_seconds,
                        }],
                        "stopping_signals": {},
                    })
                    query_performance.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": f"{method}_fixed_top_{depth}",
                        "retrieval_latency_seconds": retrieval_seconds,
                        "cross_encoder_inference_seconds": float(reranker.last_inference_seconds),
                        "reranking_latency_seconds": rerank_seconds,
                        "end_to_end_latency_seconds": retrieval_seconds + rerank_seconds,
                        "number_candidates_retrieved": len(candidates),
                        "number_candidates_processed": len(candidates),
                        "number_candidates_cross_encoder_scored": len(candidates),
                        "number_candidate_results_returned_total": len(candidates),
                    })

                for strategy, threshold in (
                    ("score_gap", SCORE_GAP_THRESHOLD),
                    ("confidence", CONFIDENCE_THRESHOLD),
                ):
                    adaptive = AdaptiveRetriever(
                        retriever,
                        strategy=strategy,
                        min_depth=10,
                        max_depth=50,
                        threshold=threshold,
                        initial_depth=10,
                    )
                    candidates, decision, retrieval_seconds, _ = _adaptive_candidates(
                        method, retriever, adaptive, claim, query_id, corpus
                    )
                    rerank_start = time.perf_counter()
                    ranked = reranker.rerank(claim, candidates, top_k=None, query_id=query_id)
                    rerank_seconds = time.perf_counter() - rerank_start
                    ranked_ids = [str(row["document_id"]) for row in ranked]
                    candidate_ids = [str(row["document_id"]) for row in candidates]
                    if len(ranked_ids) != len(set(ranked_ids)) or not set(ranked_ids).issubset(set(candidate_ids)):
                        raise AssertionError(f"Adaptive reranking changed candidate IDs: {query_id}, {method}, {strategy}")
                    metrics = compute_retrieval_metrics(ranked_ids[:10], qrels[query_id], (10,))
                    configuration = f"{method}_{strategy}_adaptive"
                    query_metrics.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": configuration,
                        "adaptive_strategy": strategy,
                        **metrics,
                    })
                    query_rankings.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": configuration,
                        "rankings": [
                            {
                                "document_id": str(row["document_id"]),
                                "rank": int(row["rank"]),
                                "original_rank": int(row["original_rank"]),
                                "retrieval_score": float(row["retrieval_score"]),
                                "cross_encoder_score": float(row["cross_encoder_score"]),
                            }
                            for row in ranked
                        ],
                    })
                    query_decisions.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": configuration,
                        "adaptive_strategy": strategy,
                        "initial_candidate_depth": 10,
                        "final_candidate_depth": decision["final_candidate_depth"],
                        "number_of_expansion_stages": decision["number_of_expansion_stages"],
                        "number_of_retrieval_stages": decision["number_of_retrieval_stages"],
                        "stage_depths": decision["stage_depths"],
                        "stopped_early": decision["final_candidate_depth"] < 50,
                        "stopping_depth": decision["final_candidate_depth"],
                        "number_candidates_retrieved": len(candidates),
                        "number_candidates_processed": len(candidates),
                        "number_candidates_cross_encoder_scored": len(candidates),
                        "number_candidate_results_returned_total": sum(
                            int(stage["backend_result_count"]) for stage in decision["stage_records"]
                        ),
                        "number_unique_candidates_seen_across_stages": decision["number_unique_candidates_seen_across_stages"],
                        "unique_candidate_ids_seen_across_stages": decision["unique_candidate_ids_seen_across_stages"],
                        "candidate_retrieval_scores": decision["candidate_retrieval_scores"],
                        "candidate_ids": candidate_ids,
                        "stage_records": decision["stage_records"],
                        "stopping_signals": decision["stopping_signals"],
                    })
                    query_performance.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "configuration": configuration,
                        "retrieval_latency_seconds": retrieval_seconds,
                        "cross_encoder_inference_seconds": float(reranker.last_inference_seconds),
                        "reranking_latency_seconds": rerank_seconds,
                        "end_to_end_latency_seconds": retrieval_seconds + rerank_seconds,
                        "number_candidates_retrieved": len(candidates),
                        "number_candidates_processed": len(candidates),
                        "number_candidates_cross_encoder_scored": len(candidates),
                        "number_candidate_results_returned_total": sum(
                            int(stage["backend_result_count"]) for stage in decision["stage_records"]
                        ),
                    })

            checkpoint = {
                "query_id": query_id,
                "metric_rows": query_metrics,
                "ranking_rows": query_rankings,
                "decision_rows": query_decisions,
                "performance_rows": query_performance,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(checkpoint_path, checkpoint)

        metric_rows.extend(checkpoint["metric_rows"])
        ranking_rows.extend(checkpoint["ranking_rows"])
        decision_rows.extend(checkpoint["decision_rows"])
        performance_rows.extend(checkpoint["performance_rows"])
        if index % 10 == 0 or index == len(queries):
            elapsed = time.perf_counter() - started
            _log(
                f"evaluated={index}/{len(queries)} "
                f"elapsed_minutes={elapsed / 60:.1f}"
            )

    expected = 300 * 10
    if len(metric_rows) != expected or len(ranking_rows) != expected:
        raise AssertionError(f"Expected {expected} query/config records; found {len(metric_rows)} metrics and {len(ranking_rows)} rankings")
    _atomic_json(run_dir / "per_query_metrics.json", metric_rows)
    _atomic_json(run_dir / "per_query_rankings.json", ranking_rows)
    _atomic_jsonl(run_dir / "per_query_rankings.jsonl", ranking_rows)
    _atomic_json(run_dir / "adaptive_retrieval_decisions.json", decision_rows)
    _atomic_jsonl(run_dir / "adaptive_retrieval_decisions.jsonl", decision_rows)
    _atomic_json(run_dir / "performance_measurements.json", performance_rows)

    with (run_dir / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "query_id", "retrieval_method", "configuration", "adaptive_strategy",
            *_metric_names(), "precision@10",
        ])
        writer.writeheader()
        writer.writerows(metric_rows)
    with (run_dir / "adaptive_retrieval_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "query_id", "retrieval_method", "configuration", "adaptive_strategy",
            "initial_candidate_depth", "final_candidate_depth", "number_of_expansion_stages",
            "number_of_retrieval_stages",
            "stage_depths", "stopped_early", "stopping_depth",
            "number_candidates_retrieved", "number_candidates_processed",
            "number_candidates_cross_encoder_scored", "number_candidate_results_returned_total",
            "number_unique_candidates_seen_across_stages", "unique_candidate_ids_seen_across_stages",
            "candidate_retrieval_scores", "candidate_ids", "stage_records", "stopping_signals",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in decision_rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items() if key in fields})
    with (run_dir / "query_performance.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "query_id", "retrieval_method", "configuration", "retrieval_latency_seconds",
            "cross_encoder_inference_seconds", "reranking_latency_seconds",
            "end_to_end_latency_seconds", "number_candidates_retrieved", "number_candidates_processed",
            "number_candidates_cross_encoder_scored", "number_candidate_results_returned_total",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(performance_rows)
    return metric_rows, ranking_rows, decision_rows, performance_rows


def _aggregate_and_statistics(
    run_dir: Path,
    metric_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    performance_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np
    from scipy.stats import wilcoxon
    from src.evaluation.statistical_tests import holm_bonferroni_adjust, paired_bootstrap_ci

    metrics = _metric_names()
    configurations = []
    for method in ("dense", "hybrid"):
        configurations.extend(
            [f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS]
            + [f"{method}_{strategy}_adaptive" for strategy in ADAPTIVE_STRATEGIES]
        )

    lookup = {
        (row["retrieval_method"], row["configuration"], row["query_id"]): {
            metric: float(row[metric]) for metric in metrics
        }
        for row in metric_rows
    }
    aggregates = {}
    for method in ("dense", "hybrid"):
        for depth in FIXED_DEPTHS:
            config = f"{method}_fixed_top_{depth}"
            aggregates[config] = {
                metric: mean(
                    lookup[(method, config, query_id)][metric]
                    for query_id in {
                        row["query_id"]
                        for row in metric_rows
                        if row["retrieval_method"] == method and row["configuration"] == config
                    }
                )
                for metric in metrics
            }
        for strategy in ADAPTIVE_STRATEGIES:
            config = f"{method}_{strategy}_adaptive"
            aggregates[config] = {
                metric: mean(
                    lookup[(method, config, query_id)][metric]
                    for query_id in {
                        row["query_id"]
                        for row in metric_rows
                        if row["retrieval_method"] == method and row["configuration"] == config
                    }
                )
                for metric in metrics
            }

    candidate_summary = {}
    for method in ("dense", "hybrid"):
        for config in [
            f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS
        ] + [f"{method}_{strategy}_adaptive" for strategy in ADAPTIVE_STRATEGIES]:
            rows = [row for row in decision_rows if row["retrieval_method"] == method and row["configuration"] == config]
            candidate_summary[config] = {
                "average_candidates_retrieved": mean(float(row["number_candidates_retrieved"]) for row in rows),
                "average_candidates_processed": mean(float(row["number_candidates_processed"]) for row in rows),
                "average_unique_candidates_seen_across_stages": mean(
                    float(row["number_unique_candidates_seen_across_stages"]) for row in rows
                ),
                "average_candidates_cross_encoder_scored": mean(float(row["number_candidates_cross_encoder_scored"]) for row in rows),
                "average_backend_results_returned_across_stages": mean(float(row["number_candidate_results_returned_total"]) for row in rows),
                "average_expansion_stages": mean(float(row["number_of_expansion_stages"]) for row in rows),
                "percentage_stopping_at_10": 100.0 * sum(row["stopping_depth"] == 10 for row in rows) / len(rows),
                "percentage_stopping_at_20": 100.0 * sum(row["stopping_depth"] == 20 for row in rows) / len(rows),
                "percentage_stopping_at_50": 100.0 * sum(row["stopping_depth"] == 50 for row in rows) / len(rows),
            }

    performance_summary = {}
    for method in ("dense", "hybrid"):
        for config in [
            f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS
        ] + [f"{method}_{strategy}_adaptive" for strategy in ADAPTIVE_STRATEGIES]:
            rows = [row for row in performance_rows if row["retrieval_method"] == method and row["configuration"] == config]
            performance_summary[config] = {
                "retrieval_latency": _latency([float(row["retrieval_latency_seconds"]) for row in rows]),
                "cross_encoder_inference": _latency([float(row["cross_encoder_inference_seconds"]) for row in rows]),
                "reranking_latency": _latency([float(row["reranking_latency_seconds"]) for row in rows]),
                "end_to_end_latency": _latency([float(row["end_to_end_latency_seconds"]) for row in rows]),
            }

    comparisons = []
    query_ids = sorted({row["query_id"] for row in metric_rows})
    for method in ("dense", "hybrid"):
        adaptive_configs = [f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive"]
        baseline_configs = [f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS]
        for adaptive in adaptive_configs:
            for baseline in baseline_configs:
                for metric in metrics:
                    adaptive_scores = np.asarray([
                        lookup[(method, adaptive, qid)][metric] for qid in query_ids
                    ], dtype=float)
                    baseline_scores = np.asarray([
                        lookup[(method, baseline, qid)][metric] for qid in query_ids
                    ], dtype=float)
                    differences = adaptive_scores - baseline_scores
                    bootstrap = paired_bootstrap_ci(
                        adaptive_scores.tolist(),
                        baseline_scores.tolist(),
                        n_bootstrap=BOOTSTRAP_ITERATIONS,
                        confidence_level=0.95,
                        seed=SEED,
                    )
                    if np.all(differences == 0):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(
                            adaptive_scores, baseline_scores,
                            zero_method="wilcox",
                            alternative="two-sided",
                            method="auto",
                        ).pvalue)
                    difference_mean = float(differences.mean())
                    std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
                    comparisons.append({
                        "retrieval_method": method,
                        "adaptive_configuration": adaptive,
                        "fixed_comparison": baseline,
                        "metric": metric,
                        "adaptive_mean": float(adaptive_scores.mean()),
                        "fixed_mean": float(baseline_scores.mean()),
                        "mean_difference_adaptive_minus_fixed": difference_mean,
                        "paired_cohens_dz": difference_mean / std if std else None,
                        "bootstrap_ci95": bootstrap["ci95"],
                        "bootstrap_resamples": BOOTSTRAP_ITERATIONS,
                        "test": "two-sided Wilcoxon signed-rank test",
                        "p_value": p_value,
                        "sample_size": len(query_ids),
                    })
    expected_comparison_count = 2 * len(ADAPTIVE_STRATEGIES) * len(FIXED_DEPTHS) * len(metrics)
    if len(comparisons) != expected_comparison_count:
        raise AssertionError(f"Expected {expected_comparison_count} paired comparisons, found {len(comparisons)}")
    adjusted = holm_bonferroni_adjust([row["p_value"] for row in comparisons])
    for row, adjusted_p in zip(comparisons, adjusted):
        row["p_value_holm_adjusted"] = float(adjusted_p)
        row["statistically_significant_after_holm"] = adjusted_p < 0.05
        if not 0.0 <= adjusted_p <= 1.0:
            raise AssertionError("Holm-adjusted p-value is outside [0, 1]")
    ordered_adjusted = [
        row["p_value_holm_adjusted"]
        for row in sorted(comparisons, key=lambda row: row["p_value"])
    ]
    if any(left > right for left, right in zip(ordered_adjusted, ordered_adjusted[1:])):
        raise AssertionError("Holm-adjusted p-values are not monotone in sorted p-value order")

    statistics = {
        "pairs": comparisons,
        "sample_size": len(query_ids),
        "bootstrap_resamples_per_comparison": BOOTSTRAP_ITERATIONS,
        "confidence_level": 0.95,
        "correction": "Holm-Bonferroni",
        "comparison_family_size": len(comparisons),
        "paired_test": "two-sided Wilcoxon signed-rank test",
    }
    quality_efficiency = {}
    quality_comparisons = []
    for method in ("dense", "hybrid"):
        for config in [
            f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS
        ] + [f"{method}_{strategy}_adaptive" for strategy in ADAPTIVE_STRATEGIES]:
            candidates = candidate_summary[config]["average_candidates_processed"]
            inference_seconds = performance_summary[config]["cross_encoder_inference"]["mean_seconds"]
            end_to_end_seconds = performance_summary[config]["end_to_end_latency"]["mean_seconds"]
            ndcg = aggregates[config]["ndcg@10"]
            quality_efficiency[config] = {
                "ndcg_at_10": ndcg,
                "average_candidates_processed": candidates,
                "ndcg_per_processed_candidate": ndcg / candidates if candidates else 0.0,
                "ndcg_per_cross_encoder_second": ndcg / inference_seconds if inference_seconds else 0.0,
                "ndcg_per_end_to_end_second": ndcg / end_to_end_seconds if end_to_end_seconds else 0.0,
            }
        for adaptive in (f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive"):
            for depth in FIXED_DEPTHS:
                baseline = f"{method}_fixed_top_{depth}"
                adaptive_candidates = candidate_summary[adaptive]["average_candidates_processed"]
                baseline_candidates = candidate_summary[baseline]["average_candidates_processed"]
                adaptive_inference = performance_summary[adaptive]["cross_encoder_inference"]["mean_seconds"]
                baseline_inference = performance_summary[baseline]["cross_encoder_inference"]["mean_seconds"]
                adaptive_latency = performance_summary[adaptive]["end_to_end_latency"]["mean_seconds"]
                baseline_latency = performance_summary[baseline]["end_to_end_latency"]["mean_seconds"]
                quality_comparisons.append({
                    "retrieval_method": method,
                    "adaptive_configuration": adaptive,
                    "fixed_configuration": baseline,
                    "quality_differences": {
                        metric: aggregates[adaptive][metric] - aggregates[baseline][metric]
                        for metric in metrics
                    },
                    "ndcg_at_10_difference": aggregates[adaptive]["ndcg@10"] - aggregates[baseline]["ndcg@10"],
                    "relative_candidate_processing_reduction": 1.0 - adaptive_candidates / baseline_candidates if baseline_candidates else 0.0,
                    "relative_cross_encoder_time_reduction": 1.0 - adaptive_inference / baseline_inference if baseline_inference else 0.0,
                    "relative_end_to_end_latency_reduction": 1.0 - adaptive_latency / baseline_latency if baseline_latency else 0.0,
                    "adaptive_candidates_processed_mean": adaptive_candidates,
                    "fixed_candidates_processed_mean": baseline_candidates,
                    "adaptive_cross_encoder_seconds_mean": adaptive_inference,
                    "fixed_cross_encoder_seconds_mean": baseline_inference,
                    "adaptive_end_to_end_seconds_mean": adaptive_latency,
                    "fixed_end_to_end_seconds_mean": baseline_latency,
                })
    _atomic_json(run_dir / "aggregate_metrics.json", aggregates)
    _atomic_json(run_dir / "aggregate_metrics.csv", aggregates)
    _atomic_json(run_dir / "candidate_depth_distribution.json", candidate_summary)
    _atomic_json(run_dir / "performance_metrics.json", performance_summary)
    _atomic_json(run_dir / "statistical_tests.json", statistics)
    _atomic_json(run_dir / "confidence_intervals.json", {
        "confidence_level": 0.95,
        "bootstrap_resamples": BOOTSTRAP_ITERATIONS,
        "comparisons": comparisons,
    })
    _atomic_json(run_dir / "quality_efficiency_tradeoff.json", quality_efficiency)
    _atomic_json(run_dir / "quality_efficiency_comparison.json", quality_comparisons)

    with (run_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["configuration", "retrieval_method", *metrics]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config, values in aggregates.items():
            method = "dense" if config.startswith("dense") else "hybrid"
            writer.writerow({"configuration": config, "retrieval_method": method, **values})
    with (run_dir / "candidate_depth_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "configuration", "average_candidates_retrieved", "average_candidates_processed",
            "average_unique_candidates_seen_across_stages",
            "average_candidates_cross_encoder_scored", "average_backend_results_returned_across_stages",
            "average_expansion_stages", "percentage_stopping_at_10",
            "percentage_stopping_at_20", "percentage_stopping_at_50",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config, values in candidate_summary.items():
            writer.writerow({"configuration": config, **values})
    with (run_dir / "quality_efficiency_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "retrieval_method", "adaptive_configuration", "fixed_configuration", "metric",
            "quality_difference_adaptive_minus_fixed", "relative_candidate_processing_reduction",
            "relative_cross_encoder_time_reduction", "relative_end_to_end_latency_reduction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in quality_comparisons:
            for metric, difference in comparison["quality_differences"].items():
                writer.writerow({
                    "retrieval_method": comparison["retrieval_method"],
                    "adaptive_configuration": comparison["adaptive_configuration"],
                    "fixed_configuration": comparison["fixed_configuration"],
                    "metric": metric,
                    "quality_difference_adaptive_minus_fixed": difference,
                    "relative_candidate_processing_reduction": comparison["relative_candidate_processing_reduction"],
                    "relative_cross_encoder_time_reduction": comparison["relative_cross_encoder_time_reduction"],
                    "relative_end_to_end_latency_reduction": comparison["relative_end_to_end_latency_reduction"],
                })
    with (run_dir / "quality_vs_latency.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["configuration", "retrieval_method", "ndcg@10", "mean_end_to_end_seconds", "mean_cross_encoder_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config, values in aggregates.items():
            method = "dense" if config.startswith("dense") else "hybrid"
            writer.writerow({
                "configuration": config,
                "retrieval_method": method,
                "ndcg@10": values["ndcg@10"],
                "mean_end_to_end_seconds": performance_summary[config]["end_to_end_latency"]["mean_seconds"],
                "mean_cross_encoder_seconds": performance_summary[config]["cross_encoder_inference"]["mean_seconds"],
            })
    return aggregates, candidate_summary, {
        "statistics": statistics,
        "quality_efficiency": quality_efficiency,
        "quality_efficiency_comparisons": quality_comparisons,
    }


def _generate_figures(run_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregates = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads((run_dir / "candidate_depth_distribution.json").read_text(encoding="utf-8"))
    performance = json.loads((run_dir / "performance_metrics.json").read_text(encoding="utf-8"))
    statistics = json.loads((run_dir / "statistical_tests.json").read_text(encoding="utf-8"))
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, method in zip(axes, ("dense", "hybrid")):
        configs = [f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS] + [
            f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive"
        ]
        labels = ["Fixed 10", "Fixed 20", "Fixed 50", "Score-gap", "Confidence"]
        axis.bar(labels, [aggregates[c]["ndcg@10"] for c in configs])
        axis.set_title(f"{method.title()}: nDCG@10")
        axis.set_ylabel("Mean development-set nDCG@10")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Fixed-depth versus adaptive retrieval quality")
    fig.tight_layout()
    fig.savefig(figures / "fixed_vs_adaptive_quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, method in zip(axes, ("dense", "hybrid")):
        configs = [f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS] + [
            f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive"
        ]
        labels = ["Fixed 10", "Fixed 20", "Fixed 50", "Score-gap", "Confidence"]
        axis.bar(labels, [candidate_summary[c]["average_candidates_processed"] for c in configs])
        axis.set_title(f"{method.title()} candidate processing")
        axis.set_ylabel("Average candidates processed")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "candidate_processing_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, method in zip(axes, ("dense", "hybrid")):
        configs = [f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive"]
        labels = ["10", "20", "50"]
        width = 0.35
        x = list(range(len(labels)))
        for offset, config in zip((-width / 2, width / 2), configs):
            summary = candidate_summary[config]
            percentages = [
                summary["percentage_stopping_at_10"],
                summary["percentage_stopping_at_20"],
                summary["percentage_stopping_at_50"],
            ]
            axis.bar(
                [position + offset for position in x],
                percentages,
                width,
                label="Score-gap" if config.endswith("score_gap_adaptive") else "Confidence",
            )
        axis.set_xticks(x, labels)
        axis.set_title(f"{method.title()} stopping depth")
        axis.set_xlabel("Final candidate depth")
        axis.set_ylabel("Queries (%)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "adaptive_stopping_depth_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method in ("dense", "hybrid"):
        configs = [f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS] + [
            f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive"
        ]
        for config in configs:
            axis.scatter(
                performance[config]["end_to_end_latency"]["mean_seconds"],
                aggregates[config]["ndcg@10"],
                label=config,
            )
    axis.set_xlabel("Mean end-to-end latency (seconds/query)")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Quality versus end-to-end latency")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "quality_vs_end_to_end_latency.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method in ("dense", "hybrid"):
        configs = [f"{method}_fixed_top_{depth}" for depth in FIXED_DEPTHS] + [
            f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive"
        ]
        for config in configs:
            axis.scatter(
                performance[config]["cross_encoder_inference"]["mean_seconds"],
                aggregates[config]["ndcg@10"],
                label=config,
            )
    axis.set_xlabel("Mean cross-encoder inference time (seconds/query)")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Quality versus cross-encoder inference time")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "quality_vs_inference_time.png", dpi=180)
    plt.close(fig)

    tradeoff = json.loads((run_dir / "quality_efficiency_tradeoff.json").read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        list(tradeoff),
        [tradeoff[c]["ndcg_per_processed_candidate"] for c in tradeoff],
    )
    axis.set_ylabel("nDCG@10 per processed candidate")
    axis.set_title("Quality-efficiency trade-off")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "quality_efficiency_tradeoff.png", dpi=180)
    plt.close(fig)

    principal = [
        row for row in statistics["pairs"]
        if row["metric"] == "ndcg@10"
        and row["fixed_comparison"].endswith("fixed_top_10")
    ]
    if principal:
        fig, axis = plt.subplots(figsize=(9, 5))
        positions = list(range(len(principal)))
        differences = [row["mean_difference_adaptive_minus_fixed"] for row in principal]
        lower = [row["bootstrap_ci95"][0] for row in principal]
        upper = [row["bootstrap_ci95"][1] for row in principal]
        axis.errorbar(
            differences,
            positions,
            xerr=[
                [d - l for d, l in zip(differences, lower)],
                [u - d for d, u in zip(differences, upper)],
            ],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_yticks(
            positions,
            [
                f"{row['retrieval_method']} "
                f"{'score-gap' if 'score_gap' in row['adaptive_configuration'] else 'confidence'}"
                for row in principal
            ],
        )
        axis.set_xlabel("nDCG@10 difference (adaptive - fixed Top-10)")
        axis.set_title("Paired bootstrap 95% confidence intervals")
        axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "principal_confidence_intervals.png", dpi=180)
        plt.close(fig)

    return [
        "fixed_vs_adaptive_quality.png",
        "candidate_processing_comparison.png",
        "adaptive_stopping_depth_distribution.png",
        "quality_vs_end_to_end_latency.png",
        "quality_vs_inference_time.png",
        "quality_efficiency_tradeoff.png",
        "principal_confidence_intervals.png",
    ]


def _validate(
    run_dir: Path,
    expected_query_ids: set[str] | None = None,
    corpus_ids: set[str] | None = None,
    qrels: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    from src.evaluation.retrieval_metrics import compute_retrieval_metrics
    from src.retrieval.adaptive_retriever import ConfidenceAdaptiveStrategy, ScoreGapAdaptiveStrategy

    metrics = json.loads((run_dir / "per_query_metrics.json").read_text(encoding="utf-8"))
    rankings = json.loads((run_dir / "per_query_rankings.json").read_text(encoding="utf-8"))
    decisions = json.loads((run_dir / "adaptive_retrieval_decisions.json").read_text(encoding="utf-8"))
    aggregates = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    query_ids = sorted({row["query_id"] for row in metrics})
    def unique_keys(rows: list[dict[str, Any]], artifact: str) -> set[tuple[str, str, str]]:
        keys = [
            (str(row["query_id"]), str(row["retrieval_method"]), str(row["configuration"]))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            raise AssertionError(f"Duplicate query-configuration records in {artifact}")
        return set(keys)

    config_keys = unique_keys(metrics, "metrics")
    ranking_keys = unique_keys(rankings, "rankings")
    decision_keys = unique_keys(decisions, "decisions")
    expected = 300 * 10
    expected_query_ids = expected_query_ids or set(query_ids)
    if len(query_ids) != 300 or set(query_ids) != expected_query_ids:
        raise AssertionError(f"Expected the exact 300 development query IDs, found {len(query_ids)}")
    if len(metrics) != expected or len(rankings) != expected or len(decisions) != expected:
        raise AssertionError("Missing query-configuration records")
    expected_keys = {
        (query_id, method, config)
        for query_id in expected_query_ids
        for method in ("dense", "hybrid")
        for config in (
            f"{method}_fixed_top_10",
            f"{method}_fixed_top_20",
            f"{method}_fixed_top_50",
            f"{method}_score_gap_adaptive",
            f"{method}_confidence_adaptive",
        )
    }
    if config_keys != expected_keys:
        raise AssertionError("Metric records do not cover the expected query/configuration pairs")
    if config_keys != ranking_keys or config_keys != decision_keys:
        raise AssertionError("Metric, ranking, and decision configuration keys differ")
    performance = json.loads((run_dir / "performance_measurements.json").read_text(encoding="utf-8"))
    if unique_keys(performance, "performance") != expected_keys:
        raise AssertionError("Performance records do not cover the expected query/configuration pairs")
    ranking_by_key = {
        (row["query_id"], row["retrieval_method"], row["configuration"]): row
        for row in rankings
    }
    performance_by_key = {
        (row["query_id"], row["retrieval_method"], row["configuration"]): row
        for row in performance
    }
    metrics_by_key = {
        (row["query_id"], row["retrieval_method"], row["configuration"]): row
        for row in metrics
    }
    configured_thresholds = {
        "score_gap": SCORE_GAP_THRESHOLD,
        "confidence": CONFIDENCE_THRESHOLD,
    }
    for row in decisions:
        candidate_ids = row["candidate_ids"]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise AssertionError("Duplicate candidate IDs in adaptive/fixed decision record")
        if row["number_candidates_processed"] != len(candidate_ids):
            raise AssertionError("Candidate processed count does not match candidate IDs")
        if row["number_candidates_retrieved"] != len(candidate_ids):
            raise AssertionError("Unique candidates retrieved do not match candidate IDs")
        if row.get("number_candidates_cross_encoder_scored") != len(candidate_ids):
            raise AssertionError("Cross-encoder candidate count does not match candidate IDs")
        if corpus_ids is not None and not set(candidate_ids).issubset(corpus_ids):
            raise AssertionError("Candidate ID is absent from the corpus")
        scores = row.get("candidate_retrieval_scores", {})
        if set(scores) != set(candidate_ids) or any(not math.isfinite(float(value)) for value in scores.values()):
            raise AssertionError("Candidate retrieval scores do not match the candidate IDs")
        stages = row.get("stage_records", [])
        if not stages or [stage["requested_depth"] for stage in stages] != row["stage_depths"]:
            raise AssertionError("Retrieval stage records do not match the recorded depths")
        previous_ids: list[str] = []
        seen_ids: set[str] = set()
        for stage in stages:
            stage_ids = stage["candidate_ids"]
            if len(stage_ids) != len(set(stage_ids)):
                raise AssertionError("A retrieval stage contains duplicate candidate IDs")
            if corpus_ids is not None and not set(stage_ids).issubset(corpus_ids):
                raise AssertionError("A retrieval stage returned an ID absent from the corpus")
            if int(stage["backend_result_count"]) != len(stage_ids):
                raise AssertionError("Stage result count does not match returned candidate IDs")
            added_ids = [candidate_id for candidate_id in stage_ids if candidate_id not in seen_ids]
            if added_ids != stage["added_candidate_ids"] or len(added_ids) != stage["new_candidate_count"]:
                raise AssertionError("Incremental candidate additions do not match stage records")
            if len(stage["retrieval_scores"]) != len(stage_ids):
                raise AssertionError("Retrieval scores do not match stage candidate IDs")
            latency = float(stage["retrieval_latency_seconds"])
            if not math.isfinite(latency) or latency < 0:
                raise AssertionError("Retrieval stage latency must be finite and non-negative")
            if row["adaptive_strategy"] != "fixed" and stage["requested_depth"] < 50:
                signal_key = f"after_{stage['requested_depth']}"
                signal = row["stopping_signals"].get(signal_key)
                if signal is None:
                    raise AssertionError(f"Missing adaptive stopping signal at depth {stage['requested_depth']}")
                threshold = configured_thresholds[row["adaptive_strategy"]]
                if float(signal["threshold"]) != threshold:
                    raise AssertionError("Recorded stopping threshold differs from experiment configuration")
                strategy_type = (
                    ScoreGapAdaptiveStrategy
                    if row["adaptive_strategy"] == "score_gap"
                    else ConfidenceAdaptiveStrategy
                )
                strategy = strategy_type(
                    threshold=threshold, min_depth=10, max_depth=50, fallback_depth=20
                )
                expected_normalized = strategy.normalize_scores(stage["retrieval_scores"])
                if len(signal["normalized_scores"]) != len(expected_normalized) or any(
                    abs(float(actual) - expected) > 1e-12
                    for actual, expected in zip(signal["normalized_scores"], expected_normalized)
                ):
                    raise AssertionError("Recorded normalized stopping scores are inconsistent")
                expected_stop = bool(strategy.should_stop(stage["retrieval_scores"]))
                if bool(signal["should_stop"]) != expected_stop:
                    raise AssertionError("Recorded stopping decision differs from the configured strategy")
                if stage["requested_depth"] != stages[-1]["requested_depth"] and expected_stop:
                    raise AssertionError("Adaptive retriever expanded after satisfying its stopping rule")
                if stage["requested_depth"] == stages[-1]["requested_depth"] and stage["requested_depth"] < 50 and not expected_stop:
                    raise AssertionError("Adaptive retriever stopped before max depth without satisfying its rule")
            previous_ids = list(stage_ids)
            seen_ids.update(stage_ids)
        if previous_ids != candidate_ids:
            raise AssertionError("Final candidate IDs differ from the final retrieval stage")
        if (
            int(row["number_unique_candidates_seen_across_stages"]) != len(seen_ids)
            or set(row["unique_candidate_ids_seen_across_stages"]) != seen_ids
        ):
            raise AssertionError("Unique candidates seen across retrieval stages are inconsistent")
        if sum(int(stage["backend_result_count"]) for stage in stages) != row["number_candidate_results_returned_total"]:
            raise AssertionError("Cumulative backend candidate count is inconsistent")
        ranking = ranking_by_key[(row["query_id"], row["retrieval_method"], row["configuration"])]["rankings"]
        ranked_ids = [str(item["document_id"]) for item in ranking]
        if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != set(candidate_ids):
            raise AssertionError("Ranking records do not cover each cross-encoder candidate exactly once")
        if [int(item["rank"]) for item in ranking] != list(range(1, len(ranking) + 1)):
            raise AssertionError("Ranking records do not have contiguous ranks")
        key = (row["query_id"], row["retrieval_method"], row["configuration"])
        performance_row = performance_by_key[key]
        for latency_key in (
            "retrieval_latency_seconds",
            "cross_encoder_inference_seconds",
            "reranking_latency_seconds",
            "end_to_end_latency_seconds",
        ):
            value = float(performance_row[latency_key])
            if not math.isfinite(value) or value < 0:
                raise AssertionError(f"Invalid performance measurement: {latency_key}")
        if row["adaptive_strategy"] == "fixed":
            if (
                row["stage_depths"] != [row["final_candidate_depth"]]
                or row["number_of_expansion_stages"] != 0
                or row["stopped_early"] is not False
            ):
                raise AssertionError("Fixed-depth configuration has invalid stage accounting")
        else:
            if row["stage_depths"] not in ([10], [10, 20], [10, 20, 50]):
                raise AssertionError("Adaptive retrieval did not expand progressively through configured depths")
            if row["number_of_expansion_stages"] != len(row["stage_depths"]) - 1:
                raise AssertionError("Adaptive expansion-stage count is inconsistent")
            if row["number_candidate_results_returned_total"] < len(candidate_ids):
                raise AssertionError("Cumulative backend results cannot be fewer than unique candidates")
            if row["stopping_depth"] != row["final_candidate_depth"]:
                raise AssertionError("Adaptive stopping depth differs from final candidate depth")
            if bool(row["stopped_early"]) != (row["final_candidate_depth"] < 50):
                raise AssertionError("Adaptive early-stop flag differs from its final candidate depth")
        if (
            int(performance_row["number_candidates_processed"]) != len(candidate_ids)
            or int(performance_row["number_candidates_cross_encoder_scored"]) != len(candidate_ids)
            or int(performance_row["number_candidate_results_returned_total"])
            != int(row["number_candidate_results_returned_total"])
        ):
            raise AssertionError("Performance candidate counts differ from retrieval decision records")
        expected_end_to_end = (
            float(performance_row["retrieval_latency_seconds"])
            + float(performance_row["reranking_latency_seconds"])
        )
        if abs(float(performance_row["end_to_end_latency_seconds"]) - expected_end_to_end) > 1e-9:
            raise AssertionError("End-to-end latency does not equal retrieval plus reranking latency")
        if qrels is not None:
            recomputed_metrics = compute_retrieval_metrics(ranked_ids[:10], qrels[row["query_id"]], (10,))
            metric_row = metrics_by_key[key]
            for name, value in recomputed_metrics.items():
                if abs(float(metric_row[name]) - float(value)) > 1e-12:
                    raise AssertionError(f"Per-query metric mismatch: {row['configuration']} {name}")
        if row["adaptive_strategy"] != "fixed":
            stage_depths = row["stage_depths"]
            if stage_depths and stage_depths != sorted(set(stage_depths)):
                raise AssertionError("Adaptive expansion stages are not ordered")
            if stage_depths and stage_depths != [10] and stage_depths[0] != 10:
                raise AssertionError("Adaptive retrieval did not start at depth 10")
            if row["stopping_depth"] not in (10, 20, 50):
                raise AssertionError("Adaptive stopping depth is invalid")
    if set(aggregates) != {
        f"{method}_fixed_top_{depth}"
        for method in ("dense", "hybrid")
        for depth in FIXED_DEPTHS
    } | {
        f"{method}_{strategy}_adaptive"
        for method in ("dense", "hybrid")
        for strategy in ADAPTIVE_STRATEGIES
    }:
        raise AssertionError("Aggregate metrics are missing or contain unexpected configurations")
    for config, values in aggregates.items():
        method = "dense" if config.startswith("dense") else "hybrid"
        for metric in _metric_names():
            rows = [
                row for row in metrics
                if row["retrieval_method"] == method
                and row["configuration"] == config
            ]
            recomputed_value = mean(float(row[metric]) for row in rows)
            if abs(recomputed_value - float(values[metric])) > 1e-12:
                raise AssertionError(f"Aggregate mismatch: {config} {metric}")
    candidate_summary = json.loads((run_dir / "candidate_depth_distribution.json").read_text(encoding="utf-8"))
    performance_summary = json.loads((run_dir / "performance_metrics.json").read_text(encoding="utf-8"))
    for config in aggregates:
        method = "dense" if config.startswith("dense") else "hybrid"
        selected_decisions = [
            row for row in decisions
            if row["retrieval_method"] == method and row["configuration"] == config
        ]
        selected_performance = [
            row for row in performance
            if row["retrieval_method"] == method and row["configuration"] == config
        ]
        expected_candidates = mean(float(row["number_candidates_processed"]) for row in selected_decisions)
        if abs(candidate_summary[config]["average_candidates_processed"] - expected_candidates) > 1e-12:
            raise AssertionError(f"Candidate aggregate mismatch: {config}")
        expected_unique_seen = mean(
            float(row["number_unique_candidates_seen_across_stages"])
            for row in selected_decisions
        )
        if abs(candidate_summary[config]["average_unique_candidates_seen_across_stages"] - expected_unique_seen) > 1e-12:
            raise AssertionError(f"Distinct stage-candidate aggregate mismatch: {config}")
        expected_expansions = mean(float(row["number_of_expansion_stages"]) for row in selected_decisions)
        if abs(candidate_summary[config]["average_expansion_stages"] - expected_expansions) > 1e-12:
            raise AssertionError(f"Expansion-stage aggregate mismatch: {config}")
        for depth in (10, 20, 50):
            expected_percentage = 100.0 * sum(
                row["stopping_depth"] == depth for row in selected_decisions
            ) / len(selected_decisions)
            actual_percentage = candidate_summary[config][f"percentage_stopping_at_{depth}"]
            if abs(actual_percentage - expected_percentage) > 1e-12:
                raise AssertionError(f"Stopping-depth distribution mismatch: {config}, depth {depth}")
        for summary_key, row_key in (
            ("retrieval_latency", "retrieval_latency_seconds"),
            ("cross_encoder_inference", "cross_encoder_inference_seconds"),
            ("reranking_latency", "reranking_latency_seconds"),
            ("end_to_end_latency", "end_to_end_latency_seconds"),
        ):
            expected_latency = _latency([float(row[row_key]) for row in selected_performance])
            actual_latency = performance_summary[config][summary_key]
            for statistic in ("mean_seconds", "median_seconds", "p95_seconds"):
                if abs(float(actual_latency[statistic]) - float(expected_latency[statistic])) > 1e-9:
                    raise AssertionError(f"Latency aggregate mismatch: {config}, {summary_key}, {statistic}")
    statistics = json.loads((run_dir / "statistical_tests.json").read_text(encoding="utf-8"))
    from src.evaluation.statistical_tests import holm_bonferroni_adjust

    p_values = [float(row["p_value"]) for row in statistics["pairs"]]
    corrected = holm_bonferroni_adjust(p_values)
    if len(p_values) != 60 or statistics["comparison_family_size"] != len(p_values):
        raise AssertionError("Statistical comparison family size is not 60")
    if any(
        abs(float(row["p_value_holm_adjusted"]) - expected_adjusted) > 1e-12
        for row, expected_adjusted in zip(statistics["pairs"], corrected)
    ):
        raise AssertionError("Reported Holm-adjusted p-values do not match recomputation")
    sorted_adjusted = [
        row["p_value_holm_adjusted"]
        for row in sorted(statistics["pairs"], key=lambda item: item["p_value"])
    ]
    if any(left > right for left, right in zip(sorted_adjusted, sorted_adjusted[1:])):
        raise AssertionError("Holm-adjusted p-values fail monotonicity validation")
    return {
        "status": "passed",
        "query_count": len(query_ids),
        "metric_record_count": len(metrics),
        "ranking_record_count": len(rankings),
        "decision_record_count": len(decisions),
        "performance_record_count": len(performance),
        "duplicate_query_configuration_records": False,
        "candidate_ids_unique": True,
        "adaptive_incremental_expansion_verified": True,
        "aggregate_metrics_match_per_query": True,
        "statistical_correction_valid": True,
        "official_test_claim_file_opened": False,
        "protected_runs_checked": ["run_0001", "run_0002", "run_0003", "run_0004", "run_0005", "run_0006"],
    }


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


def _compare_experiment_a(
    run_dir: Path,
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = RUN_ROOT / "run_0006" / "per_query_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Experiment A metrics are required for comparison: {path}")
    experiment_a_rows = json.loads(path.read_text(encoding="utf-8"))
    if len(experiment_a_rows) != 300 * 2 * len(FIXED_DEPTHS) * 2:
        raise AssertionError("Experiment A per-query metrics do not contain the expected 3,600 records")
    experiment_a = {
        (
            str(row["query_id"]),
            str(row["base_method"]),
            int(row["candidate_pool_size"]),
            str(row["condition"]),
        ): row
        for row in experiment_a_rows
    }
    experiment_b = {
        (str(row["query_id"]), str(row["retrieval_method"]), str(row["configuration"])): row
        for row in metric_rows
    }
    comparisons = []
    for method in ("dense", "hybrid"):
        for depth in FIXED_DEPTHS:
            differences = {metric: [] for metric in _metric_names()}
            for query_id in sorted({str(row["query_id"]) for row in metric_rows}):
                old = experiment_a.get((query_id, method, depth, "reranked"))
                new = experiment_b.get((query_id, method, f"{method}_fixed_top_{depth}"))
                if old is None or new is None:
                    raise AssertionError(
                        f"Experiment A/B comparison is missing {method} depth {depth}, query {query_id}"
                    )
                for metric in _metric_names():
                    differences[metric].append(float(new[metric]) - float(old[metric]))
            summaries = {
                metric: {
                    "experiment_a_mean": mean(float(row[metric]) for row in experiment_a_rows
                        if row["base_method"] == method
                        and int(row["candidate_pool_size"]) == depth
                        and row["condition"] == "reranked"),
                    "experiment_b_mean": mean(
                        float(row[metric]) for row in metric_rows
                        if row["retrieval_method"] == method
                        and row["configuration"] == f"{method}_fixed_top_{depth}"
                    ),
                    "mean_difference_experiment_b_minus_a": mean(values),
                    "maximum_absolute_per_query_difference": max(abs(value) for value in values),
                    "per_query_match_within_1e-12": all(abs(value) <= 1e-12 for value in values),
                }
                for metric, values in differences.items()
            }
            comparisons.append({
                "retrieval_method": method,
                "candidate_depth": depth,
                "query_count": 300,
                "metrics": summaries,
                "all_metrics_match_within_1e-12": all(
                    item["per_query_match_within_1e-12"] for item in summaries.values()
                ),
            })
    result = {
        "experiment_a_run_id": "run_0006",
        "source": "results/runs/run_0006/per_query_metrics.json",
        "comparison_is_descriptive_only": True,
        "comparisons": comparisons,
    }
    _atomic_json(run_dir / "experiment_a_comparison.json", result)
    return result


def _build_config(
    run_id: str,
    *,
    status: str,
    resumed: bool,
    git_commit: str | None,
    code_sha256: str,
    dataset_manifest_sha256: str,
    dev_claims_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment": "Phase 3 Experiment B",
        "run_id": run_id,
        "status": status,
        "dataset": "BEIR SciFact",
        "dataset_split": "official SciFact development",
        "query_count": 300,
        "dev_claims_path": "data/processed/raw_scifact/data/claims_dev.jsonl",
        "dev_claims_sha256": dev_claims_sha256,
        "dataset_manifest": "manifests/dataset_manifest.json",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "qrels": "BeIR/scifact-qrels test split mapped to official development",
        "qrels_selection": "development evaluation only; no qrels used for stopping decisions",
        "retrieval_methods": ["dense", "hybrid"],
        "fixed_depths": list(FIXED_DEPTHS),
        "adaptive_strategies": {
            "score_gap": {
                "threshold": SCORE_GAP_THRESHOLD,
                "signal": "min-max-normalized top-score minus second-score",
                "rule": "stop when signal >= threshold; otherwise expand 10 -> 20 -> 50",
            },
            "confidence": {
                "threshold": CONFIDENCE_THRESHOLD,
                "signal": "z-like distance of top normalized retrieval score from score-distribution mean",
                "rule": "stop when signal >= threshold; otherwise expand 10 -> 20 -> 50",
            },
        },
        "initial_depth": 10,
        "maximum_depth": 50,
        "evaluation_cutoff": 10,
        "hybrid_rrf_constant": RRF_CONSTANT,
        "reranker": {
            "model_name": MODEL_NAME,
            "device": "cpu",
            "batch_size": BATCH_SIZE,
            "strict_model_loading": True,
            "model_load_seconds": None,
        },
        "bootstrap_resamples": BOOTSTRAP_ITERATIONS,
        "seed": SEED,
        "comparison_family_size": 2 * len(ADAPTIVE_STRATEGIES) * len(FIXED_DEPTHS) * len(_metric_names()),
        "code_sha256": code_sha256,
        "git_commit": git_commit,
        "official_test_claim_file_opened": False,
        "resumed": resumed,
    }


def _write_report(
    run_dir: Path,
    aggregates: dict[str, Any],
    candidate_summary: dict[str, Any],
    performance: dict[str, Any],
    extra: dict[str, Any],
    validation: dict[str, Any],
    smoke: dict[str, Any],
    figures: list[str],
    test_result: dict[str, Any],
    experiment_a_comparison: dict[str, Any],
) -> Path:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    statistics = extra["statistics"]
    tradeoff = extra["quality_efficiency"]
    comparisons = extra["quality_efficiency_comparisons"]
    best_adaptive = max(
        (config for config in tradeoff if config.endswith("_adaptive")),
        key=lambda config: tradeoff[config]["ndcg_per_end_to_end_second"],
    )
    best_method = "dense" if best_adaptive.startswith("dense_") else "hybrid"
    best_strategy = "score-gap" if "score_gap" in best_adaptive else "confidence"
    best_fixed_50 = f"{best_method}_fixed_top_50"
    best_vs_50 = next(
        row for row in comparisons
        if row["adaptive_configuration"] == best_adaptive
        and row["fixed_configuration"] == best_fixed_50
    )
    best_stat = next(
        row for row in statistics["pairs"]
        if row["adaptive_configuration"] == best_adaptive
        and row["fixed_comparison"] == best_fixed_50
        and row["metric"] == "ndcg@10"
    )
    best_ci = best_stat["bootstrap_ci95"]
    lines = [
        "# Phase 3 — Experiment B: Adaptive Retrieval",
        "",
        "> Official SciFact development-set results only. This report is not a held-out test-set result.",
        "",
        f"- Run: `{run_dir.name}` (`{run_dir.as_posix()}`)",
        f"- Git commit: `{config['git_commit'] or 'unavailable'}`; runner SHA-256: `{config['code_sha256']}`.",
        "- Dataset: 300 official SciFact development claims.",
        "- Qrels: BEIR `scifact-qrels` `test` split, established by the project manifest to align with official development claims.",
        "- Dataset manifest: `manifests/dataset_manifest.json`; official held-out test claims and labels were not accessed.",
        "- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, batch size 16.",
        f"- Smoke test: {smoke['query_count']} queries, {smoke['system_query_count']} method/query runs; measured estimated runtime {smoke['estimated_full_end_to_end_minutes']:.1f} minutes.",
        f"- Integrity tests: {test_result['status']}; `tests/test_dataset_validation.py` was intentionally excluded because its split loader reads every official claim split, including the prohibited held-out claims file.",
        "",
        "## Research objective",
        "",
        "Test whether score-gap or confidence-based adaptive retrieval can reduce the candidate pool and cross-encoder work while preserving top-10 reranking quality, compared with fixed Top-10, Top-20, and Top-50 retrieval. The stopping rules are unsupervised and are not calibrated against evaluation qrels.",
        "",
        "## Experimental design and methodology",
        "",
        "- All 300 official development claims were paired across 10 configurations: Dense and Hybrid × fixed depths 10/20/50 and adaptive Score-gap/Confidence.",
        "- Fixed configurations retrieve and rerank the full candidate pool. Adaptive configurations request depth 10 first and issue a new depth-20 or depth-50 retrieval only after the current score signal fails its stopping rule.",
        "- The retrieval API returns a ranked prefix on each expansion. Each stage records its returned prefix, newly added IDs, cumulative backend results, and measured retrieval latency; repeated prefix work is not counted as new unique candidates.",
        "- Score-gap stopping threshold: 0.10 on the min-max-normalized difference between the two leading retrieval scores. Confidence stopping threshold: 0.25 on the leading normalized score's z-like distance from the normalized score distribution mean. Stop at 10 or 20 when the strategy signal meets its threshold; otherwise cap at 50.",
        "- Thresholds are the existing unsupervised values; qrels and per-query relevance metrics are not used in stopping decisions or threshold tuning.",
        "- Dense uses the project FAISS index; Hybrid uses BM25 + Dense reciprocal-rank fusion (RRF constant 60). The cross-encoder model is loaded once per process; all inference is CPU and batch size 16.",
        "- In Experiment B, Hybrid requests both component retrievers at the current candidate depth (10, 20, or 50); adaptive stages therefore recompute RRF with deeper component lists as they expand.",
        "- All configurations use the same reranker and final evaluation cutoff of 10. Metrics are Hit@10, Recall@10, MRR@10, MAP@10, and nDCG@10.",
        "- Per-query metrics, full reranked candidate lists, retrieval scores, expansion decisions, timing, checkpoint files, code hash, environment, and run metadata are stored in the run directory.",
        "",
        "## Main results",
        "",
        "| Configuration | Hit@10 | Recall@10 | MRR@10 | MAP@10 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("dense", "hybrid"):
        for config in (
            f"{method}_fixed_top_10", f"{method}_fixed_top_20", f"{method}_fixed_top_50",
            f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive",
        ):
            row = aggregates[config]
            lines.append(
                f"| {config} | {row['hit@10']:.6f} | {row['recall@10']:.6f} | "
                f"{row['mrr@10']:.6f} | {row['map@10']:.6f} | {row['ndcg@10']:.6f} |"
            )

    lines.extend([
        "",
        "## Adaptive depth and computational cost",
        "",
        "Latency is measured per query/configuration. Cross-encoder inference time covers the actual model `predict` call; end-to-end time is retrieval plus the complete reranking call. Adaptive candidate pools are counted once by unique candidate ID, and cumulative distinct IDs plus raw backend result counts expose overlap across later stages. For fixed baselines, the depth percentages denote the configured fixed depth rather than an early-stop decision.",
        "",
        "| Configuration | Avg final candidates retrieved/processed | Avg distinct IDs seen across stages | Avg backend results returned | Avg expansions | Stop 10 | Stop 20 | Stop 50 | Retrieval mean / median / P95 (s) | CE mean / median / P95 (s) | End-to-end mean / median / P95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in ("dense", "hybrid"):
        for config in (
            f"{method}_fixed_top_10", f"{method}_fixed_top_20", f"{method}_fixed_top_50",
            f"{method}_score_gap_adaptive", f"{method}_confidence_adaptive",
        ):
            summary = candidate_summary[config]
            perf = performance[config]
            lines.append(
                f"| {config} | {summary['average_candidates_processed']:.2f} | "
                f"{summary['average_unique_candidates_seen_across_stages']:.2f} | "
                f"{summary['average_backend_results_returned_across_stages']:.2f} | "
                f"{summary['average_expansion_stages']:.2f} | "
                f"{summary['percentage_stopping_at_10']:.1f}% | "
                f"{summary['percentage_stopping_at_20']:.1f}% | "
                f"{summary['percentage_stopping_at_50']:.1f}% | "
                f"{perf['retrieval_latency']['mean_seconds']:.4f} / "
                f"{perf['retrieval_latency']['median_seconds']:.4f} / "
                f"{perf['retrieval_latency']['p95_seconds']:.4f} | "
                f"{perf['cross_encoder_inference']['mean_seconds']:.4f} / "
                f"{perf['cross_encoder_inference']['median_seconds']:.4f} / "
                f"{perf['cross_encoder_inference']['p95_seconds']:.4f} | "
                f"{perf['end_to_end_latency']['mean_seconds']:.4f} / "
                f"{perf['end_to_end_latency']['median_seconds']:.4f} / "
                f"{perf['end_to_end_latency']['p95_seconds']:.4f} |"
            )

    lines.extend([
        "",
        "## Quality differences and efficiency savings",
        "",
        "Differences below are adaptive minus the same-method fixed baseline; savings are relative to that baseline. Full differences for all five quality metrics are in `quality_efficiency_comparison.csv`.",
        "",
        "| Method | Adaptive | Fixed baseline | Δ nDCG@10 | Candidate reduction | CE time reduction | End-to-end reduction |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['retrieval_method']} | {row['adaptive_configuration']} | "
            f"{row['fixed_configuration']} | {row['ndcg_at_10_difference']:.6f} | "
            f"{row['relative_candidate_processing_reduction']:.1%} | "
            f"{row['relative_cross_encoder_time_reduction']:.1%} | "
            f"{row['relative_end_to_end_latency_reduction']:.1%} |"
        )

    lines.extend([
        "",
        "## Statistical analysis",
        "",
        f"- Paired bootstrap: {BOOTSTRAP_ITERATIONS:,} resamples, 95% confidence intervals, seed {SEED}.",
        "- Paired test: two-sided Wilcoxon signed-rank.",
        f"- Holm-Bonferroni comparison family: {statistics['comparison_family_size']} tests.",
        "- One family contains all adaptive-vs-fixed comparisons: 2 retrieval methods × 2 adaptive strategies × 3 fixed depths × 5 metrics = 60 paired tests. Bootstrap confidence intervals use paired per-query resampling; the reported effect size is paired Cohen's dz.",
        "- The Holm adjustment was recomputed and validated for range and monotonicity in sorted p-value order. Statistical significance uses adjusted p < 0.05.",
        "",
        "| Method | Adaptive | Fixed | nDCG@10 difference | 95% paired bootstrap CI | Paired Cohen's dz | Holm-adjusted p | Significant |",
        "|---|---|---|---:|---|---:|---:|---|",
    ])
    principal = [
        row for row in statistics["pairs"]
        if row["metric"] == "ndcg@10"
        and row["fixed_comparison"].endswith("fixed_top_10")
    ]
    for row in principal:
        row_ci = row["bootstrap_ci95"]
        lines.append(
            f"| {row['retrieval_method']} | {row['adaptive_configuration']} | "
            f"{row['fixed_comparison']} | {row['mean_difference_adaptive_minus_fixed']:.6f} | "
            f"[{row_ci[0]:.6f}, {row_ci[1]:.6f}] | "
            f"{row['paired_cohens_dz'] if row['paired_cohens_dz'] is not None else 'NA'} | "
            f"{row['p_value_holm_adjusted']:.6g} | "
            f"{'yes' if row['statistically_significant_after_holm'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Quality per unit of computation",
        "",
        "| Configuration | nDCG@10 | Candidates processed | nDCG/candidate | nDCG/CE second | nDCG/end-to-end second |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for config, row in tradeoff.items():
        lines.append(
            f"| {config} | {row['ndcg_at_10']:.6f} | "
            f"{row['average_candidates_processed']:.2f} | "
            f"{row['ndcg_per_processed_candidate']:.6f} | "
            f"{row['ndcg_per_cross_encoder_second']:.6f} | "
            f"{row['ndcg_per_end_to_end_second']:.6f} |"
        )

    lines.extend([
        "",
        "## Experiment A comparison (run_0006)",
        "",
        "This is a descriptive per-query comparison of Experiment B fixed reranking and Experiment A's nominally matching pool-size configuration. Differences are B minus A; exact matching is checked at the per-query level. Adaptive conditions are new and have no Experiment A counterpart.",
        "",
        "The Hybrid candidate sets are not identical between experiments: Experiment A retrieved BM25 and Dense component lists to depth 100 before truncating the fused ranking, whereas Experiment B requests each component at its fixed candidate depth (10, 20, or 50). The Hybrid metric differences below are therefore expected and must not be interpreted as a reranker-only change or an exact replication. Dense fixed-depth metrics match Experiment A exactly.",
        "",
        "| Method | Pool | A nDCG@10 | B nDCG@10 | Δ nDCG@10 | A Hit@10 | B Hit@10 | Δ Hit@10 | Exact all-metric per-query match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for item in experiment_a_comparison["comparisons"]:
        metrics = item["metrics"]
        ndcg = metrics["ndcg@10"]
        hit = metrics["hit@10"]
        lines.append(
            f"| {item['retrieval_method']} | {item['candidate_depth']} | "
            f"{ndcg['experiment_a_mean']:.6f} | {ndcg['experiment_b_mean']:.6f} | "
            f"{ndcg['mean_difference_experiment_b_minus_a']:.6f} | "
            f"{hit['experiment_a_mean']:.6f} | {hit['experiment_b_mean']:.6f} | "
            f"{hit['mean_difference_experiment_b_minus_a']:.6f} | "
            f"{item['all_metrics_match_within_1e-12']} |"
        )

    candidate_reduction = best_vs_50["relative_candidate_processing_reduction"]
    inference_reduction = best_vs_50["relative_cross_encoder_time_reduction"]
    end_to_end_reduction = best_vs_50["relative_end_to_end_latency_reduction"]
    conclusion = (
        f"The best observed adaptive nDCG-per-end-to-end-second configuration was `{best_adaptive}` "
        f"({tradeoff[best_adaptive]['ndcg_per_end_to_end_second']:.6f}), with nDCG@10 "
        f"{aggregates[best_adaptive]['ndcg@10']:.6f}. Against same-method fixed Top-50, it "
        f"processed {candidate_summary[best_adaptive]['average_candidates_processed']:.2f} versus "
        f"{candidate_summary[best_fixed_50]['average_candidates_processed']:.2f} candidates/query "
        f"({candidate_reduction:.1%} fewer), reduced mean cross-encoder inference time by "
        f"{inference_reduction:.1%}, and reduced mean end-to-end latency by {end_to_end_reduction:.1%}. "
        f"Its nDCG@10 difference versus fixed Top-50 was "
        f"{best_stat['mean_difference_adaptive_minus_fixed']:.6f} "
        f"(95% paired bootstrap CI [{best_ci[0]:.6f}, {best_ci[1]:.6f}], Holm-adjusted p="
        f"{best_stat['p_value_holm_adjusted']:.6g}). These measured results "
        f"{'show no statistically significant difference' if not best_stat['statistically_significant_after_holm'] else 'show a statistically significant difference'} "
        "for that comparison; lack of significance is not a formal equivalence or non-inferiority result. "
        "The evidence therefore supports the observed cost/quality trade-off above, but not a broader claim of quality equivalence."
    )
    lines.extend([
        "",
        "## Figures",
        "",
    ])
    for name in figures:
        lines.append(f"- `results/runs/{run_dir.name}/figures/{name}`")

    lines.extend([
        "",
        "## Limitations and threats to validity",
        "",
        "- This is a development-set experiment; it does not estimate final held-out test performance.",
        "- The score-gap and confidence thresholds are unsupervised heuristics, not calibrated probabilities. No formal non-inferiority margin was specified, so non-significance is not proof of quality retention.",
        "- Dense and Hybrid retrieval are evaluated with the existing corpus/index and fixed RRF constant; conclusions may not generalize to other corpora or retrieval backends.",
        "- Adaptive stages call the backend at successively larger depths. The returned ranked prefixes overlap, and the recorded stage work and latency expose this repeated backend work rather than treating it as free.",
        "- CPU inference latency depends on machine load, cache state, model/runtime versions, and batch behavior; smoke measurements are excluded from final aggregates.",
        "- Fixed and adaptive configurations are independently timed; candidate scores are not shared across configurations.",
        "",
        "## Reproducibility and artifacts",
        "",
        "The run configuration records the dataset split, qrels mapping, thresholds, models, code hash, Git commit when available, and runtime environment. Per-query checkpoints are stored under `checkpoints/queries/` and are validated before reuse.",
        "",
        "Start a new run (the run manager allocates the next unique `run_NNNN`):",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe experiments\\run_adaptive_retrieval_experiment.py",
        "```",
        "",
        "Resume this run after interruption:",
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe experiments\\run_adaptive_retrieval_experiment.py --resume {run_dir.name}",
        "```",
        "",
        f"Test status: **{test_result['status']}**. `tests/test_dataset_validation.py` was intentionally excluded because its split loader reads every official claim split, including the prohibited held-out claims file.",
        "",
        f"Validation status: **{validation['status']}**; {validation['query_count']} exact development IDs and {validation['metric_record_count']} metric, {validation['ranking_record_count']} ranking, {validation['decision_record_count']} decision, and {validation['performance_record_count']} performance records.",
        "",
        "## Conclusion",
        "",
        conclusion,
    ])
    return ROOT / "docs" / "RESEARCH_PAPER.md"


def _run_tests(run_dir: Path) -> dict[str, Any]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "-q",
        "tests/test_phase3_retrieval.py",
        "tests/test_adaptive_retrieval_experiment.py",
        "tests/test_reranking_experiment.py",
        "tests/test_bm25_hybrid_retrieval.py",
        "tests/test_retrieval_metrics.py",
        "tests/test_statistical_tests.py",
        "tests/test_run_manager.py",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (run_dir / "test_output.txt").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "suite_scope": "Phase 3 adaptive/cross-encoder retrieval, BM25/hybrid retrieval, retrieval metrics, Experiment B and Experiment A integrity, statistical, and run-manager tests; held-out-claims-reading tests excluded",
        "excluded_tests": [
            "tests/test_dataset_validation.py: split loader reads every official claim split, including claims_test.jsonl",
        ],
        "output": (completed.stdout + "\n" + completed.stderr).strip(),
    }


def main(resume_run_id: str | None = None) -> None:
    global ACTIVE_RUN_DIR, RUN_LOG

    run_started = time.perf_counter()
    queries = _read_queries()
    qrels = _load_qrels({row["query_id"] for row in queries})
    protected_paths = [ROOT / "results" / "runs" / f"run_{number:04d}" for number in range(1, 7)]
    protected_before = {path.name: _sha256_tree(path) for path in protected_paths}
    protected_report = ROOT / "docs" / "RESEARCH_PAPER.md"
    protected_report_hash = hashlib.sha256(protected_report.read_bytes()).hexdigest()
    report_path = protected_report

    if resume_run_id:
        if not resume_run_id.startswith("run_") or "/" in resume_run_id or "\\" in resume_run_id:
            raise ValueError("--resume must be a run directory name such as run_0007")
        run_dir = RUN_ROOT / resume_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run does not exist: {run_dir}")
        if not (run_dir / "config.json").is_file():
            raise ValueError(f"Cannot resume without a saved Experiment B config: {run_dir}")
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if saved_config.get("experiment") != "Phase 3 Experiment B" or saved_config.get("run_id") != run_dir.name:
            raise ValueError(f"Run config does not match the requested Experiment B run: {run_dir}")
    else:
        run_dir = unique_run_dir(RUN_ROOT, prefix="run")

    import torch

    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path = ROOT / "manifests" / "dataset_manifest.json"
    config = _build_config(
        run_dir.name,
        status="running",
        resumed=bool(resume_run_id),
        git_commit=_git_commit(),
        code_sha256=code_sha256,
        dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        dev_claims_sha256=hashlib.sha256(DEV_CLAIMS_PATH.read_bytes()).hexdigest(),
    )
    if resume_run_id:
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for field in ("dataset", "dataset_split", "query_count", "fixed_depths", "adaptive_strategies", "initial_depth", "maximum_depth", "evaluation_cutoff"):
            if saved_config.get(field) != config.get(field):
                raise ValueError(f"Saved run config differs from this protocol in {field}: {run_dir}")

    ACTIVE_RUN_DIR = run_dir
    RUN_LOG = (run_dir / "experiment.log").open("a", encoding="utf-8")
    from src.experiments.artifact_manager import ArtifactManager

    artifact_manager = ArtifactManager(run_dir)
    _atomic_json(artifact_manager.raw_path("config.json"), config)
    metadata = {
        "run_id": run_dir.name,
        "experiment": "Phase 3 Experiment B",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "query_count": 300,
        "git_commit": config["git_commit"],
        "code_sha256": code_sha256,
        "official_test_claim_file_opened": False,
    }
    if resume_run_id and (run_dir / "metadata.json").is_file():
        old_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata["created_at"] = old_metadata.get("created_at", metadata["created_at"])
    _atomic_json(artifact_manager.raw_path("metadata.json"), metadata)
    _atomic_json(run_dir / "run_status.json", {
        "run_id": run_dir.name,
        "status": "running",
        "query_count": 300,
        "completed_query_count": 0,
        "configuration_count": 10,
        "record_count": 3000,
        "resumed": bool(resume_run_id),
    })
    _log(f"run_id={run_dir.name}")
    _log("experiment=Phase 3 Experiment B")

    torch.set_num_threads(TORCH_THREADS)
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.retriever import Retriever
    from src.retrieval import embedder as dense_embedder

    if dense_embedder._MODEL is None:
        raise RuntimeError("Dense embedding model is unavailable")
    dense = Retriever(rebuild=False)
    bm25 = BM25Retriever(corpus_path=CORPUS_PATH, candidate_depth=50)
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        rrf_constant=RRF_CONSTANT,
        bm25_depth=50,
        dense_depth=50,
    )
    model_start = time.perf_counter()
    reranker = CrossEncoderReranker(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device="cpu",
        max_candidates=50,
        strict=True,
    )
    reranker.model_load_seconds = time.perf_counter() - model_start
    if reranker._model is None:
        raise RuntimeError("Required cross-encoder model is unavailable")

    smoke_path = run_dir / "smoke_test.json"
    if resume_run_id and smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "passed":
            raise ValueError(f"Saved smoke test did not pass: {smoke_path}")
    else:
        smoke = _smoke_test(run_dir, queries, dense, hybrid, reranker, dense.corpus_by_id)
    _log(
        f"smoke_test=passed estimated_full_end_to_end_minutes={smoke['estimated_full_end_to_end_minutes']:.1f}"
    )
    metric_rows, ranking_rows, decision_rows, performance_rows = _evaluate(
        run_dir, queries, qrels, dense, hybrid, reranker
    )
    aggregates, candidate_summary, extra = _aggregate_and_statistics(
        run_dir, metric_rows, decision_rows, performance_rows
    )
    performance = json.loads((run_dir / "performance_metrics.json").read_text(encoding="utf-8"))
    figures = _generate_figures(run_dir)
    test_result = _run_tests(run_dir)
    if test_result["status"] != "passed":
        raise RuntimeError(f"Experiment B focused test suite failed; see {run_dir / 'test_output.txt'}")
    experiment_a_comparison = _compare_experiment_a(run_dir, metric_rows)
    validation = _validate(
        run_dir,
        expected_query_ids={row["query_id"] for row in queries},
        corpus_ids=set(dense.corpus_by_id),
        qrels=qrels,
    )
    validation["test_status"] = test_result["status"]
    validation["test_output"] = test_result["output"]

    protected_after = {path.name: _sha256_tree(path) for path in protected_paths}
    protected_report_hash_after = hashlib.sha256(protected_report.read_bytes()).hexdigest()
    if protected_after != protected_before or protected_report_hash_after != protected_report_hash:
        raise RuntimeError("A protected run changed during Experiment B")

    total_seconds = time.perf_counter() - run_started
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "git_commit": config["git_commit"],
        "code_sha256": code_sha256,
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "dev_claims_sha256": config["dev_claims_sha256"],
        "device": "cpu",
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "peak_memory": _memory_snapshot(),
        "protected_run_hashes_before": protected_before,
        "protected_run_hashes_after": protected_after,
        "protected_phase3_report_sha256_before": protected_report_hash,
        "protected_phase3_report_sha256_after": protected_report_hash_after,
        "official_test_claim_file_opened": False,
    }
    _atomic_json(run_dir / "environment_metadata.json", environment)

    config["status"] = "completed"
    config["reranker"]["model_load_seconds"] = reranker.model_load_seconds
    config["run_wall_time_seconds"] = total_seconds
    _atomic_json(artifact_manager.raw_path("config.json"), config)
    metadata.update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "protected_runs_unchanged": True,
        "protected_phase3_report_unchanged": True,
    })
    _atomic_json(artifact_manager.raw_path("metadata.json"), metadata)
    extra["quality_efficiency"] = json.loads(
        (run_dir / "quality_efficiency_tradeoff.json").read_text(encoding="utf-8")
    )
    performance["model_load_seconds"] = reranker.model_load_seconds
    performance["total_run_wall_time_seconds"] = total_seconds
    performance["peak_memory"] = environment["peak_memory"]
    _atomic_json(run_dir / "performance_metrics.json", performance)
    _atomic_json(run_dir / "validation.json", validation)
    _atomic_json(run_dir / "run_status.json", {
        "run_id": run_dir.name,
        "status": "completed",
        "query_count": 300,
        "configuration_count": 10,
        "record_count": 3000,
        "run_wall_time_seconds": total_seconds,
        "official_test_claim_file_opened": False,
        "report": "docs/RESEARCH_PAPER.md",
        "figures": figures,
        "test_status": test_result["status"],
        "historical_runs_unchanged": True,
    })
    _log(f"run_directory={run_dir}")
    _log(f"total_runtime_minutes={total_seconds / 60:.2f}")
    _log(f"tests={test_result['status']}")
    RUN_LOG.close()
    RUN_LOG = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adaptive retrieval experiments.")
    parser.add_argument("--resume", metavar="RUN_ID")
    args = parser.parse_args()
    try:
        main(resume_run_id=args.resume)
    except BaseException as exc:
        if ACTIVE_RUN_DIR is not None:
            _atomic_json(ACTIVE_RUN_DIR / "run_status.json", {
                "run_id": ACTIVE_RUN_DIR.name,
                "status": "failed",
                "error": str(exc),
                "checkpointed": True,
            })
        raise
