from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEV_CLAIMS_PATH = ROOT / "data" / "processed" / "raw_scifact" / "data" / "claims_dev.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
RUN_ROOT = ROOT / "results" / "runs"
A_RUN_ID = "run_0006"
CANDIDATE_POOLS = (10, 20, 50)
RANKING_CONDITIONS = (
    "original",
    "cross_encoder",
    "cross_encoder_lexical",
    "cross_encoder_scientific",
    "evidence_aware_full",
)
METRIC_NAMES = ("hit@10", "recall@10", "mrr@10", "map@10", "ndcg@10")
BOOTSTRAP_ITERATIONS = 10_000
SEED = 42
SMOKE_QUERY_COUNT = 3
ACTIVE_RUN_DIR: Path | None = None
RUN_LOG: Any = None

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.statistical_tests import holm_bonferroni_adjust, paired_bootstrap_ci
from src.experiments.run_manager import unique_run_dir
from src.retrieval.evidence_aware_reranker import EvidenceAwareReranker


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


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
                record = json.loads(line)
                rows.append({"query_id": str(record["id"]), "claim_text": str(record.get("claim") or "")})
    ids = [row["query_id"] for row in rows]
    if len(rows) != 300 or len(set(ids)) != 300:
        raise ValueError("Expected exactly 300 unique official SciFact development claims")
    return rows


def _read_corpus() -> dict[str, dict[str, Any]]:
    corpus = {}
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                corpus_id = str(row.get("corpus_id") or row.get("document_id") or "")
                if corpus_id:
                    corpus[corpus_id] = row
    return corpus


def _load_qrels(query_ids: set[str]) -> dict[str, dict[str, float]]:
    from datasets import load_dataset
    from collections import defaultdict

    split = load_dataset("BeIR/scifact-qrels", split="test")
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in split:
        query_id = str(row["query-id"])
        if query_id not in query_ids:
            raise ValueError(f"BEIR qrels contains query outside official development IDs: {query_id}")
        qrels[query_id][str(row["corpus-id"])] = float(row["score"])
    if set(qrels) != query_ids:
        raise ValueError("BEIR development qrels do not cover all 300 official development queries")
    return dict(qrels)


def _sha256_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        raise FileNotFoundError(f"Protected run is missing: {root}")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _memory_snapshot() -> dict[str, int | str | None]:
    return {"status": "unavailable", "peak_working_set_bytes": None, "working_set_bytes": None}


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = max(0, min(len(values) - 1, int(round(0.95 * (len(values) - 1)))))
    return float(values[index])


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": mean(values) if values else 0.0,
        "median_seconds": median(values) if values else 0.0,
        "p95_seconds": _p95(values),
    }


def _validated_experiment_a(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metrics = json.loads((run_dir / "per_query_metrics.json").read_text(encoding="utf-8"))
    rankings = json.loads((run_dir / "per_query_rankings.json").read_text(encoding="utf-8"))
    performance = {}
    performance_csv = run_dir / "query_performance.csv"
    if performance_csv.exists():
        with performance_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["query_id"]), str(row["base_method"]), int(row["candidate_pool_size"]), str(row["condition"]))
                performance[key] = row
    return metrics, rankings, performance


def _rank_lookup(rankings: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], list[dict[str, Any]]]:
    result = {}
    for row in rankings:
        key = (str(row["query_id"]), str(row["base_method"]), int(row["candidate_pool_size"]), str(row["condition"]))
        result[key] = row["rankings"]
    return result


def _hydrate(ranking: list[dict[str, Any]], corpus: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hydrated = []
    for row in ranking:
        doc_id = str(row["document_id"])
        doc = corpus.get(doc_id)
        if doc is None:
            raise ValueError(f"Candidate document ID missing from corpus: {doc_id}")
        hydrated.append({
            **doc,
            "document_id": doc_id,
            "corpus_id": doc_id,
            "original_rank": int(row.get("original_rank", row.get("rank", len(hydrated) + 1))),
            "rank": int(row.get("rank", len(hydrated) + 1)),
            "retrieval_score": float(row.get("retrieval_score", row.get("score", 0.0))),
            "cross_encoder_score": float(row.get("cross_encoder_score", 0.0)),
        })
    return hydrated


def _valid_checkpoint(path: Path, query_id: str, configs: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if str(checkpoint.get("query_id")) != query_id:
        raise ValueError(f"Checkpoint query ID mismatch: {path}")
    for field in ("metric_rows", "ranking_rows", "component_scores", "performance_rows"):
        if field not in checkpoint or not isinstance(checkpoint[field], list):
            raise ValueError(f"Incomplete {field} checkpoint: {path}")
    for field in ("metric_rows", "ranking_rows", "performance_rows"):
        seen = {str(row["configuration"]) for row in checkpoint[field]}
        if len(seen) != len(checkpoint[field]) or seen != set(configs):
            raise ValueError(f"Duplicate or missing configuration in {field}: {path}")
    return checkpoint


def _configuration_name(method: str, pool: int, condition: str) -> str:
    return f"{method}_pool_{pool}_{condition}"


def _ranked_ids(ranking: list[dict[str, Any]]) -> list[str]:
    ids = [str(row["document_id"]) for row in ranking]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate candidate IDs")
    return ids


def _reranker_config(condition: str) -> EvidenceAwareReranker | None:
    if condition == "cross_encoder_lexical":
        return EvidenceAwareReranker({"cross_encoder": 0.55, "lexical_overlap": 0.20})
    if condition == "cross_encoder_scientific":
        return EvidenceAwareReranker({"cross_encoder": 0.55, "scientific_overlap": 0.15})
    if condition == "evidence_aware_full":
        return EvidenceAwareReranker()
    return None


def _evidence_rank(
    claim: str,
    candidates: list[dict[str, Any]],
    cross_scores: dict[str, float],
    reranker: EvidenceAwareReranker,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    ordered_candidates = []
    for candidate in candidates:
        doc_id = str(candidate["document_id"])
        score = cross_scores.get(doc_id)
        if score is None:
            raise ValueError(f"Missing cached cross-encoder score for candidate {doc_id}")
        ordered_candidates.append(candidate)
    aligned_scores = [cross_scores[str(candidate["document_id"])] for candidate in ordered_candidates]
    start = time.perf_counter()
    ranked = reranker.rerank(claim, ordered_candidates, top_k=len(ordered_candidates), cross_encoder_scores=aligned_scores)
    elapsed = time.perf_counter() - start
    component_scores = []
    for row in ranked:
        component_scores.append({
            "document_id": str(row["document_id"]),
            "cross_encoder_score": float(row["cross_encoder_score"]),
            "cross_encoder_normalized": float(row["cross_encoder_normalized"]),
            "lexical_overlap": float(row["lexical_overlap"]),
            "scientific_overlap": float(row["scientific_overlap"]),
            "retrieval_score": float(row["retrieval_score"]),
            "retrieval_score_normalized": float(row["retrieval_score_normalized"]),
            "retrieval_rank": int(row["retrieval_rank"]),
            "rank_signal": float(row["rank_signal"]),
            "evidence_score": float(row["evidence_score"]),
        })
    return ranked, component_scores, elapsed


def _smoke_test(run_dir: Path, queries: list[dict[str, str]], rank_lookup: dict, corpus: dict, ranking_performance: dict) -> dict[str, Any]:
    rows = []
    started = time.perf_counter()
    for query in queries[:SMOKE_QUERY_COUNT]:
        for method in ("dense", "hybrid"):
            for pool in CANDIDATE_POOLS:
                for condition in RANKING_CONDITIONS:
                    reranker = _reranker_config(condition)
                    if reranker is None:
                        continue
                    baseline = rank_lookup[(query["query_id"], method, pool, "baseline")]
                    reranked = rank_lookup[(query["query_id"], method, pool, "reranked")]
                    candidates = _hydrate(baseline, corpus)
                    cross_scores = {str(row["document_id"]): float(row["cross_encoder_score"]) for row in reranked if "cross_encoder_score" in row}
                    _, _, seconds = _evidence_rank(query["claim_text"], candidates, cross_scores, reranker)
                    rows.append({
                        "query_id": query["query_id"],
                        "retrieval_method": method,
                        "candidate_pool_size": pool,
                        "configuration": condition,
                        "reranking_seconds": seconds,
                    })
    elapsed = time.perf_counter() - started
    return {
        "status": "passed",
        "purpose": "reuse-only technical smoke test; no smoke metrics included in final evaluation",
        "query_count": min(SMOKE_QUERY_COUNT, len(queries)),
        "configuration_runs": len(rows),
        "mean_evidence_reranking_seconds": mean(row["reranking_seconds"] for row in rows) if rows else 0.0,
        "estimated_full_evidence_reranking_minutes": mean(row["reranking_seconds"] for row in rows) * len(queries) * 2 * 3 / 60.0 if rows else 0.0,
        "wall_seconds": elapsed,
        "model_inference_rerun": False,
        "measurements": rows,
    }


def _evaluate(
    run_dir: Path,
    queries: list[dict[str, str]],
    qrels: dict[str, dict[str, float]],
    rank_lookup: dict,
    corpus: dict,
    ranking_performance: dict,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_dir = run_dir / "checkpoints" / "queries"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        _configuration_name(method, pool, condition)
        for method in ("dense", "hybrid")
        for pool in CANDIDATE_POOLS
        for condition in RANKING_CONDITIONS
    ]
    expected_config_keys = [
        (method, pool, condition)
        for method in ("dense", "hybrid")
        for pool in CANDIDATE_POOLS
        for condition in RANKING_CONDITIONS
    ]
    checkpoints = {}
    started = time.perf_counter()
    run_memory_snapshot = _memory_snapshot()

    for query_index, query in enumerate(queries, start=1):
        query_id = query["query_id"]
        checkpoint_path = checkpoint_dir / f"{query_id}.json"
        checkpoint = _valid_checkpoint(checkpoint_path, query_id, configs)
        if checkpoint is None:
            metric_rows = []
            ranking_rows = []
            component_rows = []
            performance_rows = []
            for method, pool, condition in expected_config_keys:
                baseline = rank_lookup[(query_id, method, pool, "baseline")]
                reranked = rank_lookup[(query_id, method, pool, "reranked")]
                baseline_hydrated = _hydrate(baseline, corpus)
                cross_scores = {
                    str(row["document_id"]): float(row["cross_encoder_score"])
                    for row in reranked
                    if "cross_encoder_score" in row
                }
                reranking_seconds = 0.0
                if condition == "original":
                    ranked = [{**row, "rank": index} for index, row in enumerate(baseline_hydrated, start=1)]
                elif condition == "cross_encoder":
                    ranked = _hydrate(reranked, corpus)
                    reranking_seconds = 0.0
                else:
                    reranker = _reranker_config(condition)
                    if reranker is None:
                        raise AssertionError("Missing evidence reranker")
                    ranked, components, reranking_seconds = _evidence_rank(
                        query["claim_text"], baseline_hydrated, cross_scores, reranker
                    )
                    component_rows.append({
                        "query_id": query_id,
                        "retrieval_method": method,
                        "candidate_pool_size": pool,
                        "configuration": condition,
                        "weights": reranker._normalized_weights(),
                        "components": components,
                    })

                ids = _ranked_ids(ranked)
                if len(ids) != pool:
                    raise AssertionError(f"Candidate count mismatch: {query_id} {method} {pool} {condition}")
                metrics = compute_retrieval_metrics(ids, qrels.get(query_id, {}), (10,))
                metric_rows.append({
                    "query_id": query_id,
                    "retrieval_method": method,
                    "candidate_pool_size": pool,
                    "configuration": _configuration_name(method, pool, condition),
                    **metrics,
                })
                ranking_rows.append({
                    "query_id": query_id,
                    "retrieval_method": method,
                    "candidate_pool_size": pool,
                    "configuration": _configuration_name(method, pool, condition),
                    "rankings": [
                        {
                            "document_id": str(row["document_id"]),
                            "original_rank": int(row.get("original_rank", row.get("rank", index))),
                            "rank": index,
                            "retrieval_score": float(row.get("retrieval_score", row.get("score", 0.0))),
                            "cross_encoder_score": float(row.get("cross_encoder_score", 0.0)) if "cross_encoder_score" in row else None,
                        }
                        for index, row in enumerate(ranked, start=1)
                    ],
                })

                performance_row = ranking_performance.get((query_id, method, pool, "baseline"), {})
                retrieval_latency = float(performance_row.get("retrieval_latency_seconds", 0.0))
                performance_rows.append({
                    "query_id": query_id,
                    "retrieval_method": method,
                    "candidate_pool_size": pool,
                    "configuration": _configuration_name(method, pool, condition),
                    "retrieval_latency_seconds": retrieval_latency,
                    "reranking_latency_seconds": reranking_seconds,
                    "cross_encoder_inference_seconds": 0.0,
                    "end_to_end_latency_seconds": retrieval_latency + reranking_seconds,
                    "candidate_pool_processed": pool,
                    "cross_encoder_scores_reused": True,
                    "memory_bytes": run_memory_snapshot,
                })

            checkpoint = {
                "query_id": query_id,
                "metric_rows": metric_rows,
                "ranking_rows": ranking_rows,
                "component_scores": component_rows,
                "performance_rows": performance_rows,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(checkpoint_path, checkpoint)
        checkpoints[query_id] = checkpoint
        if query_index % 10 == 0 or query_index == len(queries):
            elapsed = time.perf_counter() - started
            remaining = (elapsed / query_index) * (len(queries) - query_index) if query_index else 0.0
            _log(f"evaluated={query_index}/{len(queries)} elapsed_minutes={elapsed/60:.1f} estimated_remaining_minutes={remaining/60:.1f}")

    metric_rows = [row for query in queries for row in checkpoints[query["query_id"]]["metric_rows"]]
    ranking_rows = [row for query in queries for row in checkpoints[query["query_id"]]["ranking_rows"]]
    component_rows = [row for query in queries for row in checkpoints[query["query_id"]]["component_scores"]]
    performance_rows = [row for query in queries for row in checkpoints[query["query_id"]]["performance_rows"]]

    _atomic_json(run_dir / "per_query_metrics.json", metric_rows)
    _atomic_json(run_dir / "per_query_rankings.json", ranking_rows)
    _atomic_json(run_dir / "evidence_component_scores.json", component_rows)
    _atomic_json(run_dir / "performance_measurements.json", performance_rows)

    with (run_dir / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "retrieval_method", "candidate_pool_size", "configuration", *METRIC_NAMES, "precision@10"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (run_dir / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for row in ranking_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "evidence_component_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in component_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "query_performance.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "query_id", "retrieval_method", "candidate_pool_size", "configuration",
            "retrieval_latency_seconds", "reranking_latency_seconds",
            "cross_encoder_inference_seconds", "end_to_end_latency_seconds",
            "candidate_pool_processed", "cross_encoder_scores_reused",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in performance_rows:
            writer.writerow({key: row[key] for key in fields})

    aggregates = {}
    for method in ("dense", "hybrid"):
        for pool in CANDIDATE_POOLS:
            for condition in RANKING_CONDITIONS:
                name = _configuration_name(method, pool, condition)
                rows = [row for row in metric_rows if row["configuration"] == name]
                aggregates[name] = {
                    metric: mean(float(row[metric]) for row in rows)
                    for metric in METRIC_NAMES
                }
    _atomic_json(run_dir / "aggregate_metrics.json", aggregates)

    performance = {}
    for name in sorted({row["configuration"] for row in performance_rows}):
        rows = [row for row in performance_rows if row["configuration"] == name]
        for field, target in (
            ("reranking_latency_seconds", "reranking"),
            ("end_to_end_latency_seconds", "end_to_end"),
            ("retrieval_latency_seconds", "retrieval"),
        ):
            performance[f"{name}_{target}_latency"] = _latency_summary([float(row[field]) for row in rows])
        performance[f"{name}_cross_encoder_inference"] = _latency_summary([float(row["cross_encoder_inference_seconds"]) for row in rows])
    performance["evaluation_wall_time_seconds"] = time.perf_counter() - started
    performance["query_count"] = len(queries)
    performance["cross_encoder_inference_rerun_count"] = 0
    _atomic_json(run_dir / "performance_metrics.json", performance)

    return {"metric_rows": metric_rows, "ranking_rows": ranking_rows, "component_rows": component_rows, "performance_rows": performance_rows}, aggregates, performance


def _statistical_analysis(run_dir: Path, metric_rows: list[dict[str, Any]], queries: list[dict[str, str]]) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in metric_rows:
        lookup[(str(row["query_id"]), str(row["configuration"]), 0)] = {
            metric: float(row[metric]) for metric in METRIC_NAMES
        }

    comparisons = []
    comparison_pairs = (
        ("original", "cross_encoder", "cross_encoder_minus_original"),
        ("original", "evidence_aware_full", "evidence_full_minus_original"),
        ("cross_encoder", "evidence_aware_full", "evidence_full_minus_cross_encoder"),
        ("cross_encoder", "cross_encoder_lexical", "lexical_minus_cross_encoder"),
        ("cross_encoder", "cross_encoder_scientific", "scientific_minus_cross_encoder"),
    )
    for method in ("dense", "hybrid"):
        for pool in CANDIDATE_POOLS:
            for comparator, treatment, comparison_name in comparison_pairs:
                comparator_name = _configuration_name(method, pool, comparator)
                treatment_name = _configuration_name(method, pool, treatment)
                for metric in METRIC_NAMES:
                    comparator_scores = [
                        lookup[(query["query_id"], comparator_name, 0)][metric]
                        for query in queries
                    ]
                    treatment_scores = [
                        lookup[(query["query_id"], treatment_name, 0)][metric]
                        for query in queries
                    ]
                    differences = [b - a for a, b in zip(comparator_scores, treatment_scores)]
                    bootstrap = paired_bootstrap_ci(
                        treatment_scores,
                        comparator_scores,
                        n_bootstrap=BOOTSTRAP_ITERATIONS,
                        confidence_level=0.95,
                        seed=SEED,
                    )
                    try:
                        result = wilcoxon(treatment_scores, comparator_scores, alternative="two-sided", zero_method="wilcox")
                        statistic = float(result.statistic)
                        p_value = float(result.pvalue)
                    except Exception:
                        statistic = 0.0
                        p_value = 1.0
                    denominator = (sum((d - mean(differences)) ** 2 for d in differences) / max(1, len(differences) - 1)) ** 0.5
                    effect = mean(differences) / denominator if denominator > 0.0 else 0.0
                    comparisons.append({
                        "retrieval_method": method,
                        "candidate_pool_size": pool,
                        "comparison": comparison_name,
                        "comparator": comparator,
                        "treatment": treatment,
                        "metric": metric,
                        "comparator_mean": mean(comparator_scores),
                        "treatment_mean": mean(treatment_scores),
                        "mean_difference_treatment_minus_comparator": mean(differences),
                        "bootstrap_ci95": bootstrap["ci95"],
                        "paired_cohens_dz": effect,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "sample_size": len(differences),
                    })

    p_values = [row["p_value"] for row in comparisons]
    adjusted = holm_bonferroni_adjust(p_values)
    for row, adjusted_p in zip(comparisons, adjusted):
        row["p_value_holm_adjusted"] = float(adjusted_p)
        row["statistically_significant_after_holm"] = bool(adjusted_p < 0.05)

    payload = {
        "paired_test": "two-sided Wilcoxon signed-rank",
        "bootstrap_resamples_per_comparison": BOOTSTRAP_ITERATIONS,
        "confidence_level": 0.95,
        "seed": SEED,
        "correction": "Holm-Bonferroni",
        "comparison_family_size": len(comparisons),
        "comparisons": comparisons,
    }
    _atomic_json(run_dir / "statistical_tests.json", payload)
    ci_payload = {
        "confidence_level": 0.95,
        "resamples": BOOTSTRAP_ITERATIONS,
        "confidence_intervals": [
            {
                "retrieval_method": row["retrieval_method"],
                "candidate_pool_size": row["candidate_pool_size"],
                "comparison": row["comparison"],
                "metric": row["metric"],
                "mean_difference": row["mean_difference_treatment_minus_comparator"],
                "ci95": row["bootstrap_ci95"],
            }
            for row in comparisons
            if row["metric"] == "ndcg@10"
        ],
    }
    _atomic_json(run_dir / "confidence_intervals.json", ci_payload)
    return payload


def _generate_figures(run_dir: Path, aggregates: dict[str, Any], performance: dict[str, Any], statistics: dict[str, Any]) -> list[str]:
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    methods = ("dense", "hybrid")
    conditions = ("original", "cross_encoder", "evidence_aware_full")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, method in zip(axes, methods):
        for pool in CANDIDATE_POOLS:
            values = [aggregates[_configuration_name(method, pool, condition)]["ndcg@10"] for condition in conditions]
            axis.plot(range(len(values)), values, marker="o", label=f"Pool {pool}")
        axis.set_xticks(range(len(conditions)), ["Original", "Cross-encoder", "Evidence-aware"])
        axis.set_xlabel("Ranking configuration")
        axis.set_ylabel("Mean nDCG@10")
        axis.set_title(f"{method.title()} retrieval")
        axis.legend()
    fig.tight_layout()
    path = figure_dir / "baseline_vs_cross_encoder_vs_evidence_aware_ndcg10.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path.name)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method in methods:
        values = [aggregates[_configuration_name(method, pool, "evidence_aware_full")]["ndcg@10"] for pool in CANDIDATE_POOLS]
        axis.plot(CANDIDATE_POOLS, values, marker="o", label=method.title())
    axis.set_xlabel("Candidate pool size")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Evidence-aware performance across candidate pool sizes")
    axis.legend()
    fig.tight_layout()
    path = figure_dir / "performance_across_candidate_pool_sizes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ablations = ("cross_encoder", "cross_encoder_lexical", "cross_encoder_scientific", "evidence_aware_full")
    for axis, method in zip(axes, methods):
        values = [aggregates[_configuration_name(method, 50, condition)]["ndcg@10"] for condition in ablations]
        axis.bar(["CE", "CE+Lexical", "CE+Scientific", "Full"], values)
        axis.set_ylabel("Mean nDCG@10")
        axis.set_xlabel("Pool-50 ranking configuration")
        axis.set_title(f"{method.title()} pool-50 ablation")
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = figure_dir / "individual_evidence_signal_contribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path.name)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method in methods:
        for condition in RANKING_CONDITIONS:
            name = _configuration_name(method, 50, condition)
            latency = performance[f"{name}_end_to_end_latency"]["mean_seconds"]
            ndcg = aggregates[name]["ndcg@10"]
            axis.scatter(latency, ndcg, label=f"{method.title()} {condition}")
    axis.set_xlabel("Mean end-to-end latency (seconds)")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Pool-50 quality versus latency")
    axis.legend(fontsize=7)
    fig.tight_layout()
    path = figure_dir / "quality_vs_latency.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path.name)

    principal = [
        row for row in statistics["comparisons"]
        if row["metric"] == "ndcg@10" and row["comparison"] == "evidence_full_minus_cross_encoder"
    ]
    if principal:
        labels = [f"{row['retrieval_method']} pool {row['candidate_pool_size']}" for row in principal]
        differences = [row["mean_difference_treatment_minus_comparator"] for row in principal]
        lows = [row["bootstrap_ci95"][0] for row in principal]
        highs = [row["bootstrap_ci95"][1] for row in principal]
        fig, axis = plt.subplots(figsize=(9, 4.5))
        y_positions = list(range(len(labels)))
        axis.errorbar(
            differences,
            y_positions,
            xerr=[[d - low for d, low in zip(differences, lows)], [high - d for d, high in zip(differences, highs)]],
            fmt="o",
            capsize=4,
        )
        axis.axvline(0.0, linestyle="--", linewidth=1)
        axis.set_yticks(y_positions, labels)
        axis.set_xlabel("Full evidence-aware minus cross-encoder nDCG@10")
        axis.set_title("Principal paired bootstrap confidence intervals")
        fig.tight_layout()
        path = figure_dir / "principal_confidence_intervals.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(path.name)
    return figures


def _validate(
    run_dir: Path,
    queries: list[dict[str, str]],
    metric_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    performance_rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    expected = len(queries) * len(RANKING_CONDITIONS) * 2 * len(CANDIDATE_POOLS)
    if len(metric_rows) != expected or len(ranking_rows) != expected or len(performance_rows) != expected:
        raise AssertionError("Missing or extra configuration records")
    identities = [(row["query_id"], row["retrieval_method"], row["candidate_pool_size"], row["configuration"]) for row in metric_rows]
    if len(identities) != len(set(identities)):
        raise AssertionError("Duplicate query/configuration records")
    if set(row["query_id"] for row in metric_rows) != set(query["query_id"] for query in queries):
        raise AssertionError("Development query coverage mismatch")

    for row in ranking_rows:
        ids = [str(item["document_id"]) for item in row["rankings"]]
        if len(ids) != len(set(ids)):
            raise AssertionError("Duplicate candidate IDs in ranking")
        if len(ids) != int(row["candidate_pool_size"]):
            raise AssertionError("Ranking candidate count mismatch")

    for row in performance_rows:
        for field in ("retrieval_latency_seconds", "reranking_latency_seconds", "cross_encoder_inference_seconds", "end_to_end_latency_seconds"):
            if float(row[field]) < 0.0:
                raise AssertionError("Negative runtime measurement")

    for name, aggregate in aggregates.items():
        expected_name_rows = [row for row in metric_rows if row["configuration"] == name]
        if len(expected_name_rows) != len(queries):
            raise AssertionError("Aggregate query coverage mismatch")
        for metric in METRIC_NAMES:
            value = mean(float(row[metric]) for row in expected_name_rows)
            if abs(value - float(aggregate[metric])) > 1e-12:
                raise AssertionError("Aggregate metric mismatch")

    adjusted = [row["p_value_holm_adjusted"] for row in statistics["comparisons"]]
    if any(value < 0.0 or value > 1.0 for value in adjusted):
        raise AssertionError("Invalid Holm-adjusted p-values")
    if len(statistics["comparisons"]) != 150:
        raise AssertionError("Expected 150 Experiment C paired tests")

    payload = {
        "status": "passed",
        "query_count": len(queries),
        "metric_record_count": len(metric_rows),
        "ranking_record_count": len(ranking_rows),
        "component_score_record_count": len(component_rows),
        "performance_record_count": len(performance_rows),
        "configuration_count": len(RANKING_CONDITIONS) * 2 * len(CANDIDATE_POOLS),
        "duplicate_query_configuration_records": False,
        "aggregate_metrics_match_per_query": True,
        "candidate_ids_unique": True,
        "statistical_correction_valid": True,
        "paired_comparison_count": len(statistics["comparisons"]),
        "official_test_claim_file_opened": False,
    }
    _atomic_json(run_dir / "validation.json", payload)
    return payload


def _write_report(
    run_dir: Path,
    smoke: dict[str, Any],
    aggregates: dict[str, Any],
    performance: dict[str, Any],
    statistics: dict[str, Any],
    validation: dict[str, Any],
    figures: list[str],
) -> Path:
    full_vs_cross = [
        row for row in statistics["comparisons"]
        if row["metric"] == "ndcg@10" and row["comparison"] == "evidence_full_minus_cross_encoder"
    ]
    full_vs_original = [
        row for row in statistics["comparisons"]
        if row["metric"] == "ndcg@10" and row["comparison"] == "evidence_full_minus_original"
    ]
    lex_vs_cross = [
        row for row in statistics["comparisons"]
        if row["metric"] == "ndcg@10" and row["comparison"] == "lexical_minus_cross_encoder"
    ]
    sci_vs_cross = [
        row for row in statistics["comparisons"]
        if row["metric"] == "ndcg@10" and row["comparison"] == "scientific_minus_cross_encoder"
    ]

    best_name = max(
        aggregates,
        key=lambda name: aggregates[name]["ndcg@10"]
        / performance[f"{name}_end_to_end_latency"]["mean_seconds"]
        if performance[f"{name}_end_to_end_latency"]["mean_seconds"] > 0.0
        else aggregates[name]["ndcg@10"],
    )
    code_hash = hashlib.sha256((ROOT / "experiments" / "run_experiment_c.py").read_bytes()).hexdigest()
    git_commit = ""
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = "unavailable"

    lines = [
        "# Phase 3 - Experiment C: Evidence-Aware Reranking",
        "",
        "> Official SciFact development-set results only. This report is not a held-out test-set result.",
        "",
        f"- Run: `{run_dir.name}` (`{run_dir.as_posix()}`)",
        f"- Git commit: `{git_commit}`; runner SHA-256: `{code_hash}`.",
        "- Dataset: 300 official SciFact development claims.",
        "- Qrels: BEIR `scifact-qrels` `test` split, mapped by the manifest to official development claims.",
        "- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU.",
        "- Experiment A artifacts: `run_0006` were reused for original rankings and cross-encoder scores; no cross-encoder inference was rerun.",
        "- No official held-out claims or labels were accessed.",
        "",
        "## Research objective",
        "",
        "Test whether document-level lexical, scientific-terminology, original-rank, and retrieval-score signals improve ranking over cross-encoder-only reranking.",
        "",
        "## Hypotheses",
        "",
        "- H1: Evidence-aware reranking improves nDCG@10 over cross-encoder-only reranking.",
        "- H2: Evidence-aware reranking improves ordering of scientifically relevant documents.",
        "- H3: Combining cross-encoder relevance with evidence-related signals improves over individual signals.",
        "- H4: Any quality improvement justifies the measured additional reranking cost.",
        "",
        "## Methodology",
        "",
        "- Candidate pools: 10, 20, 50.",
        "- Retrieval methods: Dense and Hybrid.",
        "- Ranking conditions: original retrieval, cached cross-encoder reranking, cross-encoder + lexical overlap, cross-encoder + scientific terminology overlap, and the full evidence-aware combination.",
        "- Total configurations: 30.",
        "- Identical candidate ID sets are used for paired conditions within each method and pool.",
        "- Cross-encoder scores and rankings are reused verbatim from Experiment A run `run_0006`; only evidence-aware post-processing was newly computed.",
        "- `tests/test_dataset_validation.py` was intentionally excluded because it can load all claim splits, including the held-out split.",
        "",
        "## Evidence-aware scoring formula",
        "",
        "For candidate i:",
        "",
        "`E_i = w_ce * CE_norm_i + w_lex * Lex_i + w_sci * Sci_i + w_rank * Rank_norm_i + w_retrieval * Retrieval_norm_i`",
        "",
        "- CE norm: min-max normalization of cached cross-encoder scores within the candidate pool.",
        "- Lexical overlap: claim/document token-overlap fraction over claim tokens.",
        "- Scientific overlap: Jaccard overlap of claim and document scientific terminology sets.",
        "- Retrieval rank: `1 - (original_rank - 1)/(pool_size - 1)`, with top rank 1 and bottom rank 0.",
        "- Retrieval score: min-max normalization within the candidate pool.",
        "- Default weights: CE 0.55, lexical 0.20, scientific 0.15, rank 0.10, retrieval score 0.00.",
        "- Ablation weights are renormalized across enabled signals. Duplicate IDs are removed, and ties are ordered by cross-encoder score, original rank, then document ID.",
        "- A scientific terminology match indicates lexical/scientific co-occurrence only; it is not entailment or evidence grounding.",
        "",
        "## Main results",
        "",
        "| Configuration | Hit@10 | Recall@10 | MRR@10 | MAP@10 | nDCG@10 | Mean reranking (s) | Mean end-to-end (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(aggregates):
        aggregate = aggregates[name]
        latency = performance[f"{name}_reranking_latency"]["mean_seconds"]
        end_latency = performance[f"{name}_end_to_end_latency"]["mean_seconds"]
        lines.append(
            f"| `{name}` | {aggregate['hit@10']:.6f} | {aggregate['recall@10']:.6f} | "
            f"{aggregate['mrr@10']:.6f} | {aggregate['map@10']:.6f} | {aggregate['ndcg@10']:.6f} | "
            f"{latency:.6f} | {end_latency:.6f} |"
        )

    lines.extend([
        "",
        "## Main nDCG@10 comparisons",
        "",
        "| Method | Pool | Full evidence - cross-encoder | 95% paired bootstrap CI | Holm p | Significant |",
        "|---|---:|---:|---|---:|---|",
    ])
    for row in sorted(full_vs_cross, key=lambda item: (item["retrieval_method"], item["candidate_pool_size"])):
        ci = row["bootstrap_ci95"]
        lines.append(
            f"| {row['retrieval_method']} | {row['candidate_pool_size']} | "
            f"{row['mean_difference_treatment_minus_comparator']:.6f} | "
            f"[{ci[0]:.6f}, {ci[1]:.6f}] | {row['p_value_holm_adjusted']:.6g} | "
            f"{'yes' if row['statistically_significant_after_holm'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Ablation findings",
        "",
        "| Method | Pool | Lexical - cross-encoder nDCG@10 | Scientific - cross-encoder nDCG@10 | Full - cross-encoder nDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ])
    lex_lookup = {(row["retrieval_method"], row["candidate_pool_size"]): row for row in lex_vs_cross}
    sci_lookup = {(row["retrieval_method"], row["candidate_pool_size"]): row for row in sci_vs_cross}
    full_lookup = {(row["retrieval_method"], row["candidate_pool_size"]): row for row in full_vs_cross}
    for method in ("dense", "hybrid"):
        for pool in CANDIDATE_POOLS:
            lex = lex_lookup[(method, pool)]
            sci = sci_lookup[(method, pool)]
            full = full_lookup[(method, pool)]
            lines.append(
                f"| {method} | {pool} | "
                f"{lex['mean_difference_treatment_minus_comparator']:.6f} | "
                f"{sci['mean_difference_treatment_minus_comparator']:.6f} | "
                f"{full['mean_difference_treatment_minus_comparator']:.6f} |"
            )

    lines.extend([
        "",
        "## Statistical analysis",
        "",
        "- Paired test: two-sided Wilcoxon signed-rank.",
        "- Paired bootstrap: 10,000 resamples, 95% confidence intervals, seed 42.",
        "- Effect size: paired Cohen's dz.",
        "- Holm-Bonferroni family: 150 tests: 2 methods x 3 pools x 5 comparison types x 5 metrics.",
        "- The comparison types cover original vs cross-encoder, original vs full evidence, cross-encoder vs full evidence, cross-encoder vs lexical ablation, and cross-encoder vs scientific ablation.",
        "- No result is significant unless the Holm-adjusted p-value is below 0.05.",
        "",
        "## Efficiency",
        "",
        f"- Smoke test: 3 queries, {smoke['configuration_runs']} evidence-aware reranking runs.",
        f"- Mean measured evidence reranking latency: {smoke['mean_evidence_reranking_seconds']:.6f} seconds per run.",
        "- Cross-encoder model inference was not rerun; all cached scores came from Experiment A.",
        "- End-to-end latency includes Experiment A's measured retrieval latency plus newly measured evidence post-processing latency.",
        "- Memory snapshots are recorded in `performance_measurements.json`.",
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
        "- This is a development-set experiment. It does not estimate held-out test performance.",
        "- Existing weights are project-configured hypotheses, not calibrated probabilities.",
        "- Scientific terminology overlap is document-level lexical overlap, not entailment or sentence-level evidence grounding.",
        "- Contradictory documents are not automatically penalized.",
        "- Hybrid comparison reuse uses Experiment A's component and RRF configuration; candidate sets are exact for paired retrieval/pool conditions.",
        "- Cross-encoder inference cost is excluded because scores were reused; evidence reranking overhead is measured directly.",
        "",
        "## Experiment A and B comparison",
        "",
        "- Experiment A's cached Dense and Hybrid cross-encoder rankings and scores are the direct cross-encoder baseline.",
        "- Experiment B adaptive candidate pools are not directly compared because Experiment C uses fixed 10/20/50 pools and cached fixed-pool configurations.",
        "- Fixed-pool Experiment C results should be interpreted alongside Experiment B's development-only adaptive measurements.",
        "",
        "## Validation and reproduction",
        "",
        f"- Validation status: **{validation['status']}**.",
        f"- Records: {validation['metric_record_count']} metric, {validation['ranking_record_count']} ranking, {validation['component_score_record_count']} component-score, and {validation['performance_record_count']} performance records.",
        "- The report and artifacts preserve run-0001 through run-0008 through SHA-256 checks.",
        "",
        "Start a new run:",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe experiments\\run_experiment_c.py",
        "```",
        "",
        "Resume this run:",
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe experiments\\run_experiment_c.py --resume {run_dir.name}",
        "```",
        "",
        "## Conclusion",
        "",
        f"The best observed nDCG@10 configuration was `{max(aggregates, key=lambda name: aggregates[name]['ndcg@10'])}` "
        f"(nDCG@10 {max(row['ndcg@10'] for row in aggregates.values()):.6f}). "
        "Statistical conclusions are limited by the corrected paired tests and the development-only data. "
        "No scientific entailment or final held-out conclusion is claimed.",
    ])
    report = ROOT / "docs" / "PHASE3_EXPERIMENT_C_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_phase3_summary() -> Path:
    lines = [
        "# Phase 3 Final Summary",
        "",
        "## Scope",
        "",
        "- Experiment A evaluated cross-encoder reranking over fixed Dense and Hybrid pools.",
        "- Experiment B evaluated fixed and adaptive Dense/Hybrid retrieval.",
        "- Experiment C evaluated evidence-aware reranking against cached Experiment A rankings.",
        "",
        "## Phase 3 coverage",
        "",
        "- Candidate pool sizes: 10, 20, 50.",
        "- Retrieval methods: Dense and Hybrid.",
        "- Reranking strategies: original retrieval, cross-encoder-only, evidence-aware reranking, and evidence-signal ablations.",
        "- Adaptive retrieval: included in Experiment B fixed/adaptive depth comparisons.",
        "",
        "## Conclusions",
        "",
        "- See `docs/PHASE3_REPORT.md`, `docs/PHASE3_EXPERIMENT_B_REPORT.md`, and `docs/PHASE3_EXPERIMENT_C_REPORT.md`.",
        "- Statistical significance must be read from Holm-corrected per-query analyses, not from development means alone.",
        "- Experiment C is document-level evidence-related reranking only, not sentence-level evidence grounding.",
        "",
        "## Remaining limitations",
        "",
        "- Official held-out test claims remain untouched.",
        "- Evidence weights are preconfigured and not calibrated on the relevance judgments.",
        "- Computational conclusions use development-only CPU measurements.",
        "",
        "## Completion assessment",
        "",
        "Phase 3 requirements are addressed across Experiments A-C. Held-out test evaluation and the evidence-grounding pipeline are outside this phase.",
    ]
    report = ROOT / "docs" / "PHASE3_FINAL_SUMMARY.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(resume_run_id: str | None = None) -> None:
    global ACTIVE_RUN_DIR, RUN_LOG

    protected_paths = [ROOT / "results" / "runs" / f"run_{number:04d}" for number in range(1, 9)]
    protected_before = {path.name: _sha256_tree(path) for path in protected_paths}
    protected_report = ROOT / "docs" / "PHASE3_REPORT.md"
    protected_report_hash = hashlib.sha256(protected_report.read_bytes()).hexdigest()

    if resume_run_id:
        run_dir = RUN_ROOT / resume_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run does not exist: {run_dir}")
    else:
        run_dir = unique_run_dir(RUN_ROOT)

    ACTIVE_RUN_DIR = run_dir
    (run_dir / "checkpoints" / "queries").mkdir(parents=True, exist_ok=True)
    RUN_LOG = (run_dir / "experiment.log").open("a", encoding="utf-8")
    _log(f"run_id={run_dir.name}")
    _log("experiment=Phase 3 Experiment C")

    a_run_dir = ROOT / "results" / "runs" / A_RUN_ID
    if not a_run_dir.exists():
        raise FileNotFoundError(f"Experiment A run is missing: {a_run_dir}")

    queries = _read_queries()
    qrels = _load_qrels({row["query_id"] for row in queries})
    corpus = _read_corpus()
    _, rankings, ranking_performance = _validated_experiment_a(a_run_dir)
    rank_lookup = _rank_lookup(rankings)

    for query in queries:
        for method in ("dense", "hybrid"):
            for pool in CANDIDATE_POOLS:
                baseline = rank_lookup.get((query["query_id"], method, pool, "baseline"))
                reranked = rank_lookup.get((query["query_id"], method, pool, "reranked"))
                if baseline is None or reranked is None:
                    raise ValueError("Incomplete Experiment A cache")
                if len(baseline) != pool or len(reranked) != pool:
                    raise AssertionError("Experiment A candidate count mismatch")
                if {str(row["document_id"]) for row in baseline} != {str(row["document_id"]) for row in reranked}:
                    raise AssertionError("Experiment A reranked candidate set mismatch")
                if any("cross_encoder_score" not in row for row in reranked):
                    raise AssertionError("Missing Experiment A cross-encoder scores")

    smoke = _smoke_test(run_dir, queries, rank_lookup, corpus, ranking_performance)
    _atomic_json(run_dir / "smoke_test.json", smoke)
    _log(
        f"smoke_test=passed estimated_full_evidence_reranking_minutes="
        f"{smoke['estimated_full_evidence_reranking_minutes']:.1f}"
    )

    started = time.perf_counter()
    result, aggregates, performance = _evaluate(
        run_dir,
        queries,
        qrels,
        rank_lookup,
        corpus,
        ranking_performance,
    )
    statistics = _statistical_analysis(run_dir, result["metric_rows"], queries)
    validation = _validate(
        run_dir,
        queries,
        result["metric_rows"],
        result["ranking_rows"],
        result["component_rows"],
        result["performance_rows"],
        aggregates,
        statistics,
    )
    figures = _generate_figures(run_dir, aggregates, performance, statistics)
    report = _write_report(run_dir, smoke, aggregates, performance, statistics, validation, figures)
    final_summary = _write_phase3_summary()

    protected_after = {path.name: _sha256_tree(path) for path in protected_paths}
    protected_report_hash_after = hashlib.sha256(protected_report.read_bytes()).hexdigest()
    if protected_before != protected_after:
        raise AssertionError("Historical run_0001 through run_0008 changed")
    if protected_report_hash != protected_report_hash_after:
        raise AssertionError("Phase 3 Experiment A report changed")

    git_commit = ""
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = ""

    code_hash = hashlib.sha256((ROOT / "experiments" / "run_experiment_c.py").read_bytes()).hexdigest()
    config = {
        "experiment": "Phase 3 Experiment C",
        "run_id": run_dir.name,
        "status": "completed",
        "dataset": "BEIR SciFact",
        "dataset_split": "official SciFact development",
        "qrels": "BeIR/scifact-qrels test split mapped to official development",
        "qrels_selection": "development evaluation only",
        "query_count": 300,
        "retrieval_methods": ["dense", "hybrid"],
        "candidate_pools": list(CANDIDATE_POOLS),
        "ranking_conditions": list(RANKING_CONDITIONS),
        "configuration_count": 30,
        "cross_encoder_source": f"results/runs/{A_RUN_ID}",
        "cross_encoder_inference_rerun": False,
        "bootstrap_resamples": BOOTSTRAP_ITERATIONS,
        "comparison_family_size": 150,
        "seed": SEED,
        "weights": {
            "cross_encoder": 0.55,
            "lexical_overlap": 0.20,
            "scientific_overlap": 0.15,
            "retrieval_rank": 0.10,
            "retrieval_score": 0.0,
        },
        "git_commit": git_commit,
        "code_sha256": code_hash,
        "official_test_claim_file_opened": False,
        "resumed": bool(resume_run_id),
    }
    _atomic_json(run_dir / "config.json", config)
    metadata = {
        "experiment": "Phase 3 Experiment C",
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "git_commit": git_commit,
        "code_sha256": code_hash,
        "official_test_claim_file_opened": False,
        "protected_runs_unchanged": True,
        "protected_phase3_report_unchanged": True,
        "run_wall_time_seconds": time.perf_counter() - started,
        "query_count": len(queries),
        "configuration_count": 30,
        "record_count": len(result["metric_rows"]),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    _atomic_json(run_dir / "environment_metadata.json", {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "peak_memory": _memory_snapshot(),
        "git_commit": git_commit,
        "code_sha256": code_hash,
        "protected_run_hashes_before": protected_before,
        "protected_run_hashes_after": protected_after,
        "protected_phase3_report_sha256_before": protected_report_hash,
        "protected_phase3_report_sha256_after": protected_report_hash_after,
        "official_test_claim_file_opened": False,
    })

    test_output = run_dir / "test_output.txt"
    import subprocess
    test_process = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "pytest", "-q", "tests/test_experiment_c.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    test_output.write_text(test_process.stdout + test_process.stderr, encoding="utf-8")
    if test_process.returncode != 0:
        raise RuntimeError(test_output.read_text(encoding="utf-8"))

    validation["test_status"] = "passed"
    validation["test_output"] = test_output.read_text(encoding="utf-8").strip()
    _atomic_json(run_dir / "validation.json", validation)
    _atomic_json(run_dir / "run_status.json", {
        "run_id": run_dir.name,
        "status": "completed",
        "query_count": len(queries),
        "configuration_count": 30,
        "record_count": len(result["metric_rows"]),
        "test_status": "passed",
        "official_test_claim_file_opened": False,
        "historical_runs_unchanged": True,
        "report": str(report.relative_to(ROOT).as_posix()),
        "final_summary": str(final_summary.relative_to(ROOT).as_posix()),
        "figures": figures,
    })
    _log(f"run_directory={run_dir}")
    _log(f"report={report}")
    _log(f"final_summary={final_summary}")
    _log("tests=passed")
    RUN_LOG.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 Experiment C.")
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume run_0009 or another run")
    args = parser.parse_args()
    try:
        main(resume_run_id=args.resume)
    except BaseException as exc:
        if ACTIVE_RUN_DIR is not None:
            _atomic_json(ACTIVE_RUN_DIR / "run_status.json", {
                "run_id": ACTIVE_RUN_DIR.name,
                "status": "failed",
                "error": str(exc),
                "official_test_claim_file_opened": False,
            })
        traceback.print_exc()
        raise
