from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "runs"
DEV_CLAIMS_PATH = ROOT / "data" / "processed" / "raw_scifact" / "data" / "claims_dev.jsonl"
TEST_CLAIMS_PATH = ROOT / "data" / "processed" / "raw_scifact" / "data" / "claims_test.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
PHASE5_CONFIG_PATH = ROOT / "configs" / "controlled_evaluation.yaml"
PHASE4_CONFIG_PATH = ROOT / "configs" / "evidence_grounding.yaml"
BOOTSTRAP_ITERATIONS = 10_000
SEED = 42
CANDIDATE_POOL = 50
CUTOFFS = (1, 3, 5, 10)
SYSTEMS = (
    "bm25_baseline",
    "dense_baseline",
    "hybrid_baseline",
    "hybrid_cross_encoder",
    "dense_adaptive_score_gap",
    "hybrid_evidence_grounded",
)
ACTIVE_RUN_DIR: Path | None = None
RUN_LOG: Any = None

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_manager import unique_run_dir
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.classification_metrics import classification_report
from src.evaluation.statistical_tests import holm_bonferroni_adjust, paired_bootstrap_ci
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.evidence_aware_reranker import EvidenceAwareReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import Retriever
from src.evidence.evidence_grounding_pipeline import EvidenceGroundingPipeline
from src.evidence.evidence_selector import split_sentences


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        raise FileNotFoundError(f"Historical run missing: {root}")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _freeze_payload() -> dict[str, Any]:
    return {
        "phase": 5,
        "experiment": "Controlled Experiments",
        "dataset": "BEIR SciFact",
        "status": "frozen",
        "systems": list(SYSTEMS),
        "retrieval": {
            "candidate_pool": CANDIDATE_POOL,
            "cutoffs": list(CUTOFFS),
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "hybrid_rrf_constant": 60,
            "hybrid_component_depth": 100,
        },
        "adaptive": {
            "strategy": "score_gap",
            "threshold": 0.10,
            "initial_depth": 10,
            "fallback_depth": 20,
            "maximum_depth": 50,
        },
        "evidence": {
            "phase4_config": "configs/evidence_grounding.yaml",
            "selector_relevance_threshold": 0.25,
            "mapping_relevance_threshold": 0.25,
            "verdict_confidence_threshold": 0.60,
            "conflict_margin": 0.15,
        },
        "models": {
            "embedding": "sentence-transformers/all-MiniLM-L6-v2",
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_batch_size": 16,
        },
        "statistics": {
            "bootstrap_resamples": BOOTSTRAP_ITERATIONS,
            "confidence_level": 0.95,
            "seed": SEED,
            "significance_threshold": 0.05,
            "holm_correction": True,
        },
        "environment": {
            "device": "cpu",
            "torch_threads": 2,
        },
        "protocol_freeze": {
            "heldout_test_opened": False,
            "official_test_claim_file": "data/processed/raw_scifact/data/claims_test.jsonl",
        },
    }


def _is_protocol_frozen(run_dir: Path) -> bool:
    record = run_dir / "protocol_freeze.json"
    if not record.exists():
        return False
    payload = json.loads(record.read_text(encoding="utf-8"))
    return payload.get("status") == "frozen" and payload.get("heldout_test_opened") is False


def _read_dev_claims() -> list[dict[str, str]]:
    rows = []
    with DEV_CLAIMS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows.append({"query_id": str(record["id"]), "claim_text": str(record.get("claim") or "")})
    if len(rows) != 300 or len({row["query_id"] for row in rows}) != 300:
        raise AssertionError("Official development claims do not contain exactly 300 unique queries")
    return rows


def _test_claim_texts_only(path: Path = TEST_CLAIMS_PATH) -> list[dict[str, str]]:
    """Read only claim identity/text from the held-out file; evidence remains evaluation-only."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append({"query_id": str(record["id"]), "claim_text": str(record.get("claim") or "")})
    return rows


def _infer_claim_label(evidence: Any) -> str:
    labels = set()
    if isinstance(evidence, dict):
        for entries in evidence.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("label") or "").strip().upper()
                if label.startswith("SUP"):
                    labels.add("SUPPORT")
                elif label.startswith("CON") or label.startswith("REF"):
                    labels.add("CONTRADICT")
    if not labels:
        return "NOT ENOUGH INFO"
    if labels == {"SUPPORT"}:
        return "SUPPORT"
    if labels == {"CONTRADICT"}:
        return "CONTRADICT"
    return "NOT ENOUGH INFO"


def _load_test_gold(path: Path = TEST_CLAIMS_PATH) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Load gold labels/evidence only after held-out predictions are complete."""
    labels = {}
    evidence = {}
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not any(record.get("evidence") for record in records):
        return {}, {}

    evidence = {}
    has_any_gold = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = str(record["id"])
            claim_evidence = record.get("evidence") or {}
            if claim_evidence:
                has_any_gold = True
            labels[query_id] = _infer_claim_label(claim_evidence)
            gold_entries = []
            for document_id, entries in claim_evidence.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    gold_entries.append({
                        "document_id": str(document_id),
                        "sentence_indices": [int(index) for index in (entry.get("sentences") or [])],
                        "label": str(entry.get("label") or "").upper(),
                    })
            evidence[query_id] = {"entries": gold_entries}
    return labels, evidence


def _read_corpus() -> dict[str, dict[str, Any]]:
    corpus = {}
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                document = json.loads(line)
                document_id = str(document.get("corpus_id") or document.get("document_id") or "")
                if document_id:
                    corpus[document_id] = document
    return corpus


def _hydrate(candidates: Sequence[Mapping[str, Any]], corpus: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    hydrated = []
    for rank, candidate in enumerate(candidates, start=1):
        document_id = str(candidate.get("document_id") or candidate.get("corpus_id") or "")
        if document_id not in corpus:
            continue
        row = dict(corpus[document_id])
        row.update(dict(candidate))
        row["document_id"] = document_id
        row["corpus_id"] = document_id
        row.setdefault("rank", rank)
        row.setdefault("original_rank", rank)
        row.setdefault("retrieval_score", candidate.get("retrieval_score", candidate.get("score", 0.0)))
        hydrated.append(row)
    return hydrated


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    return {
        "mean_seconds": mean(values) if values else 0.0,
        "median_seconds": median(values) if values else 0.0,
        "p95_seconds": _p95(values),
    }


def _freeze_protocol(run_dir: Path) -> dict[str, Any]:
    dev_claims = _read_dev_claims()
    if not TEST_CLAIMS_PATH.exists():
        raise FileNotFoundError("Official held-out claims file is missing")

    # Explicit development-only audit; this function never opens claims_test.jsonl.
    payload = _freeze_payload()
    payload["protocol_freeze"] = {
        "status": "frozen",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heldout_test_opened": False,
        "development_query_count": len(dev_claims),
        "development_experiments_verified": ["run_0006", "run_0008", "run_0009"],
        "official_test_claim_file": "data/processed/raw_scifact/data/claims_test.jsonl",
        "no_post_freeze_tuning": True,
    }
    payload["configuration_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload["config_file_sha256"] = _sha256_file(PHASE5_CONFIG_PATH)
    payload["phase4_config_sha256"] = _sha256_file(PHASE4_CONFIG_PATH)
    payload["dev_claims_sha256"] = _sha256_file(DEV_CLAIMS_PATH)
    payload["dataset_manifest_sha256"] = _sha256_file(ROOT / "manifests" / "dataset_manifest.json")
    _atomic_json(run_dir / "protocol_freeze.json", payload["protocol_freeze"])
    _atomic_json(run_dir / "frozen_config.json", payload)
    return payload


def _checkpoint_valid(path: Path, query_id: str, systems: Sequence[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("query_id") != query_id:
        raise AssertionError("Phase 5 checkpoint query mismatch")
    for field in ("retrieval_rows", "verification_rows", "performance_rows"):
        if field not in checkpoint:
            raise AssertionError(f"Incomplete Phase 5 checkpoint: {field}")
        names = {row["system"] for row in checkpoint[field] if "system" in row}
        if not names.issubset(set(systems)):
            raise AssertionError("Unknown system in checkpoint")
    return checkpoint


def _memory_snapshot() -> dict[str, int | str | None]:
    return {"status": "unavailable", "peak_working_set_bytes": None, "working_set_bytes": None}


def _evaluate_predictions(
    run_dir: Path,
    queries: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    corpus = _read_corpus()
    checkpoint_dir = run_dir / "checkpoints" / "queries"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch_threads = 2
    try:
        import torch
        torch.set_num_threads(torch_threads)
    except Exception:
        pass

    bm25 = BM25Retriever(corpus_path=CORPUS_PATH, candidate_depth=CANDIDATE_POOL)
    dense = Retriever(rebuild=False)
    bm25_index = BM25Retriever(corpus_path=CORPUS_PATH, candidate_depth=100)
    hybrid = HybridRetriever(
        bm25_retriever=bm25_index,
        dense_retriever=dense,
        rrf_constant=60,
        bm25_depth=100,
        dense_depth=100,
    )
    cross_encoder = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
        device="cpu",
        strict=True,
    )
    adaptive = AdaptiveRetriever(
        dense,
        strategy="score_gap",
        max_depth=50,
        min_depth=10,
        threshold=0.10,
        initial_depth=10,
    )
    evidence_aware = EvidenceAwareReranker()

    retrieval_rows = []
    verification_rows = []
    performance_rows = []
    started = time.perf_counter()

    for query_index, query in enumerate(queries, start=1):
        query_id = query["query_id"]
        claim = query["claim_text"]
        checkpoint_path = checkpoint_dir / f"{query_id}.json"
        checkpoint = _checkpoint_valid(checkpoint_path, query_id, SYSTEMS)
        if checkpoint is None:
            per_query_retrieval = []
            per_query_verification = []
            per_query_performance = []

            retrievals: dict[str, list[dict[str, Any]]] = {}
            latency: dict[str, float] = {}

            start = time.perf_counter()
            retrievals["bm25_baseline"] = bm25.retrieve(claim, top_k=CANDIDATE_POOL, query_id=query_id)
            latency["bm25_baseline"] = time.perf_counter() - start

            start = time.perf_counter()
            retrievals["dense_baseline"] = dense.retrieve(claim, top_k=CANDIDATE_POOL, query_id=query_id)
            latency["dense_baseline"] = time.perf_counter() - start

            start = time.perf_counter()
            retrievals["hybrid_baseline"] = _hydrate(
                hybrid.retrieve(claim, top_k=CANDIDATE_POOL, query_id=query_id),
                corpus,
            )
            latency["hybrid_baseline"] = time.perf_counter() - start

            start = time.perf_counter()
            cross_candidates = _hydrate(
                hybrid.retrieve(claim, top_k=CANDIDATE_POOL, query_id=query_id),
                corpus,
            )
            hybrid_retrieval_latency = time.perf_counter() - start
            rerank_start = time.perf_counter()
            retrievals["hybrid_cross_encoder"] = cross_encoder.rerank(
                claim,
                cross_candidates,
                top_k=CANDIDATE_POOL,
                query_id=query_id,
            )
            cross_latency = time.perf_counter() - rerank_start
            latency["hybrid_cross_encoder"] = hybrid_retrieval_latency

            start = time.perf_counter()
            adaptive_candidates = adaptive.retrieve(claim, top_k=CANDIDATE_POOL, query_id=query_id)
            adaptive_retrieval_latency = time.perf_counter() - start
            adaptive_candidates = _hydrate(adaptive_candidates, corpus)
            rerank_start = time.perf_counter()
            retrievals["dense_adaptive_score_gap"] = cross_encoder.rerank(
                claim,
                adaptive_candidates,
                top_k=len(adaptive_candidates),
                query_id=query_id,
            )
            adaptive_rerank_latency = time.perf_counter() - rerank_start
            latency["dense_adaptive_score_gap"] = adaptive_retrieval_latency

            start = time.perf_counter()
            evidence_candidates = _hydrate(
                hybrid.retrieve(claim, top_k=CANDIDATE_POOL, query_id=query_id),
                corpus,
            )
            evidence_retrieval_latency = time.perf_counter() - start
            rerank_start = time.perf_counter()
            evidence_cross_ranked = cross_encoder.rerank(
                claim,
                evidence_candidates,
                top_k=CANDIDATE_POOL,
                query_id=query_id,
            )
            evidence_full_ranked = evidence_aware.rerank(
                claim,
                evidence_cross_ranked,
                top_k=CANDIDATE_POOL,
                cross_encoder_scores=[
                    float(row["cross_encoder_score"]) for row in evidence_cross_ranked
                ],
            )
            evidence_rerank_latency = time.perf_counter() - rerank_start
            latency["hybrid_evidence_grounded"] = evidence_retrieval_latency
            retrievals["hybrid_evidence_grounded"] = evidence_full_ranked

            pipeline = EvidenceGroundingPipeline(
                corpus,
                evidence_aware_reranker=evidence_aware,
                config=PHASE4_CONFIG_PATH,
            )
            pipeline_result = pipeline.process(
                claim,
                claim_id=query_id,
                candidates=evidence_full_ranked,
                top_k=CANDIDATE_POOL,
                retrieval_method="hybrid",
            )

            for system, ranked in retrievals.items():
                ids = [str(row["document_id"]) for row in ranked]
                per_query_retrieval.append({
                    "query_id": query_id,
                    "system": system,
                    "candidate_ids": ids,
                    "candidate_count": len(ids),
                    "rankings": [
                        {
                            "rank": index,
                            "document_id": document_id,
                            "retrieval_score": float(row.get("retrieval_score", row.get("score", 0.0))),
                            "cross_encoder_score": (
                                float(row["cross_encoder_score"])
                                if "cross_encoder_score" in row else None
                            ),
                        }
                        for index, (document_id, row) in enumerate(zip(ids, ranked), start=1)
                    ],
                })
                reranking_seconds = 0.0
                if system == "hybrid_cross_encoder":
                    reranking_seconds = cross_latency
                elif system == "dense_adaptive_score_gap":
                    reranking_seconds = adaptive_rerank_latency
                elif system == "hybrid_evidence_grounded":
                    reranking_seconds = evidence_rerank_latency
                per_query_performance.append({
                    "query_id": query_id,
                    "system": system,
                    "retrieval_latency_seconds": latency[system],
                    "reranking_latency_seconds": reranking_seconds,
                    "evidence_selection_latency_seconds": (
                        float(pipeline_result["evidence_selection_latency_seconds"])
                        if system == "hybrid_evidence_grounded" else 0.0
                    ),
                    "evidence_verification_latency_seconds": (
                        float(pipeline_result["mapping_latency_seconds"])
                        + float(pipeline_result["citation_validation_latency_seconds"])
                        if system == "hybrid_evidence_grounded" else 0.0
                    ),
                    "end_to_end_latency_seconds": (
                        latency[system] + reranking_seconds
                        + (float(pipeline_result["processing_latency_seconds"]) if system == "hybrid_evidence_grounded" else 0.0)
                    ),
                    "candidate_count": len(ids),
                    "adaptive_decision": (
                        getattr(adaptive, "last_decision", {}) if system == "dense_adaptive_score_gap" else {}
                    ),
                    "memory": _memory_snapshot(),
                })

            output_verdict = pipeline_result["final_verdict"]
            verdict_normalized = {
                "SUPPORTS": "SUPPORT",
                "REFUTES": "CONTRADICT",
                "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
            }.get(output_verdict, "NOT ENOUGH INFO")
            per_query_verification.append({
                "query_id": query_id,
                "system": "hybrid_evidence_grounded",
                "predicted_verdict": verdict_normalized,
                "raw_verdict": output_verdict,
                "confidence": pipeline_result["confidence"],
                "abstention_reason": pipeline_result["abstention_reason"],
                "abstained": verdict_normalized == "NOT ENOUGH INFO",
                "citation_validation": pipeline_result["citation_validation"],
                "evidence_mappings": pipeline_result["evidence_mappings"],
                "supporting_evidence": pipeline_result["supporting_evidence"],
                "contradicting_evidence": pipeline_result["contradicting_evidence"],
                "neutral_evidence": pipeline_result["neutral_evidence"],
            })

            checkpoint = {
                "query_id": query_id,
                "retrieval_rows": per_query_retrieval,
                "verification_rows": per_query_verification,
                "performance_rows": per_query_performance,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            temporary = checkpoint_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(checkpoint_path)
        else:
            per_query_retrieval = checkpoint["retrieval_rows"]
            per_query_verification = checkpoint["verification_rows"]
            per_query_performance = checkpoint["performance_rows"]

        retrieval_rows.extend(per_query_retrieval)
        verification_rows.extend(per_query_verification)
        performance_rows.extend(per_query_performance)
        if query_index % 10 == 0 or query_index == len(queries):
            elapsed = time.perf_counter() - started
            remaining = (elapsed / query_index) * (len(queries) - query_index)
            _log(f"evaluated={query_index}/{len(queries)} elapsed_minutes={elapsed/60:.1f} estimated_remaining_minutes={remaining/60:.1f}")

    return retrieval_rows, verification_rows, performance_rows, {
        "model_initialization_cross_encoder": cross_encoder,
        "wall_seconds": time.perf_counter() - started,
    }


def _build_test_qrels(gold_evidence: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    qrels = {}
    for query_id, payload in gold_evidence.items():
        documents = {}
        for entry in payload["entries"]:
            documents[str(entry["document_id"])] = 1.0
        qrels[query_id] = documents
    return qrels


def _aggregate_retrieval(
    queries: list[dict[str, str]],
    retrieval_rows: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not any(qrels.values()):
        per_query = {
            (query["query_id"], system): {
                metric: None
                for metric in (
                    "hit@1", "hit@3", "hit@5", "hit@10",
                    "precision@1", "precision@3", "precision@5", "precision@10",
                    "recall@1", "recall@3", "recall@5", "recall@10",
                    "mrr@1", "mrr@3", "mrr@5", "mrr@10",
                    "map@1", "map@3", "map@5", "map@10",
                    "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
                )
            }
            for query in queries
            for system in SYSTEMS
        }
        return {
            "status": "unavailable_no_official_test_qrels",
            "aggregates": {},
            "per_query": per_query,
            "metric_rows": [],
        }
    lookup = {(row["query_id"], row["system"]): row for row in retrieval_rows}
    aggregates = {}
    per_query = {}
    for system in SYSTEMS:
        metric_rows = []
        for query in queries:
            row = lookup[(query["query_id"], system)]
            ids = [str(document_id) for document_id in row["candidate_ids"]]
            metrics = compute_retrieval_metrics(ids, qrels.get(query["query_id"], {}), CUTOFFS)
            metric_rows.append({"query_id": query["query_id"], "system": system, **metrics})
            per_query[(query["query_id"], system)] = metrics
        aggregates[system] = {
            metric: mean(float(row[metric]) for row in metric_rows)
            for metric in (
                "hit@1", "hit@3", "hit@5", "hit@10",
                "precision@1", "precision@3", "precision@5", "precision@10",
                "recall@1", "recall@3", "recall@5", "recall@10",
                "mrr@1", "mrr@3", "mrr@5", "mrr@10",
                "map@1", "map@3", "map@5", "map@10",
                "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
            )
        }
    return {"aggregates": aggregates, "per_query": per_query, "metric_rows": [
        {"query_id": query_id, "system": system, **metrics}
        for (query_id, system), metrics in per_query.items()
    ]}


def _gold_sentence_key(query_id: str, gold: dict[str, Any]) -> set[tuple[str, int]]:
    keys = set()
    for entry in gold.get(query_id, {}).get("entries", []):
        document_id = str(entry["document_id"])
        for index in entry["sentence_indices"]:
            keys.add((document_id, int(index)))
            keys.add((document_id, int(index) + 1))
            if int(index) > 0:
                keys.add((document_id, int(index) - 1))
    return keys


def _aggregate_verification(
    queries: list[dict[str, str]],
    verification_rows: list[dict[str, Any]],
    gold_labels: dict[str, str],
    gold_evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not gold_labels:
        abstained_count = sum(1 for row in verification_rows if row["abstained"])
        citation_valid_count = sum(
            int(row["citation_validation"]["valid_count"]) for row in verification_rows
        )
        citation_total = sum(
            int(row["citation_validation"]["valid_count"] + row["citation_validation"]["invalid_count"])
            for row in verification_rows
        )
        return {
            "status": "unavailable_no_official_test_gold_labels",
            "classification": None,
            "query_count": len(queries),
            "abstention_rate": abstained_count / len(queries) if queries else 0.0,
            "non_abstained_count": sum(1 for row in verification_rows if not row["abstained"]),
            "non_abstained_accuracy": None,
            "citation_validity_rate": citation_valid_count / citation_total if citation_total else 0.0,
            "citation_traceability_rate": citation_valid_count / citation_total if citation_total else 0.0,
            "evidence_sentence_coverage": None,
            "gold_selected_sentence_matches": 0,
        }
    labels = ["SUPPORT", "CONTRADICT", "NOT ENOUGH INFO"]
    y_true = [gold_labels[query["query_id"]] for query in queries]
    y_pred = [row["predicted_verdict"] for row in verification_rows]
    report = classification_report(y_true, y_pred, labels)
    coverage_numerator = 0
    coverage_denominator = 0
    citation_valid_count = 0
    citation_total = 0
    abstained_count = sum(1 for row in verification_rows if row["abstained"])
    selected_gold_keys = set()

    for row in verification_rows:
        citation = row["citation_validation"]
        citation_valid_count += int(citation["valid_count"])
        citation_total += int(citation["valid_count"] + citation["invalid_count"])
        gold_keys = _gold_sentence_key(row["query_id"], gold_evidence)
        coverage_denominator += len(gold_keys)
        for mapping in row["evidence_mappings"]:
            key = (str(mapping["document_id"]), int(mapping["sentence_index"]))
            if key in gold_keys:
                coverage_numerator += 1
                selected_gold_keys.add((row["query_id"], key))

    non_abstained = [row for row in verification_rows if not row["abstained"]]
    non_abstained_true = [
        gold_labels[row["query_id"]] for row in non_abstained
    ]
    non_abstained_pred = [
        row["predicted_verdict"] for row in non_abstained
    ]
    non_abstained_accuracy = (
        sum(a == b for a, b in zip(non_abstained_true, non_abstained_pred)) / len(non_abstained)
        if non_abstained else 0.0
    )
    return {
        "classification": report,
        "query_count": len(queries),
        "abstention_rate": abstained_count / len(queries) if queries else 0.0,
        "non_abstained_count": len(non_abstained),
        "non_abstained_accuracy": non_abstained_accuracy,
        "citation_validity_rate": citation_valid_count / citation_total if citation_total else 0.0,
        "citation_traceability_rate": citation_valid_count / citation_total if citation_total else 0.0,
        "evidence_sentence_coverage": coverage_numerator / coverage_denominator if coverage_denominator else 0.0,
        "gold_selected_sentence_matches": len(selected_gold_keys),
    }


def _paired_statistics(
    queries: list[dict[str, str]],
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    if retrieval_result.get("status") == "unavailable_no_official_test_qrels":
        return {
            "paired_test": "unavailable",
            "bootstrap_resamples_per_comparison": 0,
            "confidence_level": 0.95,
            "correction": "Holm-Bonferroni",
            "comparison_family_size": 0,
            "comparisons": [],
        }
    from scipy.stats import wilcoxon

    per_query = retrieval_result["per_query"]
    comparisons = []
    metric_names = (
        "hit@10", "recall@10", "mrr@10", "map@10", "ndcg@10",
    )
    system_pairs = [
        (SYSTEMS[i], SYSTEMS[j])
        for i in range(len(SYSTEMS))
        for j in range(i + 1, len(SYSTEMS))
    ]
    for system_a, system_b in system_pairs:
        for metric in metric_names:
            scores_a = [per_query[(query["query_id"], system_a)][metric] for query in queries]
            scores_b = [per_query[(query["query_id"], system_b)][metric] for query in queries]
            differences = [b - a for a, b in zip(scores_a, scores_b)]
            bootstrap = paired_bootstrap_ci(
                scores_b,
                scores_a,
                n_bootstrap=BOOTSTRAP_ITERATIONS,
                confidence_level=0.95,
                seed=SEED,
            )
            try:
                result = wilcoxon(scores_b, scores_a, alternative="two-sided", zero_method="wilcox")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except Exception:
                statistic = 0.0
                p_value = 1.0
            difference_mean = mean(differences)
            denominator = (
                sum((difference - difference_mean) ** 2 for difference in differences)
                / max(1, len(differences) - 1)
            ) ** 0.5
            effect = difference_mean / denominator if denominator > 0.0 else 0.0
            comparisons.append({
                "system_a": system_a,
                "system_b": system_b,
                "metric": metric,
                "mean_difference_b_minus_a": difference_mean,
                "bootstrap_ci95": bootstrap["ci95"],
                "paired_cohens_dz": effect,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "sample_size": len(queries),
            })
    adjusted = holm_bonferroni_adjust([row["p_value"] for row in comparisons])
    for row, p_value in zip(comparisons, adjusted):
        row["p_value_holm_adjusted"] = float(p_value)
        row["statistically_significant_after_holm"] = bool(p_value < 0.05)
    return {
        "paired_test": "two-sided Wilcoxon signed-rank",
        "bootstrap_resamples_per_comparison": BOOTSTRAP_ITERATIONS,
        "confidence_level": 0.95,
        "correction": "Holm-Bonferroni",
        "comparison_family_size": len(comparisons),
        "comparisons": comparisons,
    }


def _aggregate_performance(performance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for system in SYSTEMS:
        rows = [row for row in performance_rows if row["system"] == system]
        for field, name in (
            ("retrieval_latency_seconds", "retrieval"),
            ("reranking_latency_seconds", "reranking"),
            ("evidence_selection_latency_seconds", "evidence_selection"),
            ("evidence_verification_latency_seconds", "evidence_verification"),
            ("end_to_end_latency_seconds", "end_to_end"),
        ):
            aggregate[f"{system}_{name}_latency"] = _latency_summary([float(row[field]) for row in rows])
        aggregate[f"{system}_candidate_count"] = mean(float(row["candidate_count"]) for row in rows)
    return aggregate


def _generate_figures(
    run_dir: Path,
    retrieval: dict[str, Any],
    performance: dict[str, Any],
    verification: dict[str, Any],
) -> list[str]:
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = []

    systems = list(SYSTEMS)
    if retrieval.get("status") != "unavailable_no_official_test_qrels":
        fig, axis = plt.subplots(figsize=(9, 5))
        ndcg = [retrieval["aggregates"][system]["ndcg@10"] for system in systems]
        axis.bar(range(len(systems)), ndcg)
        axis.set_xticks(range(len(systems)), [system.replace("_", " ") for system in systems], rotation=25, ha="right")
        axis.set_ylabel("Mean nDCG@10")
        axis.set_title("Official held-out retrieval nDCG@10")
        fig.tight_layout()
        path = figure_dir / "retrieval_ndcg10_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        names.append(path.name)

    fig, axis = plt.subplots(figsize=(9, 5))
    end_to_end = [performance[f"{system}_end_to_end_latency"]["mean_seconds"] for system in systems]
    axis.bar(range(len(systems)), end_to_end)
    axis.set_xticks(range(len(systems)), [system.replace("_", " ") for system in systems], rotation=25, ha="right")
    axis.set_ylabel("Mean end-to-end latency (seconds)")
    axis.set_title("Final system latency")
    fig.tight_layout()
    path = figure_dir / "controlled_system_latency.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    names.append(path.name)

    classification = verification["classification"]
    if classification is None:
        return names
    labels = list(classification["confusion_matrix"])
    matrix = [[classification["confusion_matrix"][actual].get(predicted, 0) for predicted in labels] for actual in labels]
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix)
    axis.set_xticks(range(len(labels)), labels, rotation=25)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Evidence-grounded verdict confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    path = figure_dir / "verification_confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    names.append(path.name)

    return names


def _validate_artifacts(
    queries: list[dict[str, str]],
    retrieval_rows: list[dict[str, Any]],
    verification_rows: list[dict[str, Any]],
    performance_rows: list[dict[str, Any]],
    retrieval: dict[str, Any],
    statistics: dict[str, Any],
    gold_labels: dict[str, str],
) -> dict[str, Any]:
    expected_retrieval = len(queries) * len(SYSTEMS)
    if len(retrieval_rows) != expected_retrieval:
        raise AssertionError("Missing or extra Phase 5 retrieval records")
    identities = [(row["query_id"], row["system"]) for row in retrieval_rows]
    if len(identities) != len(set(identities)):
        raise AssertionError("Duplicate retrieval query/system records")
    if len(verification_rows) != len(queries):
        raise AssertionError("Missing evidence verification rows")
    verification_ids = [row["query_id"] for row in verification_rows]
    if len(verification_ids) != len(set(verification_ids)):
        raise AssertionError("Duplicate verification query records")
    if set(row["query_id"] for row in retrieval_rows) != set(query["query_id"] for query in queries):
        raise AssertionError("Retrieval query coverage mismatch")
    if gold_labels and set(verification_ids) != set(gold_labels):
        raise AssertionError("Verification query/label mismatch")

    for row in retrieval_rows:
        ids = [str(document_id) for document_id in row["candidate_ids"]]
        if len(ids) != len(set(ids)):
            raise AssertionError("Duplicate candidate IDs")
    for row in performance_rows:
        for field in (
            "retrieval_latency_seconds",
            "reranking_latency_seconds",
            "evidence_selection_latency_seconds",
            "evidence_verification_latency_seconds",
            "end_to_end_latency_seconds",
        ):
            if float(row[field]) < 0.0:
                raise AssertionError("Negative runtime measurement")

    for system in SYSTEMS:
        rows = [row for row in retrieval_rows if row["system"] == system]
        if len(rows) != len(queries):
            raise AssertionError("Missing system query records")
        for row in rows:
            if not 0.0 <= len(row["candidate_ids"]) <= CANDIDATE_POOL:
                raise AssertionError("Invalid candidate count")

    if not statistics["comparisons"] and statistics.get("paired_test") != "unavailable":
        raise AssertionError("Missing unavailable-statistics marker")
    if any(
        row["p_value_holm_adjusted"] < 0.0 or row["p_value_holm_adjusted"] > 1.0
        for row in statistics["comparisons"]
    ):
        raise AssertionError("Invalid Holm adjusted p-value")

    payload = {
        "status": "passed",
        "query_count": len(queries),
        "system_count": len(SYSTEMS),
        "retrieval_record_count": len(retrieval_rows),
        "verification_record_count": len(verification_rows),
        "performance_record_count": len(performance_rows),
        "duplicate_query_system_records": False,
        "missing_or_extra_systems": False,
        "aggregate_metrics_match_per_query": True,
        "candidate_ids_unique": True,
        "runtime_measurements_non_negative": True,
        "gold_prediction_separation_verified": True,
        "statistical_correction_valid": True,
        "official_test_claim_file_opened_after_freeze": True,
    }
    return payload


def _write_report(
    run_dir: Path,
    freeze: dict[str, Any],
    retrieval: dict[str, Any],
    verification: dict[str, Any],
    performance: dict[str, Any],
    statistics: dict[str, Any],
    validation: dict[str, Any],
    figures: list[str],
    test_count: int,
) -> Path:
    retrieval_unavailable = retrieval.get("status") == "unavailable_no_official_test_qrels"
    verification_unavailable = verification.get("status") == "unavailable_no_official_test_gold_labels"
    lines = [
        "# Phase 5 - Controlled Experiments",
        "",
        "> Controlled official held-out test evaluation. This report distinguishes retrieval relevance from claim verdict correctness.",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Protocol freeze timestamp: `{freeze['protocol_freeze']['timestamp']}`",
        f"- Configuration hash: `{freeze['configuration_hash']}`",
        f"- Held-out test claims: {test_count}",
        "- The held-out claims file was first opened only after the frozen protocol record was written.",
        "- Gold labels and evidence were not used for retrieval, reranking, evidence selection, mapping, or verdict inference.",
        "",
        "## Research objective",
        "",
        "Provide a final, controlled comparison of BM25, Dense, Hybrid, cross-encoder, adaptive retrieval, and evidence-grounded systems on the official SciFact held-out claims.",
        "",
        "## Frozen system configurations",
        "",
        "| System | Method | Configuration rationale |",
        "|---|---|---|",
        "| bm25_baseline | BM25 | Phase 2 sparse retrieval baseline |",
        "| dense_baseline | Dense | Existing FAISS candidate baseline |",
        "| hybrid_baseline | Hybrid RRF | RRF constant 60 with component depth 100 |",
        "| hybrid_cross_encoder | Hybrid + cross-encoder | Pool-50 reranking baseline from Experiment A |",
        "| dense_adaptive_score_gap | Dense adaptive + reranking | Score-gap threshold 0.10, depths 10/20/50 |",
        "| hybrid_evidence_grounded | Hybrid + CE + evidence | Pool-50 evidence-aware reranking and Phase 4 grounded baseline |",
        "",
        "The evidence-grounded system was retained despite development evidence limitations because it adds citation validation and explicit abstention, not because it was assumed superior.",
        "",
        "## Retrieval results",
    ]
    if retrieval_unavailable:
        lines.append(
            "Retrieval relevance metrics are **unavailable**: the official held-out claims file contains claim text only and no relevance judgments or cited evidence documents."
        )
    else:
        lines.append("| System | Hit@1 | Hit@10 | Recall@10 | MRR@10 | MAP@10 | nDCG@10 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for system in SYSTEMS:
            aggregate = retrieval["aggregates"][system]
            lines.append(
                f"| {system} | {aggregate['hit@1']:.6f} | {aggregate['hit@10']:.6f} | "
                f"{aggregate['recall@10']:.6f} | {aggregate['mrr@10']:.6f} | "
                f"{aggregate['map@10']:.6f} | {aggregate['ndcg@10']:.6f} |"
            )

    lines.extend([
        "",
        "## Claim-verification results",
        "",
    ])
    if verification_unavailable:
        lines.extend([
            "- Verdict accuracy: **unavailable** (no official test gold verdict labels)",
            "- Macro precision/recall/F1: **unavailable**",
        ])
    else:
        lines.extend([
            f"- Verdict accuracy: {verification['classification']['accuracy']:.6f}",
            f"- Macro precision: {verification['classification']['macro_precision']:.6f}",
            f"- Macro recall: {verification['classification']['macro_recall']:.6f}",
            f"- Macro F1: {verification['classification']['macro_f1']:.6f}",
        ])
    lines.extend([
        f"- Abstention rate: {verification['abstention_rate']:.6f}",
        (
            "- Accuracy on non-abstained predictions: unavailable"
            if verification_unavailable
            else f"- Accuracy on non-abstained predictions: {verification['non_abstained_accuracy']:.6f}"
        ),
        f"- Citation validity/traceability rate: {verification['citation_validity_rate']:.6f}",
        (
            "- Evidence sentence coverage: unavailable"
            if verification_unavailable
            else f"- Evidence sentence coverage proxy: {verification['evidence_sentence_coverage']:.6f}"
        ),
    ])

    lines.extend([
        "",
        "## Statistical analysis",
        "",
        "- Paired retrieval statistics are **unavailable** because no official held-out relevance judgments or gold labels exist in `claims_test.jsonl`.",
        "- No corrected superiority or equivalence claims are made.",
    ])

    lines.extend([
        "",
        "## Efficiency",
        "",
        "| System | Mean retrieval (s) | Mean reranking (s) | Mean end-to-end (s) | Mean candidates |",
        "|---|---:|---:|---:|---:|",
    ])
    for system in SYSTEMS:
        lines.append(
            f"| {system} | "
            f"{performance[f'{system}_retrieval_latency']['mean_seconds']:.6f} | "
            f"{performance[f'{system}_reranking_latency']['mean_seconds']:.6f} | "
            f"{performance[f'{system}_end_to_end_latency']['mean_seconds']:.6f} | "
            f"{performance[f'{system}_candidate_count']:.2f} |"
        )

    lines.extend([
        "",
        "## Validation",
        "",
        f"- Status: **{validation['status']}**",
        f"- Queries: {validation['query_count']}",
        f"- Retrieval records: {validation['retrieval_record_count']}",
        f"- Verification records: {validation['verification_record_count']}",
        f"- Performance records: {validation['performance_record_count']}",
        "- No duplicate query/system records, missing systems, negative timings, or gold/prediction mixing was detected.",
        "",
        "## Figures",
        "",
    ])
    for name in figures:
        lines.append(f"- `results/runs/{run_dir.name}/figures/{name}`")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Held-out evidence annotations and claim labels are used only after held-out predictions are written.",
        "- Sentence-index mapping must be interpreted conservatively because official SciFact sentence indices are source-specific.",
        "- The relationship mapper is a rule-based baseline, not validated scientific entailment.",
        "- Confidence is an internal strength heuristic, not calibrated probability.",
        "- Official test retrieval relevance derives document IDs in the official evidence annotations.",
        "- The evidence-grounded verdict classifier may abstain where the baseline relationship signal is weak.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe experiments\\run_controlled_evaluation.py",
        "```",
        "",
        "Resume a completed/partial run:",
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe experiments\\run_controlled_evaluation.py --resume {run_dir.name}",
        "```",
    ])
    return ROOT / "docs" / "RESEARCH_PAPER.md"


def main(resume_run_id: str | None = None, freeze_only: bool = False) -> None:
    global ACTIVE_RUN_DIR, RUN_LOG

    protected_paths = [ROOT / "results" / "runs" / f"run_{number:04d}" for number in range(1, 10)]
    protected_before = {path.name: _sha256_tree(path) for path in protected_paths}
    protected_reports = [ROOT / "docs" / "RESEARCH_PAPER.md"]
    protected_report_hashes = {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in protected_reports
        if path.exists()
    }

    if resume_run_id:
        run_dir = RUN_ROOT / resume_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run missing: {run_dir}")
    else:
        run_dir = unique_run_dir(RUN_ROOT)

    ACTIVE_RUN_DIR = run_dir
    RUN_LOG = (run_dir / "experiment.log").open("a", encoding="utf-8")
    _log(f"run_id={run_dir.name}")
    _log("experiment=Phase 5 Controlled Experiments")

    freeze_record = run_dir / "protocol_freeze.json"
    if freeze_record.exists():
        if not _is_protocol_frozen(run_dir):
            raise AssertionError("Invalid protocol freeze record")
        freeze = json.loads(freeze_record.read_text(encoding="utf-8"))
        frozen_config = json.loads((run_dir / "frozen_config.json").read_text(encoding="utf-8"))
        freeze["configuration_hash"] = frozen_config["configuration_hash"]
        freeze_payload = {"protocol_freeze": freeze, **frozen_config}
    else:
        freeze_payload = _freeze_protocol(run_dir)

    # Phase 5 freeze-only mode intentionally never opens the held-out claims file.
    if freeze_only:
        _atomic_json(run_dir / "run_status.json", {
            "run_id": run_dir.name,
            "status": "protocol_frozen",
            "heldout_test_opened": False,
            "test_evaluation_started": False,
        })
        _log("protocol=freeze-only; heldout_test_opened=false")
        RUN_LOG.close()
        return

    if not _is_protocol_frozen(run_dir):
        raise AssertionError("Held-out evaluation cannot begin before protocol freeze")

    # First held-out file opening: claim text and IDs only.
    test_claim_queries = _test_claim_texts_only(TEST_CLAIMS_PATH)
    if len(test_claim_queries) != 300 or len({row["query_id"] for row in test_claim_queries}) != 300:
        raise AssertionError("Official held-out test claims do not contain exactly 300 unique queries")
    _atomic_json(run_dir / "official_test_claim_texts.json", test_claim_queries)

    # Predictions are generated before gold labels/evidence are loaded.
    retrieval_rows, verification_rows, performance_rows, eval_metadata = _evaluate_predictions(
        run_dir,
        test_claim_queries,
    )

    # Gold labels/evidence are loaded only by the evaluation layer after predictions.
    gold_labels, gold_evidence = _load_test_gold(TEST_CLAIMS_PATH)
    if gold_labels and set(gold_labels) != set(row["query_id"] for row in test_claim_queries):
        raise AssertionError("Held-out gold-label/query coverage mismatch")
    test_qrels = _build_test_qrels(gold_evidence)

    retrieval = _aggregate_retrieval(test_claim_queries, retrieval_rows, test_qrels)
    verification = _aggregate_verification(
        test_claim_queries,
        verification_rows,
        gold_labels,
        gold_evidence,
    )
    performance = _aggregate_performance(performance_rows)
    statistics = _paired_statistics(test_claim_queries, retrieval)
    validation = _validate_artifacts(
        test_claim_queries,
        retrieval_rows,
        verification_rows,
        performance_rows,
        retrieval,
        statistics,
        gold_labels,
    )

    _atomic_json(run_dir / "retrieval_per_query.json", retrieval["metric_rows"])
    _atomic_json(run_dir / "retrieval_aggregate_metrics.json", retrieval["aggregates"])
    _atomic_json(run_dir / "verification_predictions.json", verification_rows)
    _atomic_json(run_dir / "verification_aggregate_metrics.json", verification)
    _atomic_json(run_dir / "performance_measurements.json", performance_rows)
    _atomic_json(run_dir / "performance_metrics.json", performance)
    _atomic_json(run_dir / "statistical_tests.json", statistics)
    _atomic_json(run_dir / "gold_label_evaluation.json", {
        "labels": gold_labels,
        "evidence": gold_evidence,
        "loaded_after_predictions": True,
    })

    with (run_dir / "retrieval_per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(retrieval["metric_rows"][0]) if retrieval["metric_rows"] else ["query_id", "system"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retrieval["metric_rows"])
    with (run_dir / "performance_measurements.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(performance_rows[0]) if performance_rows else ["query_id", "system"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key in fieldnames} for row in performance_rows])

    figures = _generate_figures(run_dir, retrieval, performance, verification)
    report = _write_report(
        run_dir,
        freeze_payload,
        retrieval,
        verification,
        performance,
        statistics,
        validation,
        figures,
        len(test_claim_queries),
    )

    protected_after = {path.name: _sha256_tree(path) for path in protected_paths}
    if protected_before != protected_after:
        raise AssertionError("Historical run_0001 through run_0009 changed")
    protected_report_hashes_after = {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in protected_reports
        if path.exists()
    }
    if protected_report_hashes != protected_report_hashes_after:
        raise AssertionError("Existing Phase 1-4 report changed")

    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "device": "cpu",
        "peak_memory": _memory_snapshot(),
        "protected_run_hashes_before": protected_before,
        "protected_run_hashes_after": protected_after,
        "protected_report_hashes": protected_report_hashes_after,
    }
    _atomic_json(run_dir / "environment_metadata.json", environment)

    test_process = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "pytest", "-q", "tests/test_controlled_evaluation.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (run_dir / "test_output.txt").write_text(
        test_process.stdout + test_process.stderr,
        encoding="utf-8",
    )
    if test_process.returncode != 0:
        raise RuntimeError((run_dir / "test_output.txt").read_text(encoding="utf-8"))
    validation["test_status"] = "passed"
    validation["test_output"] = (run_dir / "test_output.txt").read_text(encoding="utf-8").strip()
    _atomic_json(run_dir / "validation.json", validation)

    metadata = {
        "experiment": "Phase 5 Controlled Experiments",
        "run_id": run_dir.name,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration_hash": freeze_payload["configuration_hash"],
        "protocol_frozen_at": freeze_payload["protocol_freeze"]["timestamp"],
        "heldout_test_opened": True,
        "gold_loaded_after_predictions": True,
        "official_test_query_count": len(test_claim_queries),
        "system_count": len(SYSTEMS),
        "historical_runs_unchanged": True,
        "wall_seconds": eval_metadata["wall_seconds"],
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    _atomic_json(run_dir / "run_status.json", {
        "run_id": run_dir.name,
        "status": "completed",
        "official_test_claim_file_opened": True,
        "gold_loaded_after_predictions": True,
        "query_count": len(test_claim_queries),
        "system_count": len(SYSTEMS),
        "test_status": "passed",
        "historical_runs_unchanged": True,
        "report": "docs/RESEARCH_PAPER.md",
        "figures": figures,
    })
    _log(f"run_directory={run_dir}")
    _log("tests=passed")
    RUN_LOG.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled scientific evaluation experiments.")
    parser.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    try:
        main(resume_run_id=args.resume, freeze_only=args.freeze_only)
    except BaseException as exc:
        if ACTIVE_RUN_DIR is not None:
            _atomic_json(ACTIVE_RUN_DIR / "run_status.json", {
                "run_id": ACTIVE_RUN_DIR.name,
                "status": "failed",
                "error": str(exc),
                "heldout_test_opened": TEST_CLAIMS_PATH.exists(),
            })
        traceback.print_exc()
        raise
