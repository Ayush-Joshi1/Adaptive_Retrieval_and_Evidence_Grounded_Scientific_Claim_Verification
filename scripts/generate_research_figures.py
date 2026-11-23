from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
RUN_3 = ROOT / "results" / "runs" / "run_0003"
RUN_6 = ROOT / "results" / "runs" / "run_0006"
RUN_8 = ROOT / "results" / "runs" / "run_0008"
RUN_9 = ROOT / "results" / "runs" / "run_0009"
RUN_10 = ROOT / "results" / "runs" / "run_0010"


def save_and_close(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def architecture_diagrams() -> None:
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.axis("off")
    labels = [
        "Scientific claim",
        "Dense / BM25 / Hybrid retrieval",
        "Adaptive pool selection",
        "Cross-encoder reranking",
        "Evidence sentence selection",
        "Citation validation",
        "Support / Refute / Abstain",
    ]
    x_positions = [0.02, 0.18, 0.35, 0.51, 0.68, 0.83, 0.96]
    for x, label in zip(x_positions, labels):
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eaf2f8", "edgecolor": "#5d6d7e"},
        )
    for start, end in zip(x_positions[:-1], x_positions[1:]):
        axis.annotate("", xy=(end - 0.035, 0.5), xytext=(start + 0.055, 0.5),
                      arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#34495e"})
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("End-to-End Evidence-Grounded Claim Verification Architecture")
    save_and_close(fig, "evidence_grounding_architecture.png")


def retrieval_comparison() -> None:
    methods = ["BM25", "Dense", "Hybrid"]
    metrics = {
        "Hit@10": [0.7367, 0.7967, 0.8167],
        "Recall@10": [0.7229, 0.7867, 0.8011],
        "MRR@10": [0.5169, 0.5979, 0.5988],
        "MAP@10": [0.5015, 0.5887, 0.5893],
        "nDCG@10": [0.5585, 0.6402, 0.6442],
    }
    x = range(len(methods))
    fig, axis = plt.subplots(figsize=(10, 5.5))
    width = 0.15
    for index, (metric, values) in enumerate(metrics.items()):
        positions = [value + (index - 2) * width for value in x]
        axis.bar(positions, values, width=width, label=metric)
    axis.set_xticks(list(x), methods)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean score")
    axis.set_title("Retrieval Performance on the Official Study Split")
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.25)
    save_and_close(fig, "retrieval_performance_comparison.png")


def rrf_sensitivity() -> None:
    rrf = json.loads(
        (RUN_3 / "rrf_sensitivity.json").read_text(encoding="utf-8")
    ) if (RUN_3 / "rrf_sensitivity.json").exists() else None
    if rrf is None:
        import csv
        rrf = list(csv.DictReader((RUN_3 / "rrf_sensitivity.csv").open("r", encoding="utf-8")))
        values = {
            "c": [int(row["rrf_constant"]) for row in rrf],
            "Hit@5": [float(row["hit@5"]) for row in rrf],
            "Hit@10": [float(row["hit@10"]) for row in rrf],
            "Hit@20": [float(row["hit@20"]) for row in rrf],
        }
    else:
        values = {
            "c": [int(row["rrf_constant"]) for row in rrf],
            "Hit@5": [float(row["hit@5"]) for row in rrf],
            "Hit@10": [float(row["hit@10"]) for row in rrf],
            "Hit@20": [float(row["hit@20"]) for row in rrf],
        }
    fig, axis = plt.subplots(figsize=(8.5, 5))
    for metric in ("Hit@5", "Hit@10", "Hit@20"):
        axis.plot(values["c"], values[metric], marker="o", label=metric)
    axis.set_xlabel("RRF constant c")
    axis.set_ylabel("Hit rate")
    axis.set_ylim(0.7, 0.95)
    axis.set_title("Hybrid RRF Sensitivity on the Official Study Split")
    axis.legend()
    axis.grid(alpha=0.25)
    save_and_close(fig, "rrf_sensitivity.png")


def reranking_findings() -> None:
    dense = [0.640199650, 0.683094771, 0.685401466, 0.690647604]
    hybrid = [0.644193540, 0.682434673, 0.692419693, 0.693866190]
    pools = ["Baseline 10", "Rerank 10", "Rerank 20", "Rerank 50"]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(pools, dense, marker="o", label="Dense")
    axis.plot(pools, hybrid, marker="o", label="Hybrid")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Cross-Encoder Reranking Findings")
    axis.set_ylim(0.62, 0.71)
    axis.legend()
    axis.grid(alpha=0.25)
    save_and_close(fig, "reranking_findings.png")


def adaptive_efficiency() -> None:
    configurations = [
        "Dense fixed 10",
        "Dense adaptive 10/50",
        "Dense fixed 50",
        "Hybrid fixed 10",
        "Hybrid adaptive 10/50",
        "Hybrid fixed 50",
    ]
    ndcg = [0.683095, 0.690694, 0.690648, 0.694638, 0.696636, 0.697365]
    candidates = [10, 16.93, 50, 10, 36.97, 50]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(configurations, ndcg)
    axes[0].set_xlabel("Mean nDCG@10")
    axes[0].set_title("Adaptive vs Fixed Quality")
    axes[0].set_xlim(0.66, 0.71)
    axes[1].barh(configurations, candidates)
    axes[1].set_xlabel("Mean candidates processed")
    axes[1].set_title("Candidate Processing")
    fig.suptitle("Adaptive Retrieval Efficiency")
    save_and_close(fig, "adaptive_retrieval_efficiency.png")


def evidence_workflow() -> None:
    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.axis("off")
    labels = [
        "Claim",
        "Retrieved documents",
        "Relevant sentences",
        "Rule-based mapping",
        "Citation validation",
        "Verdict / abstention",
    ]
    x_positions = [0.05, 0.23, 0.41, 0.59, 0.77, 0.95]
    for x, label in zip(x_positions, labels):
        axis.text(
            x, 0.5, label, ha="center", va="center", fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8f9f9", "edgecolor": "#566573"},
        )
    for start, end in zip(x_positions[:-1], x_positions[1:]):
        axis.annotate("", xy=(end - 0.035, 0.5), xytext=(start + 0.035, 0.5),
                      arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#34495e"})
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Evidence-Grounding and Abstention Workflow")
    save_and_close(fig, "evidence_grounding_workflow.png")


def controlled_evaluation_latency() -> None:
    performance = json.loads((RUN_10 / "performance_metrics.json").read_text(encoding="utf-8"))
    systems = [
        "BM25",
        "Dense",
        "Hybrid",
        "Hybrid CE",
        "Dense adaptive",
        "Hybrid evidence",
    ]
    keys = [
        "bm25_baseline",
        "dense_baseline",
        "hybrid_baseline",
        "hybrid_cross_encoder",
        "dense_adaptive_score_gap",
        "hybrid_evidence_grounded",
    ]
    latency = [performance[f"{key}_end_to_end_latency"]["mean_seconds"] for key in keys]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(systems, latency)
    axis.set_ylabel("Mean end-to-end latency (seconds/query)")
    axis.set_title("Controlled Held-Out Pipeline Latency")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, latency):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                  ha="center", va="bottom", fontsize=9)
    save_and_close(fig, "controlled_evaluation_latency.png")


def evidence_aware_findings() -> None:
    pool = [10, 20, 50]
    dense_ce = [0.683095, 0.685401, 0.690648]
    dense_full = [0.684411, 0.687963, 0.699428]
    hybrid_ce = [0.682435, 0.692420, 0.693866]
    hybrid_full = [0.689493, 0.696065, 0.704115]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(pool, dense_ce, marker="o", label="Cross-encoder")
    axes[0].plot(pool, dense_full, marker="o", label="Full evidence-aware")
    axes[0].set_title("Dense retrieval")
    axes[1].plot(pool, hybrid_ce, marker="o", label="Cross-encoder")
    axes[1].plot(pool, hybrid_full, marker="o", label="Full evidence-aware")
    axes[1].set_title("Hybrid retrieval")
    for axis in axes:
        axis.set_xlabel("Candidate pool")
        axis.set_ylabel("Mean nDCG@10")
        axis.set_ylim(0.66, 0.71)
        axis.legend()
        axis.grid(alpha=0.25)
    fig.suptitle("Evidence-Aware Study Findings")
    save_and_close(fig, "evidence_aware_findings.png")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    architecture_diagrams()
    retrieval_comparison()
    rrf_sensitivity()
    reranking_findings()
    adaptive_efficiency()
    evidence_workflow()
    controlled_evaluation_latency()
    evidence_aware_findings()
    print("Generated research figures:")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
