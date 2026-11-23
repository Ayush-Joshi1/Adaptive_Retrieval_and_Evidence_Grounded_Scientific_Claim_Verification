import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = Path(__file__).resolve().parents[1] / "results" / "runs" / "run_0003"
fig_dir = run_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

with (run_dir / "aggregate_metrics.json").open("r", encoding="utf-8") as fh:
    aggregate = json.load(fh)

labels = ["bm25", "dense", "hybrid"]
for key in ["hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "mrr@10", "ndcg@10"]:
    values = [aggregate[method][key] for method in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["steelblue", "darkorange", "forestgreen"])
    ax.set_title(f"{key.upper()} by retriever")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    for label, value in zip(labels, values):
        ax.text(label, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{key.replace('@', '_')}_comparison.png", dpi=200)
    plt.close(fig)

with (run_dir / "performance_metrics.json").open("r", encoding="utf-8") as fh:
    performance = json.load(fh)
fig, ax = plt.subplots(figsize=(7, 4))
values = [performance[method]["mean"] for method in labels]
ax.bar(labels, values, color=["steelblue", "darkorange", "forestgreen"])
ax.set_title("Mean query latency by retriever")
ax.set_ylabel("Seconds")
fig.tight_layout()
fig.savefig(fig_dir / "latency_comparison.png", dpi=200)
plt.close(fig)

with (run_dir / "rrf_sensitivity.csv").open("r", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
fig, ax = plt.subplots(figsize=(8, 5))
rrf_values = [10, 30, 60, 100]
hit_values = {}
for row in rows:
    if int(row["k"]) == 10:
        hit_values[int(row["rrf_constant"])] = float(row["hit@k"])
ax.plot(rrf_values, [hit_values[c] for c in rrf_values], marker="o")
ax.set_title("RRF sensitivity: Hit@10")
ax.set_xlabel("RRF constant c")
ax.set_ylabel("Hit@10")
ax.set_xticks(rrf_values)
fig.tight_layout()
fig.savefig(fig_dir / "rrf_sensitivity.png", dpi=200)
plt.close(fig)

print(sorted(p.name for p in fig_dir.iterdir()))
