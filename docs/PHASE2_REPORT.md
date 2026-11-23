# Phase 2 Retrieval Comparison Report

## Canonical run and integrity status

The canonical experimental artifact for Phase 2 is `results/runs/run_0003`, created as a new run rather than overwriting the historical `run_0001` and `run_0002` directories. The corrected comparison uses the official 300-query SciFact dev claim set and the matching BEIR `scifact-qrels` test split, with the official SciFact test claims intentionally excluded.

Integrity checks confirm:

- `data/processed/raw_scifact/data/claims_dev.jsonl` contains 300 official dev claims.
- `BeIR/scifact-qrels` test rows match the official dev claim IDs.
- Historical runs remain untouched.
- The report below references the corrected run directory and the corrected aggregate metrics.

## Executive summary

This Phase 2 comparison evaluates BM25, dense retrieval, and hybrid reciprocal-rank fusion over the official SciFact dev split. The key result is that hybrid retrieval is the strongest method on the selected retrieval metrics, but the improvement over dense retrieval is modest and not statistically significant under the paired tests used here.

The corrected dev-set results are consistent with the saved artifacts in `results/runs/run_0003` and were generated without modifying the official SciFact test split or introducing new retrieval architecture.

## Data and split

- Dataset: BEIR SciFact / official SciFact claim files
- Development queries used: 300 official SciFact dev claims
- Eligible queries: 300
- Excluded queries: 0
- Qrels source: `BeIR/scifact-qrels` test split mapped to official dev claims
- Test claims reserved: official SciFact test claims were not used in this phase

## Retrieval systems

### BM25

- Implementation: `src/retrieval/bm25_retriever.py`
- Indexing: one BM25 index construction per retriever instance reused across queries
- Parameters: `k1 = 1.5`, `b = 0.75`, lowercase tokenization, no aggressive stemming
- Output: standardized retrieval results with original corpus IDs preserved

### Dense

- Implementation: `src/retrieval/retriever.py`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Index: FAISS `IndexFlatIP` over L2-normalized embeddings

### Hybrid

- Implementation: `src/retrieval/hybrid_retriever.py`
- Method: reciprocal rank fusion with `RRF(d) = sum(1 / (c + rank(d)))`
- Default constant: `c = 60`
- Candidate depths: BM25=100, dense=100

## Corrected aggregate metrics

The following values are the mean metric over the 300 official dev queries as saved in `results/runs/run_0003/aggregate_metrics.json`.

| Metric | BM25 | Dense | Hybrid |
|---|---:|---:|---:|
| Hit@1 | 0.430 | 0.497 | 0.490 |
| Hit@3 | 0.557 | 0.670 | 0.690 |
| Hit@5 | 0.630 | 0.727 | 0.743 |
| Hit@10 | 0.737 | 0.797 | 0.817 |
| Hit@20 | 0.793 | 0.843 | 0.877 |
| Precision@10 | 0.080 | 0.089 | 0.089 |
| Recall@10 | 0.723 | 0.787 | 0.801 |
| MRR@10 | 0.517 | 0.598 | 0.599 |
| MAP@10 | 0.501 | 0.589 | 0.589 |
| nDCG@10 | 0.558 | 0.640 | 0.644 |

## Paired statistical comparison

The paired inference used the same 300 dev queries for each system and followed the same query-level metric definition. For each pair of systems we computed:

- paired bootstrap 95% confidence intervals for the mean metric difference using 10,000 resamples and a fixed seed of 42,
- paired differences in mean performance,
- paired significance tests appropriate to the metric type,
- Holm-Bonferroni correction across the metric family within each system pair.

The analysis compared:

- BM25 vs Dense
- BM25 vs Hybrid
- Dense vs Hybrid

Summary of corrected results at `K = 10`:

| Pair | Metric | Mean difference | 95% bootstrap CI | Adjusted p-value | Interpretation |
|---|---|---:|---:|---:|---|
| BM25 vs Dense | MRR@10 | -0.0810 | [-0.1267, -0.0344] | 0.0014 | Dense is better; significant after correction |
| BM25 vs Dense | MAP@10 | -0.0872 | [-0.1330, -0.0413] | 0.0014 | Dense is better; significant after correction |
| BM25 vs Hybrid | hit@10 | -0.0800 | [-0.1233, -0.0367] | 2.24e-07 | Hybrid is better; significant after correction |
| BM25 vs Hybrid | MRR@10 | -0.0819 | [-0.1149, -0.0493] | 2.24e-07 | Hybrid is better; significant after correction |
| Dense vs Hybrid | hit@10 | -0.0200 | [-0.0600, 0.0200] | 1.0000 | No statistically significant difference |
| Dense vs Hybrid | MRR@10 | -0.0009 | [-0.0334, 0.0316] | 1.0000 | No statistically significant difference |

These numbers show a clear improvement of both dense and hybrid methods over BM25, but the dense-vs-hybrid gap is not statistically distinguishable on the matched-query dev set under the corrected analysis. No significance claim is made beyond the supported results above.

## RRF sensitivity analysis

The existing sweep was verified on actual dev-set retrieval outputs for `c = [10, 30, 60, 100]` and `K = [5, 10, 20]`.

The saved sensitivity table in `results/runs/run_0003/rrf_sensitivity.csv` reports the following hit-rate values:

| c | Hit@5 | Hit@10 | Hit@20 |
|---:|---:|---:|---:|
| 10 | 0.773 | 0.840 | 0.900 |
| 30 | 0.747 | 0.827 | 0.893 |
| 60 | 0.743 | 0.817 | 0.877 |
| 100 | 0.733 | 0.817 | 0.870 |

This sensitivity sweep indicates that the default `c = 60` setting remains competitive and stable, but the variation across `c` values is modest relative to the overall difference between methods. The strongest results are observed at `K=20`, while the `K=10` and `K=20` values are close across the tested `c` values.

## Performance measurements

Steady-state measurements were collected after model loading and a small warm-up pass to separate initialization overhead from query latency. The canonical results are stored in `results/runs/run_0003/performance_metrics.json`.

| Method | Mean latency (s/query) | Median latency (s/query) | P95 latency (s/query) | QPS | Index build time (s) | Peak memory (MB) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.405 | 0.385 | 0.683 | 2.47 | 1.58 | 1.09 |
| Dense | 0.063 | 0.061 | 0.085 | 15.93 | 0.18 | 0.36 |
| Hybrid | 0.605 | 0.567 | 0.998 | 1.65 | 1.76 | 1.46 |

Interpretation:

- Dense retrieval is the fastest method in steady-state query latency.
- BM25 is slower than dense retrieval but still efficient for a CPU-only sparse baseline.
- Hybrid retrieval is the slowest because it issues both sparse and dense candidate retrieval calls and fuses the results.
- The BM25 index was constructed once and reused across queries; no additional indexing per query occurred.

## Metric completeness and validation

The final comparison includes all required metrics for the selected retrieval benchmark:

- Hit@K
- Precision@K
- Recall@K
- MRR
- MAP
- nDCG

These were validated both against synthetic checks and the actual dev-set qrels. The saved per-query metrics and aggregate metrics reflect the same values, and the canonical run directory preserves the measurement provenance for each field. No unavailable metrics were silently omitted.

## Artifacts and reproducibility

Canonical run directory:

- `results/runs/run_0003/`

Saved artifacts:

- `aggregate_metrics.json`
- `metadata.json`
- `performance_metrics.json`
- `per_query_metrics.csv`
- `rrf_sensitivity.csv`
- `statistical_tests.json`
- `figures/`

Reproduction commands:

```powershell
python -m pytest -q
python experiments/run_retrieval_comparison.py
python scripts/generate_phase2_figures.py
```

## Limitations

- This report evaluates only the official SciFact development split and does not use the official SciFact test claims.
- No cross-encoder reranking, new embedding models, or generation experiments were introduced.
- The latency values are steady-state retrieval measurements after warm-up and exclude external model-download overhead.
- The conclusions are limited to retrieval quality and latency, not downstream claim verification performance.

## Conclusion

The corrected canonical Phase 2 evaluation supports the following findings:

1. BM25 remains a useful sparse baseline, but dense retrieval is more competitive on the official dev split.
2. Hybrid retrieval has the strongest retrieval quality on the swept metrics, but its improvement over dense retrieval is modest and statistically not supported in the corrected pairwise dev-set comparison.
3. The corrected official-dev run is `results/runs/run_0003`, and the reported values match the saved artifacts.

This is the final canonical Phase 2 retrieval comparison for the approved dev split without altering the underlying retrieval architecture.
