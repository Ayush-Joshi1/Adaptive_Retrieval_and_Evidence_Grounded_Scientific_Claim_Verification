# Phase 3.2 — Experiment A: Cross-Encoder Reranking

> This report covers Experiment A only. Phase 3 is not complete; Experiments B and C have not been run.

## Protocol and integrity

- Run: `run_0006` (`results/runs/run_0006/`).
- Dataset and split: all 300 claims from the official SciFact development split (`claims_dev.jsonl`); every configuration uses identical query IDs and qrels.
- Qrels: `BeIR/scifact-qrels` `test` split, previously aligned to the official SciFact development set in the Phase 2 manifest; this run verified 300 qrel query IDs, 339 relevance judgments, and no query IDs outside official dev.
- The official SciFact test claims file was not opened. No generative model or additional pretrained model was used.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, batch size 16; dense retrieval uses the Phase 2 embedding model and hybrid uses RRF constant 60.
- Candidate pools: 10, 20, 50; evaluation cutoffs: 1, 3, 5, 10.
- Candidate scores for the maximum pool of 50 were computed once per query and retrieval system, then reused for pool sizes 10 and 20. Cross-encoder query-document scores are independent of other candidates, so this preserves the exact rankings while avoiding redundant inference.
- Historical `results/runs/run_0001` through `run_0003` trees were SHA-256 checked before and after the run.
- Dense and hybrid baseline per-query metrics at pool 20 were compared with the Phase 2 canonical run and matched at full precision.

## CPU smoke test

- Status: passed; 3 dev queries, both retrieval systems, 50 candidates each.
- Cross-encoder load time: 7.8608 s.
- Mean reranking latency at pool 50: 10.0149 s/query-system.
- Mean cross-encoder inference at pool 50: 10.0121 s/query-system.
- Estimated full retrieval/reranking runtime: 103.4 min based on smoke retrieval and reranking measurements.
- Estimated full reranking time with score reuse: 100.15 min.
- Smoke-test measurements were used only for technical validation and runtime estimation; they are not included in the final metrics.

## Aggregate retrieval metrics

All values are means over the same 300 development queries. The machine-readable artifact retains full-precision values.

### Candidate pool 10

| Metric | Dense | Dense + reranking | Hybrid | Hybrid + reranking |
|---|---:|---:|---:|---:|
| hit@1 | 0.496666667 | 0.590000000 | 0.490000000 | 0.570000000 |
| precision@1 | 0.496666667 | 0.590000000 | 0.490000000 | 0.570000000 |
| recall@1 | 0.477222222 | 0.562611111 | 0.472888889 | 0.546777778 |
| mrr@1 | 0.496666667 | 0.590000000 | 0.490000000 | 0.570000000 |
| map@1 | 0.496666667 | 0.590000000 | 0.490000000 | 0.570000000 |
| ndcg@1 | 0.496666667 | 0.590000000 | 0.490000000 | 0.570000000 |
| hit@3 | 0.670000000 | 0.703333333 | 0.690000000 | 0.710000000 |
| precision@3 | 0.234444444 | 0.246666667 | 0.237777778 | 0.245555556 |
| recall@3 | 0.645333333 | 0.679222222 | 0.662277778 | 0.682111111 |
| mrr@3 | 0.575000000 | 0.638888889 | 0.576666667 | 0.632777778 |
| map@3 | 0.559814815 | 0.620370370 | 0.561203704 | 0.613981481 |
| ndcg@3 | 0.586901356 | 0.641340806 | 0.592094365 | 0.637171504 |
| hit@5 | 0.726666667 | 0.753333333 | 0.743333333 | 0.760000000 |
| precision@5 | 0.157333333 | 0.164000000 | 0.161333333 | 0.162000000 |
| recall@5 | 0.710888889 | 0.735444444 | 0.726388889 | 0.738222222 |
| mrr@5 | 0.588333333 | 0.650555556 | 0.588833333 | 0.644277778 |
| map@5 | 0.576282407 | 0.634384259 | 0.578000926 | 0.626949074 |
| ndcg@5 | 0.613548685 | 0.664466167 | 0.619054058 | 0.659769666 |
| hit@10 | 0.796666667 | 0.796666667 | 0.816666667 | 0.816666667 |
| precision@10 | 0.089000000 | 0.089000000 | 0.089333333 | 0.089333333 |
| recall@10 | 0.786666667 | 0.786666667 | 0.801111111 | 0.801111111 |
| mrr@10 | 0.597925926 | 0.657017196 | 0.598834656 | 0.652650794 |
| map@10 | 0.588730820 | 0.644347222 | 0.589274735 | 0.638426587 |
| ndcg@10 | 0.640199650 | 0.683094771 | 0.644193540 | 0.682434673 |

### Candidate pool 20

| Metric | Dense | Dense + reranking | Hybrid | Hybrid + reranking |
|---|---:|---:|---:|---:|
| hit@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.573333333 |
| precision@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.573333333 |
| recall@1 | 0.477222222 | 0.555944444 | 0.472888889 | 0.547611111 |
| mrr@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.573333333 |
| map@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.573333333 |
| ndcg@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.573333333 |
| hit@3 | 0.670000000 | 0.700000000 | 0.690000000 | 0.713333333 |
| precision@3 | 0.234444444 | 0.244444444 | 0.237777778 | 0.247777778 |
| recall@3 | 0.645333333 | 0.675055556 | 0.662277778 | 0.687555556 |
| mrr@3 | 0.575000000 | 0.633888889 | 0.576666667 | 0.635000000 |
| map@3 | 0.559814815 | 0.614629630 | 0.561203704 | 0.615462963 |
| ndcg@3 | 0.586901356 | 0.635995108 | 0.592094365 | 0.639769349 |
| hit@5 | 0.726666667 | 0.756666667 | 0.743333333 | 0.753333333 |
| precision@5 | 0.157333333 | 0.164666667 | 0.161333333 | 0.162666667 |
| recall@5 | 0.710888889 | 0.738333333 | 0.726388889 | 0.735166667 |
| mrr@5 | 0.588333333 | 0.646555556 | 0.588833333 | 0.644333333 |
| map@5 | 0.576282407 | 0.629912037 | 0.578000926 | 0.627606481 |
| ndcg@5 | 0.613548685 | 0.661923465 | 0.619054058 | 0.659485074 |
| hit@10 | 0.796666667 | 0.820000000 | 0.816666667 | 0.843333333 |
| precision@10 | 0.089000000 | 0.091000000 | 0.089333333 | 0.092666667 |
| recall@10 | 0.786666667 | 0.807888889 | 0.801111111 | 0.832944444 |
| mrr@10 | 0.597925926 | 0.654939153 | 0.598834656 | 0.656718254 |
| map@10 | 0.588730820 | 0.641009392 | 0.589274735 | 0.642588294 |
| ndcg@10 | 0.640199650 | 0.685401466 | 0.644193540 | 0.692419693 |

### Candidate pool 50

| Metric | Dense | Dense + reranking | Hybrid | Hybrid + reranking |
|---|---:|---:|---:|---:|
| hit@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.576666667 |
| precision@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.576666667 |
| recall@1 | 0.477222222 | 0.554277778 | 0.472888889 | 0.551944444 |
| mrr@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.576666667 |
| map@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.576666667 |
| ndcg@1 | 0.496666667 | 0.583333333 | 0.490000000 | 0.576666667 |
| hit@3 | 0.670000000 | 0.713333333 | 0.690000000 | 0.726666667 |
| precision@3 | 0.234444444 | 0.247777778 | 0.237777778 | 0.251111111 |
| recall@3 | 0.645333333 | 0.685611111 | 0.662277778 | 0.697277778 |
| mrr@3 | 0.575000000 | 0.640555556 | 0.576666667 | 0.642222222 |
| map@3 | 0.559814815 | 0.618888889 | 0.561203704 | 0.621018519 |
| ndcg@3 | 0.586901356 | 0.642335864 | 0.592094365 | 0.646880605 |
| hit@5 | 0.726666667 | 0.760000000 | 0.743333333 | 0.763333333 |
| precision@5 | 0.157333333 | 0.164000000 | 0.161333333 | 0.164666667 |
| recall@5 | 0.710888889 | 0.738888889 | 0.726388889 | 0.743888889 |
| mrr@5 | 0.588333333 | 0.651388889 | 0.588833333 | 0.651222222 |
| map@5 | 0.576282407 | 0.631893519 | 0.578000926 | 0.633626852 |
| ndcg@5 | 0.613548685 | 0.664318516 | 0.619054058 | 0.666687206 |
| hit@10 | 0.796666667 | 0.830000000 | 0.816666667 | 0.836666667 |
| precision@10 | 0.089000000 | 0.091666667 | 0.089333333 | 0.092333333 |
| recall@10 | 0.786666667 | 0.815555556 | 0.801111111 | 0.823888889 |
| mrr@10 | 0.597925926 | 0.660888889 | 0.598834656 | 0.660855820 |
| map@10 | 0.588730820 | 0.644590608 | 0.589274735 | 0.646600397 |
| ndcg@10 | 0.640199650 | 0.690647604 | 0.644193540 | 0.693866190 |

## Paired statistical analysis

- Pairwise comparisons: dense vs dense + reranking, and hybrid vs hybrid + reranking, for each candidate pool.
- Paired bootstrap: 10,000 resamples, 95% confidence intervals, seed 42.
- Tests: exact two-sided McNemar for Hit@K and two-sided Wilcoxon signed-rank for the other metrics.
- Holm-Bonferroni correction across the full family of 144 tests (all metrics, cutoffs, pools, and both base systems).
- Practical significance threshold: absolute mean metric difference ≥ 0.01. Positive differences favor reranking.

| System | Pool | Metric | Difference | 95% bootstrap CI | Holm-adjusted p | Interpretation |
|---|---:|---|---:|---|---:|---|
| dense | 10 | hit@10 | 0.000000000 | [0.000000000, 0.000000000] | 1 | no statistically or practically significant difference |
| dense | 10 | precision@10 | 0.000000000 | [0.000000000, 0.000000000] | 1 | no statistically or practically significant difference |
| dense | 10 | recall@10 | 0.000000000 | [0.000000000, 0.000000000] | 1 | no statistically or practically significant difference |
| dense | 10 | mrr@10 | 0.059091270 | [0.026499405, 0.092913790] | 0.111511555 | practical-sized improvement without corrected statistical significance |
| dense | 10 | map@10 | 0.055616402 | [0.025109706, 0.087468171] | 0.0829556934 | practical-sized improvement without corrected statistical significance |
| dense | 10 | ndcg@10 | 0.042895121 | [0.019521098, 0.067132863] | 0.073346899 | practical-sized improvement without corrected statistical significance |
| dense | 20 | hit@10 | 0.023333333 | [0.000000000, 0.050000000] | 1 | practical-sized improvement without corrected statistical significance |
| dense | 20 | precision@10 | 0.002000000 | [-0.000666667, 0.004666667] | 1 | no statistically or practically significant difference |
| dense | 20 | recall@10 | 0.021222222 | [-0.003777778, 0.046888889] | 1 | practical-sized improvement without corrected statistical significance |
| dense | 20 | mrr@10 | 0.057013228 | [0.021886243, 0.092685218] | 0.223082834 | practical-sized improvement without corrected statistical significance |
| dense | 20 | map@10 | 0.052278571 | [0.018692130, 0.086723955] | 0.276397042 | practical-sized improvement without corrected statistical significance |
| dense | 20 | ndcg@10 | 0.045201817 | [0.017424723, 0.073450873] | 0.274272745 | practical-sized improvement without corrected statistical significance |
| dense | 50 | hit@10 | 0.033333333 | [-0.003333333, 0.070000000] | 1 | practical-sized improvement without corrected statistical significance |
| dense | 50 | precision@10 | 0.002666667 | [-0.001000000, 0.006333333] | 1 | no statistically or practically significant difference |
| dense | 50 | recall@10 | 0.028888889 | [-0.007777778, 0.064444444] | 1 | practical-sized improvement without corrected statistical significance |
| dense | 50 | mrr@10 | 0.062962963 | [0.023878571, 0.102225000] | 0.226682088 | practical-sized improvement without corrected statistical significance |
| dense | 50 | map@10 | 0.055859788 | [0.018856978, 0.093208466] | 0.337217259 | practical-sized improvement without corrected statistical significance |
| dense | 50 | ndcg@10 | 0.050447955 | [0.018130201, 0.082724054] | 0.36894214 | practical-sized improvement without corrected statistical significance |
| hybrid | 10 | hit@10 | 0.000000000 | [0.000000000, 0.000000000] | 1 | no statistically or practically significant difference |
| hybrid | 10 | precision@10 | 0.000000000 | [0.000000000, 0.000000000] | 1 | no statistically or practically significant difference |
| hybrid | 10 | recall@10 | 0.000000000 | [0.000000000, 0.000000000] | 1 | no statistically or practically significant difference |
| hybrid | 10 | mrr@10 | 0.053816138 | [0.020562765, 0.087201356] | 0.19869691 | practical-sized improvement without corrected statistical significance |
| hybrid | 10 | map@10 | 0.049151852 | [0.016596994, 0.082113046] | 0.284332942 | practical-sized improvement without corrected statistical significance |
| hybrid | 10 | ndcg@10 | 0.038241133 | [0.013457982, 0.063208152] | 0.274272745 | practical-sized improvement without corrected statistical significance |
| hybrid | 20 | hit@10 | 0.026666667 | [-0.003333333, 0.056666667] | 1 | practical-sized improvement without corrected statistical significance |
| hybrid | 20 | precision@10 | 0.003333333 | [0.000000000, 0.006666667] | 1 | no statistically or practically significant difference |
| hybrid | 20 | recall@10 | 0.031833333 | [0.002777778, 0.061111111] | 1 | practical-sized improvement without corrected statistical significance |
| hybrid | 20 | mrr@10 | 0.057883598 | [0.021597718, 0.093781812] | 0.192623925 | practical-sized improvement without corrected statistical significance |
| hybrid | 20 | map@10 | 0.053313558 | [0.017771048, 0.089056524] | 0.276397042 | practical-sized improvement without corrected statistical significance |
| hybrid | 20 | ndcg@10 | 0.048226153 | [0.018648272, 0.077694610] | 0.19869691 | practical-sized improvement without corrected statistical significance |
| hybrid | 50 | hit@10 | 0.020000000 | [-0.020000000, 0.060000000] | 1 | practical-sized improvement without corrected statistical significance |
| hybrid | 50 | precision@10 | 0.003000000 | [-0.001333333, 0.007666667] | 1 | no statistically or practically significant difference |
| hybrid | 50 | recall@10 | 0.022777778 | [-0.016501389, 0.061500000] | 1 | practical-sized improvement without corrected statistical significance |
| hybrid | 50 | mrr@10 | 0.062021164 | [0.022809226, 0.099964815] | 0.208553111 | practical-sized improvement without corrected statistical significance |
| hybrid | 50 | map@10 | 0.057325661 | [0.018912708, 0.094907159] | 0.276397042 | practical-sized improvement without corrected statistical significance |
| hybrid | 50 | ndcg@10 | 0.049672651 | [0.015658196, 0.082837167] | 0.36894214 | practical-sized improvement without corrected statistical significance |

The complete 144-test output, paired tests, confidence intervals, effect sizes, and practical-significance classifications are in `statistical_tests.json`.

## CPU performance

- Cross-encoder model load: 7.8608 s.
- Dense index load: 0.1528 s; BM25 index build: 1.0236 s.
- Query latency uses one cached retrieval at depth 50 per query and system; candidate prefixes are reused for all three pool sizes.
- Pool 10/20 reranking latency includes the shared 50-candidate scoring pass; it is not represented as a separate smaller-pool inference timing.

| Stage | Mean (s) | Median (s) | P95 (s) | Throughput (query-systems/s) |
|---|---:|---:|---:|---:|
| Initial retrieval at depth 50 | 0.247085 | 0.137061 | 0.760485 | 4.047190 |
| Cross-encoder inference | 11.968542 | 11.369041 | 18.679154 | 0.083552 |
| Reranking (end-to-end call) | 11.969517 | 11.369578 | 18.679801 | 0.083546 |

| System | Pool | Condition | Mean (s) | Median (s) | P95 (s) | Throughput (q/s) |
|---|---:|---|---:|---:|---:|---:|
| dense | 10 | baseline | 0.043514 | 0.036205 | 0.072104 | 22.981218 |
| dense | 10 | reranked | 12.028522 | 10.799329 | 18.802624 | 0.083136 |
| dense | 20 | baseline | 0.043514 | 0.036205 | 0.072104 | 22.981218 |
| dense | 20 | reranked | 12.028536 | 10.799354 | 18.802636 | 0.083136 |
| dense | 50 | baseline | 0.043514 | 0.036205 | 0.072104 | 22.981218 |
| dense | 50 | reranked | 12.028582 | 10.799370 | 18.802685 | 0.083135 |
| hybrid | 10 | baseline | 0.450656 | 0.393694 | 0.973800 | 2.218986 |
| hybrid | 10 | reranked | 12.405015 | 11.934881 | 19.249225 | 0.080613 |
| hybrid | 20 | baseline | 0.450656 | 0.393694 | 0.973800 | 2.218986 |
| hybrid | 20 | reranked | 12.405033 | 11.934889 | 19.249280 | 0.080612 |
| hybrid | 50 | baseline | 0.450656 | 0.393694 | 0.973800 | 2.218986 |
| hybrid | 50 | reranked | 12.405081 | 11.934911 | 19.249307 | 0.080612 |

- Peak process working set: 701542400 bytes; current working set at measurement: 351211520 bytes (measured via Windows GetProcessMemoryInfo).
- Total run wall time, including artifact generation and validation: 126.67 minutes.
- Test suite excluding the sole official-test-file reader: **passed**, 28 passed (exit code 0).
- `tests/test_dataset_validation.py` was excluded because it reads the official SciFact test claims; running it would violate the no-test-split-access rule.

## Phase 2 comparison and limitations

- Dense and hybrid baseline metrics at pool 20 are compared query-by-query with canonical `results/runs/run_0003`; validation requires equality within 1e-12.
- The Phase 2 comparison was exact (maximum absolute per-query metric difference: 0.0 across 300 queries per method and all 24 metrics).

| Phase 2 baseline | Phase 2 Hit@10 | Experiment A Hit@10 | Phase 2 MRR@10 | Experiment A MRR@10 | Phase 2 nDCG@10 | Experiment A nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Dense, pool 20 | 0.796666667 | 0.796666667 | 0.597925926 | 0.597925926 | 0.640199650 | 0.640199650 |
| Hybrid, pool 20 | 0.816666667 | 0.816666667 | 0.598834656 | 0.598834656 | 0.644193540 | 0.644193540 |

- This is an official development-set experiment, not final held-out evaluation. Development qrels and results must not be interpreted as test-set performance.
- CPU timings are machine- and cache-dependent. Reranking scores are measured at pool 50 once and reused for smaller prefixes; no separate 10/20 inference latency is claimed.

## Reproducibility

From the repository root, run `\.venv\Scripts\python.exe experiments\run_experiment_a.py` to allocate a fresh unique run. To resume an interrupted run, pass `--resume <run_id>`; validated per-query checkpoints are reused.

## Figures

- `results/runs/run_0006/figures/dense_vs_reranking.png`
- `results/runs/run_0006/figures/hybrid_vs_reranking.png`
- `results/runs/run_0006/figures/candidate_pool_comparison.png`
- `results/runs/run_0006/figures/quality_vs_latency.png`
- `results/runs/run_0006/figures/statistical_confidence_intervals.png`

## Artifacts

- Metadata/config: `results/runs/run_0006/metadata.json`, `config.json`
- Smoke test: `results/runs/run_0006/smoke_test.json`
- Per-query rankings: `results/runs/run_0006/per_query_rankings.json` (plus JSONL)
- Per-query metrics: `results/runs/run_0006/per_query_metrics.json` (plus CSV)
- Aggregate metrics: `results/runs/run_0006/aggregate_metrics.json`
- Statistical results: `results/runs/run_0006/statistical_tests.json`
- Phase 2 baseline reproduction check: `results/runs/run_0006/baseline_reproduction_check.json`
- Query performance: `results/runs/run_0006/query_performance.csv`
- Performance summary: `results/runs/run_0006/performance_metrics.json`
- Environment metadata: `results/runs/run_0006/environment_metadata.json`
- Test output: `results/runs/run_0006/validation.json`
- Plots: `results/runs/run_0006/figures/`

## Scope

Only Experiment A was executed. No Phase 3 completion claim is made; Experiments B and C remain unstarted.
