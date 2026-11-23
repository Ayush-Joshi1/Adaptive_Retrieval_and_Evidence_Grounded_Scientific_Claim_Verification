# Phase 3 — Experiment B: Adaptive Retrieval

> Official SciFact development-set results only. This report is not a held-out test-set result.

- Run: `run_0008` (`C:/Users/admin/Desktop/rag project upd/results/runs/run_0008`)
- Git commit: `127c69b1e766632224fc569de2be724b48199546`; runner SHA-256: `be1045584519a1c481ca89c05a3f3142ac9d263c836974f60f78f957a7782a90`.
- Dataset: 300 official SciFact development claims.
- Qrels: BEIR `scifact-qrels` `test` split, established by the project manifest to align with official development claims.
- Dataset manifest: `manifests/dataset_manifest.json`; official held-out test claims and labels were not accessed.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, batch size 16.
- Smoke test: 3 queries, 30 method/query runs; measured estimated runtime 247.3 minutes.
- Integrity tests: passed; `tests/test_dataset_validation.py` was intentionally excluded because its split loader reads every official claim split, including the prohibited held-out claims file.

## Research objective

Test whether score-gap or confidence-based adaptive retrieval can reduce the candidate pool and cross-encoder work while preserving top-10 reranking quality, compared with fixed Top-10, Top-20, and Top-50 retrieval. The stopping rules are unsupervised and are not calibrated against evaluation qrels.

## Experimental design and methodology

- All 300 official development claims were paired across 10 configurations: Dense and Hybrid × fixed depths 10/20/50 and adaptive Score-gap/Confidence.
- Fixed configurations retrieve and rerank the full candidate pool. Adaptive configurations request depth 10 first and issue a new depth-20 or depth-50 retrieval only after the current score signal fails its stopping rule.
- The retrieval API returns a ranked prefix on each expansion. Each stage records its returned prefix, newly added IDs, cumulative backend results, and measured retrieval latency; repeated prefix work is not counted as new unique candidates.
- Score-gap stopping threshold: 0.10 on the min-max-normalized difference between the two leading retrieval scores. Confidence stopping threshold: 0.25 on the leading normalized score's z-like distance from the normalized score distribution mean. Stop at 10 or 20 when the strategy signal meets its threshold; otherwise cap at 50.
- Thresholds are the existing unsupervised values; qrels and per-query relevance metrics are not used in stopping decisions or threshold tuning.
- Dense uses the project FAISS index; Hybrid uses BM25 + Dense reciprocal-rank fusion (RRF constant 60). The cross-encoder model is loaded once per process; all inference is CPU and batch size 16.
- In Experiment B, Hybrid requests both component retrievers at the current candidate depth (10, 20, or 50); adaptive stages therefore recompute RRF with deeper component lists as they expand.
- All configurations use the same reranker and final evaluation cutoff of 10. Metrics are Hit@10, Recall@10, MRR@10, MAP@10, and nDCG@10.
- Per-query metrics, full reranked candidate lists, retrieval scores, expansion decisions, timing, checkpoint files, code hash, environment, and run metadata are stored in the run directory.

## Main results

| Configuration | Hit@10 | Recall@10 | MRR@10 | MAP@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| dense_fixed_top_10 | 0.796667 | 0.786667 | 0.657017 | 0.644347 | 0.683095 |
| dense_fixed_top_20 | 0.820000 | 0.807889 | 0.654939 | 0.641009 | 0.685401 |
| dense_fixed_top_50 | 0.830000 | 0.815556 | 0.660889 | 0.644591 | 0.690648 |
| dense_score_gap_adaptive | 0.806667 | 0.796667 | 0.664206 | 0.651194 | 0.690694 |
| dense_confidence_adaptive | 0.796667 | 0.786667 | 0.657017 | 0.644347 | 0.683095 |
| hybrid_fixed_top_10 | 0.820000 | 0.813889 | 0.661873 | 0.651618 | 0.694638 |
| hybrid_fixed_top_20 | 0.866667 | 0.857889 | 0.676565 | 0.663144 | 0.714116 |
| hybrid_fixed_top_50 | 0.846667 | 0.833889 | 0.663348 | 0.648326 | 0.697365 |
| hybrid_score_gap_adaptive | 0.843333 | 0.830556 | 0.662975 | 0.648064 | 0.696636 |
| hybrid_confidence_adaptive | 0.820000 | 0.813889 | 0.661873 | 0.651618 | 0.694638 |

## Adaptive depth and computational cost

Latency is measured per query/configuration. Cross-encoder inference time covers the actual model `predict` call; end-to-end time is retrieval plus the complete reranking call. Adaptive candidate pools are counted once by unique candidate ID, and cumulative distinct IDs plus raw backend result counts expose overlap across later stages. For fixed baselines, the depth percentages denote the configured fixed depth rather than an early-stop decision.

| Configuration | Avg final candidates retrieved/processed | Avg distinct IDs seen across stages | Avg backend results returned | Avg expansions | Stop 10 | Stop 20 | Stop 50 | Retrieval mean / median / P95 (s) | CE mean / median / P95 (s) | End-to-end mean / median / P95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_fixed_top_10 | 10.00 | 10.00 | 10.00 | 0.00 | 100.0% | 0.0% | 0.0% | 0.0333 / 0.0285 / 0.0455 | 2.7032 / 2.4599 / 4.1414 | 2.7367 / 2.4865 / 4.1770 |
| dense_fixed_top_20 | 20.00 | 20.00 | 20.00 | 0.00 | 0.0% | 100.0% | 0.0% | 0.0377 / 0.0291 / 0.0721 | 3.9123 / 3.6333 / 5.5774 | 3.9503 / 3.6701 / 5.6252 |
| dense_fixed_top_50 | 50.00 | 50.00 | 50.00 | 0.00 | 0.0% | 0.0% | 100.0% | 0.0258 / 0.0240 / 0.0395 | 8.6948 / 8.1310 / 12.9143 | 8.7213 / 8.1547 / 12.9565 |
| dense_score_gap_adaptive | 16.93 | 16.93 | 22.13 | 0.35 | 82.7% | 0.0% | 17.3% | 0.0353 / 0.0244 / 0.0848 | 3.1911 / 1.9281 / 9.1167 | 3.2267 / 1.9472 / 9.2044 |
| dense_confidence_adaptive | 10.00 | 10.00 | 10.00 | 0.00 | 100.0% | 0.0% | 0.0% | 0.0269 / 0.0229 / 0.0383 | 1.9946 / 1.8794 / 2.9689 | 2.0217 / 1.9038 / 2.9986 |
| hybrid_fixed_top_10 | 10.00 | 10.00 | 10.00 | 0.00 | 100.0% | 0.0% | 0.0% | 0.3370 / 0.2954 / 0.6749 | 2.0865 / 1.9191 / 2.9769 | 2.4237 / 2.2472 / 3.5627 |
| hybrid_fixed_top_20 | 20.00 | 20.00 | 20.00 | 0.00 | 0.0% | 100.0% | 0.0% | 0.3352 / 0.2937 / 0.6462 | 3.8961 / 3.6534 / 5.5382 | 4.2316 / 3.9034 / 6.0916 |
| hybrid_fixed_top_50 | 50.00 | 50.00 | 50.00 | 0.00 | 0.0% | 0.0% | 100.0% | 0.3289 / 0.2901 / 0.6229 | 8.7472 / 8.1181 / 12.7447 | 9.0768 / 8.4336 / 13.3528 |
| hybrid_score_gap_adaptive | 36.97 | 36.97 | 57.30 | 1.37 | 29.3% | 4.3% | 66.3% | 0.7593 / 0.6610 / 1.6500 | 6.7203 / 7.4554 / 12.2050 | 7.4800 / 8.2184 / 13.3555 |
| hybrid_confidence_adaptive | 10.00 | 10.00 | 10.00 | 0.00 | 100.0% | 0.0% | 0.0% | 0.3418 / 0.2899 / 0.6690 | 2.0756 / 1.8922 / 3.0510 | 2.4176 / 2.1719 / 3.4573 |

## Quality differences and efficiency savings

Differences below are adaptive minus the same-method fixed baseline; savings are relative to that baseline. Full differences for all five quality metrics are in `quality_efficiency_comparison.csv`.

| Method | Adaptive | Fixed baseline | Δ nDCG@10 | Candidate reduction | CE time reduction | End-to-end reduction |
|---|---|---|---:|---:|---:|---:|
| dense | dense_score_gap_adaptive | dense_fixed_top_10 | 0.007599 | -69.3% | -18.0% | -17.9% |
| dense | dense_score_gap_adaptive | dense_fixed_top_20 | 0.005293 | 15.3% | 18.4% | 18.3% |
| dense | dense_score_gap_adaptive | dense_fixed_top_50 | 0.000047 | 66.1% | 63.3% | 63.0% |
| dense | dense_confidence_adaptive | dense_fixed_top_10 | 0.000000 | 0.0% | 26.2% | 26.1% |
| dense | dense_confidence_adaptive | dense_fixed_top_20 | -0.002307 | 50.0% | 49.0% | 48.8% |
| dense | dense_confidence_adaptive | dense_fixed_top_50 | -0.007553 | 80.0% | 77.1% | 76.8% |
| hybrid | hybrid_score_gap_adaptive | hybrid_fixed_top_10 | 0.001998 | -269.7% | -222.1% | -208.6% |
| hybrid | hybrid_score_gap_adaptive | hybrid_fixed_top_20 | -0.017480 | -84.8% | -72.5% | -76.8% |
| hybrid | hybrid_score_gap_adaptive | hybrid_fixed_top_50 | -0.000730 | 26.1% | 23.2% | 17.6% |
| hybrid | hybrid_confidence_adaptive | hybrid_fixed_top_10 | 0.000000 | 0.0% | 0.5% | 0.3% |
| hybrid | hybrid_confidence_adaptive | hybrid_fixed_top_20 | -0.019478 | 50.0% | 46.7% | 42.9% |
| hybrid | hybrid_confidence_adaptive | hybrid_fixed_top_50 | -0.002728 | 80.0% | 76.3% | 73.4% |

## Statistical analysis

- Paired bootstrap: 10,000 resamples, 95% confidence intervals, seed 42.
- Paired test: two-sided Wilcoxon signed-rank.
- Holm-Bonferroni comparison family: 60 tests.
- One family contains all adaptive-vs-fixed comparisons: 2 retrieval methods × 2 adaptive strategies × 3 fixed depths × 5 metrics = 60 paired tests. Bootstrap confidence intervals use paired per-query resampling; the reported effect size is paired Cohen's dz.
- The Holm adjustment was recomputed and validated for range and monotonicity in sorted p-value order. Statistical significance uses adjusted p < 0.05.

| Method | Adaptive | Fixed | nDCG@10 difference | 95% paired bootstrap CI | Paired Cohen's dz | Holm-adjusted p | Significant |
|---|---|---|---:|---|---:|---:|---|
| dense | dense_score_gap_adaptive | dense_fixed_top_10 | 0.007599 | [-0.004688, 0.021604] | 0.06467411923214021 | 1 | no |
| dense | dense_confidence_adaptive | dense_fixed_top_10 | 0.000000 | [0.000000, 0.000000] | NA | 1 | no |
| hybrid | hybrid_score_gap_adaptive | hybrid_fixed_top_10 | 0.001998 | [-0.016139, 0.020857] | 0.012293935433266187 | 1 | no |
| hybrid | hybrid_confidence_adaptive | hybrid_fixed_top_10 | 0.000000 | [0.000000, 0.000000] | NA | 1 | no |

## Quality per unit of computation

| Configuration | nDCG@10 | Candidates processed | nDCG/candidate | nDCG/CE second | nDCG/end-to-end second |
|---|---:|---:|---:|---:|---:|
| dense_confidence_adaptive | 0.683095 | 10.00 | 0.068309 | 0.342466 | 0.337882 |
| dense_fixed_top_10 | 0.683095 | 10.00 | 0.068309 | 0.252702 | 0.249604 |
| dense_fixed_top_20 | 0.685401 | 20.00 | 0.034270 | 0.175192 | 0.173506 |
| dense_fixed_top_50 | 0.690648 | 50.00 | 0.013813 | 0.079432 | 0.079191 |
| dense_score_gap_adaptive | 0.690694 | 16.93 | 0.040789 | 0.216445 | 0.214058 |
| hybrid_confidence_adaptive | 0.694638 | 10.00 | 0.069464 | 0.334661 | 0.287320 |
| hybrid_fixed_top_10 | 0.694638 | 10.00 | 0.069464 | 0.332917 | 0.286600 |
| hybrid_fixed_top_20 | 0.714116 | 20.00 | 0.035706 | 0.183288 | 0.168758 |
| hybrid_fixed_top_50 | 0.697365 | 50.00 | 0.013947 | 0.079725 | 0.076830 |
| hybrid_score_gap_adaptive | 0.696636 | 36.97 | 0.018845 | 0.103662 | 0.093133 |

## Experiment A comparison (run_0006)

This is a descriptive per-query comparison of Experiment B fixed reranking and Experiment A's nominally matching pool-size configuration. Differences are B minus A; exact matching is checked at the per-query level. Adaptive conditions are new and have no Experiment A counterpart.

The Hybrid candidate sets are not identical between experiments: Experiment A retrieved BM25 and Dense component lists to depth 100 before truncating the fused ranking, whereas Experiment B requests each component at its fixed candidate depth (10, 20, or 50). The Hybrid metric differences below are therefore expected and must not be interpreted as a reranker-only change or an exact replication. Dense fixed-depth metrics match Experiment A exactly.

| Method | Pool | A nDCG@10 | B nDCG@10 | Δ nDCG@10 | A Hit@10 | B Hit@10 | Δ Hit@10 | Exact all-metric per-query match |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 10 | 0.683095 | 0.683095 | 0.000000 | 0.796667 | 0.796667 | 0.000000 | True |
| dense | 20 | 0.685401 | 0.685401 | 0.000000 | 0.820000 | 0.820000 | 0.000000 | True |
| dense | 50 | 0.690648 | 0.690648 | 0.000000 | 0.830000 | 0.830000 | 0.000000 | True |
| hybrid | 10 | 0.682435 | 0.694638 | 0.012203 | 0.816667 | 0.820000 | 0.003333 | False |
| hybrid | 20 | 0.692420 | 0.714116 | 0.021696 | 0.843333 | 0.866667 | 0.023333 | False |
| hybrid | 50 | 0.693866 | 0.697365 | 0.003499 | 0.836667 | 0.846667 | 0.010000 | False |

## Figures

- `results/runs/run_0008/figures/fixed_vs_adaptive_quality.png`
- `results/runs/run_0008/figures/candidate_processing_comparison.png`
- `results/runs/run_0008/figures/adaptive_stopping_depth_distribution.png`
- `results/runs/run_0008/figures/quality_vs_end_to_end_latency.png`
- `results/runs/run_0008/figures/quality_vs_inference_time.png`
- `results/runs/run_0008/figures/quality_efficiency_tradeoff.png`
- `results/runs/run_0008/figures/principal_confidence_intervals.png`

## Limitations and threats to validity

- This is a development-set experiment; it does not estimate final held-out test performance.
- The score-gap and confidence thresholds are unsupervised heuristics, not calibrated probabilities. No formal non-inferiority margin was specified, so non-significance is not proof of quality retention.
- Dense and Hybrid retrieval are evaluated with the existing corpus/index and fixed RRF constant; conclusions may not generalize to other corpora or retrieval backends.
- Adaptive stages call the backend at successively larger depths. The returned ranked prefixes overlap, and the recorded stage work and latency expose this repeated backend work rather than treating it as free.
- CPU inference latency depends on machine load, cache state, model/runtime versions, and batch behavior; smoke measurements are excluded from final aggregates.
- Fixed and adaptive configurations are independently timed; candidate scores are not shared across configurations.

## Reproducibility and artifacts

The run configuration records the dataset split, qrels mapping, thresholds, models, code hash, Git commit when available, and runtime environment. Per-query checkpoints are stored under `checkpoints/queries/` and are validated before reuse.

Start a new run (the run manager allocates the next unique `run_NNNN`):

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment_b.py
```

Resume this run after interruption:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment_b.py --resume run_0008
```

Test status: **passed**. `tests/test_dataset_validation.py` was intentionally excluded because its split loader reads every official claim split, including the prohibited held-out claims file.

Validation status: **passed**; 300 exact development IDs and 3000 metric, 3000 ranking, 3000 decision, and 3000 performance records. Candidate/stage accounting, non-negative timings, recomputed per-query and aggregate metrics, and unchanged run_0001 through run_0006 and `docs/PHASE3_REPORT.md` were verified.

## Conclusion

The best observed adaptive nDCG-per-end-to-end-second configuration was `dense_confidence_adaptive` (0.337882), with nDCG@10 0.683095. Against same-method fixed Top-50, it processed 10.00 versus 50.00 candidates/query (80.0% fewer), reduced mean cross-encoder inference time by 77.1%, and reduced mean end-to-end latency by 76.8%. Its nDCG@10 difference versus fixed Top-50 was -0.007553 (95% paired bootstrap CI [-0.030995, 0.014729], Holm-adjusted p=1). These measured results show no statistically significant difference for that comparison; lack of significance is not a formal equivalence or non-inferiority result. The evidence therefore supports the observed cost/quality trade-off above, but not a broader claim of quality equivalence.
