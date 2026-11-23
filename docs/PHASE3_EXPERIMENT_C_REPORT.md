# Phase 3 - Experiment C: Evidence-Aware Reranking

> Official SciFact development-set results only. This report is not a held-out test-set result.

- Run: `run_0009` (`C:/Users/admin/Desktop/rag project upd/results/runs/run_0009`)
- Git commit: `127c69b1e766632224fc569de2be724b48199546`; runner SHA-256: `a93ad78237ad6c396a434f946ce4c579dadf98924095c883c8950f1ca707988b`.
- Dataset: 300 official SciFact development claims.
- Qrels: BEIR `scifact-qrels` `test` split, mapped by the manifest to official development claims.
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU.
- Experiment A artifacts: `run_0006` were reused for original rankings and cross-encoder scores; no cross-encoder inference was rerun.
- No official held-out claims or labels were accessed.

## Research objective

Test whether document-level lexical, scientific-terminology, original-rank, and retrieval-score signals improve ranking over cross-encoder-only reranking.

## Hypotheses

- H1: Evidence-aware reranking improves nDCG@10 over cross-encoder-only reranking.
- H2: Evidence-aware reranking improves ordering of scientifically relevant documents.
- H3: Combining cross-encoder relevance with evidence-related signals improves over individual signals.
- H4: Any quality improvement justifies the measured additional reranking cost.

## Methodology

- Candidate pools: 10, 20, 50.
- Retrieval methods: Dense and Hybrid.
- Ranking conditions: original retrieval, cached cross-encoder reranking, cross-encoder + lexical overlap, cross-encoder + scientific terminology overlap, and the full evidence-aware combination.
- Total configurations: 30.
- Identical candidate ID sets are used for paired conditions within each method and pool.
- Cross-encoder scores and rankings are reused verbatim from Experiment A run `run_0006`; only evidence-aware post-processing was newly computed.
- `tests/test_dataset_validation.py` was intentionally excluded because it can load all claim splits, including the held-out split.

## Evidence-aware scoring formula

For candidate i:

`E_i = w_ce * CE_norm_i + w_lex * Lex_i + w_sci * Sci_i + w_rank * Rank_norm_i + w_retrieval * Retrieval_norm_i`

- CE norm: min-max normalization of cached cross-encoder scores within the candidate pool.
- Lexical overlap: claim/document token-overlap fraction over claim tokens.
- Scientific overlap: Jaccard overlap of claim and document scientific terminology sets.
- Retrieval rank: `1 - (original_rank - 1)/(pool_size - 1)`, with top rank 1 and bottom rank 0.
- Retrieval score: min-max normalization within the candidate pool.
- Default weights: CE 0.55, lexical 0.20, scientific 0.15, rank 0.10, retrieval score 0.00.
- Ablation weights are renormalized across enabled signals. Duplicate IDs are removed, and ties are ordered by cross-encoder score, original rank, then document ID.
- A scientific terminology match indicates lexical/scientific co-occurrence only; it is not entailment or evidence grounding.

## Main results

| Configuration | Hit@10 | Recall@10 | MRR@10 | MAP@10 | nDCG@10 | Mean reranking (s) | Mean end-to-end (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense_pool_10_cross_encoder` | 0.796667 | 0.786667 | 0.657017 | 0.644347 | 0.683095 | 0.000000 | 0.043514 |
| `dense_pool_10_cross_encoder_lexical` | 0.796667 | 0.786667 | 0.657886 | 0.646048 | 0.684093 | 0.002409 | 0.045923 |
| `dense_pool_10_cross_encoder_scientific` | 0.796667 | 0.786667 | 0.659851 | 0.646958 | 0.685104 | 0.002139 | 0.045653 |
| `dense_pool_10_evidence_aware_full` | 0.796667 | 0.786667 | 0.655669 | 0.647094 | 0.684411 | 0.002174 | 0.045688 |
| `dense_pool_10_original` | 0.796667 | 0.786667 | 0.597926 | 0.588731 | 0.640200 | 0.000000 | 0.043514 |
| `dense_pool_20_cross_encoder` | 0.820000 | 0.807889 | 0.654939 | 0.641009 | 0.685401 | 0.000000 | 0.043514 |
| `dense_pool_20_cross_encoder_lexical` | 0.816667 | 0.802889 | 0.657566 | 0.645468 | 0.687332 | 0.004447 | 0.047960 |
| `dense_pool_20_cross_encoder_scientific` | 0.816667 | 0.804556 | 0.653347 | 0.639250 | 0.683351 | 0.004457 | 0.047971 |
| `dense_pool_20_evidence_aware_full` | 0.813333 | 0.802333 | 0.657328 | 0.646404 | 0.687963 | 0.004309 | 0.047823 |
| `dense_pool_20_original` | 0.796667 | 0.786667 | 0.597926 | 0.588731 | 0.640200 | 0.000000 | 0.043514 |
| `dense_pool_50_cross_encoder` | 0.830000 | 0.815556 | 0.660889 | 0.644591 | 0.690648 | 0.000000 | 0.043514 |
| `dense_pool_50_cross_encoder_lexical` | 0.826667 | 0.808056 | 0.666775 | 0.651368 | 0.693898 | 0.010898 | 0.054412 |
| `dense_pool_50_cross_encoder_scientific` | 0.826667 | 0.810556 | 0.653427 | 0.636657 | 0.683574 | 0.010909 | 0.054422 |
| `dense_pool_50_evidence_aware_full` | 0.843333 | 0.824556 | 0.668792 | 0.653728 | 0.699428 | 0.010784 | 0.054298 |
| `dense_pool_50_original` | 0.796667 | 0.786667 | 0.597926 | 0.588731 | 0.640200 | 0.000000 | 0.043514 |
| `hybrid_pool_10_cross_encoder` | 0.816667 | 0.801111 | 0.652651 | 0.638427 | 0.682435 | 0.000000 | 0.450656 |
| `hybrid_pool_10_cross_encoder_lexical` | 0.816667 | 0.801111 | 0.654849 | 0.642204 | 0.684923 | 0.002187 | 0.452843 |
| `hybrid_pool_10_cross_encoder_scientific` | 0.816667 | 0.801111 | 0.652520 | 0.638601 | 0.682470 | 0.002211 | 0.452868 |
| `hybrid_pool_10_evidence_aware_full` | 0.816667 | 0.801111 | 0.661464 | 0.647868 | 0.689493 | 0.002218 | 0.452875 |
| `hybrid_pool_10_original` | 0.816667 | 0.801111 | 0.598835 | 0.589275 | 0.644194 | 0.000000 | 0.450656 |
| `hybrid_pool_20_cross_encoder` | 0.843333 | 0.832944 | 0.656718 | 0.642588 | 0.692420 | 0.000000 | 0.450656 |
| `hybrid_pool_20_cross_encoder_lexical` | 0.840000 | 0.827111 | 0.660161 | 0.647108 | 0.694272 | 0.004224 | 0.454880 |
| `hybrid_pool_20_cross_encoder_scientific` | 0.836667 | 0.826278 | 0.653728 | 0.640005 | 0.688921 | 0.004302 | 0.454959 |
| `hybrid_pool_20_evidence_aware_full` | 0.836667 | 0.826389 | 0.661189 | 0.649504 | 0.696065 | 0.004159 | 0.454815 |
| `hybrid_pool_20_original` | 0.816667 | 0.801111 | 0.598835 | 0.589275 | 0.644194 | 0.000000 | 0.450656 |
| `hybrid_pool_50_cross_encoder` | 0.836667 | 0.823889 | 0.660856 | 0.646600 | 0.693866 | 0.000000 | 0.450656 |
| `hybrid_pool_50_cross_encoder_lexical` | 0.846667 | 0.831389 | 0.663865 | 0.651807 | 0.699044 | 0.010347 | 0.461003 |
| `hybrid_pool_50_cross_encoder_scientific` | 0.826667 | 0.813889 | 0.650825 | 0.637042 | 0.684534 | 0.010369 | 0.461026 |
| `hybrid_pool_50_evidence_aware_full` | 0.860000 | 0.844167 | 0.666468 | 0.654475 | 0.704115 | 0.011257 | 0.461913 |
| `hybrid_pool_50_original` | 0.816667 | 0.801111 | 0.598835 | 0.589275 | 0.644194 | 0.000000 | 0.450656 |

## Main nDCG@10 comparisons

| Method | Pool | Full evidence - cross-encoder | 95% paired bootstrap CI | Holm p | Significant |
|---|---:|---:|---|---:|---|
| dense | 10 | 0.001316 | [-0.003429, 0.006128] | 1 | no |
| dense | 20 | 0.002562 | [-0.005742, 0.010708] | 1 | no |
| dense | 50 | 0.008780 | [-0.000989, 0.018927] | 1 | no |
| hybrid | 10 | 0.007058 | [0.000406, 0.014370] | 1 | no |
| hybrid | 20 | 0.003645 | [-0.006654, 0.014273] | 1 | no |
| hybrid | 50 | 0.010249 | [-0.001432, 0.022185] | 1 | no |

## Ablation findings

| Method | Pool | Lexical - cross-encoder nDCG@10 | Scientific - cross-encoder nDCG@10 | Full - cross-encoder nDCG@10 |
|---|---:|---:|---:|---:|
| dense | 10 | 0.000998 | 0.002010 | 0.001316 |
| dense | 20 | 0.001931 | -0.002050 | 0.002562 |
| dense | 50 | 0.003251 | -0.007074 | 0.008780 |
| hybrid | 10 | 0.002488 | 0.000035 | 0.007058 |
| hybrid | 20 | 0.001853 | -0.003499 | 0.003645 |
| hybrid | 50 | 0.005178 | -0.009332 | 0.010249 |

## Statistical analysis

- Paired test: two-sided Wilcoxon signed-rank.
- Paired bootstrap: 10,000 resamples, 95% confidence intervals, seed 42.
- Effect size: paired Cohen's dz.
- Holm-Bonferroni family: 150 tests: 2 methods x 3 pools x 5 comparison types x 5 metrics.
- The comparison types cover original vs cross-encoder, original vs full evidence, cross-encoder vs full evidence, cross-encoder vs lexical ablation, and cross-encoder vs scientific ablation.
- No result is significant unless the Holm-adjusted p-value is below 0.05.
- Across the 150-test family, 2 comparisons remain significant after Holm correction: Dense pool-10 full evidence vs original retrieval for MAP@10 (+0.058363; adjusted p=0.036981), and Dense pool-50 full evidence vs original retrieval for MRR@10 (+0.070866; adjusted p=0.036073). Neither is a full-evidence vs cross-encoder nDCG@10 comparison.

## Efficiency

- Smoke test: 3 queries, 54 evidence-aware reranking runs.
- Mean measured evidence reranking latency: 0.004939 seconds per run.
- Cross-encoder model inference was not rerun; all cached scores came from Experiment A.
- End-to-end latency includes Experiment A's measured retrieval latency plus newly measured evidence post-processing latency.
- Memory snapshots are recorded in `performance_measurements.json`.

## Figures

- `results/runs/run_0009/figures/baseline_vs_cross_encoder_vs_evidence_aware_ndcg10.png`
- `results/runs/run_0009/figures/performance_across_candidate_pool_sizes.png`
- `results/runs/run_0009/figures/individual_evidence_signal_contribution.png`
- `results/runs/run_0009/figures/quality_vs_latency.png`
- `results/runs/run_0009/figures/principal_confidence_intervals.png`

## Limitations and threats to validity

- This is a development-set experiment. It does not estimate held-out test performance.
- Existing weights are project-configured hypotheses, not calibrated probabilities.
- Scientific terminology overlap is document-level lexical overlap, not entailment or sentence-level evidence grounding.
- Contradictory documents are not automatically penalized.
- Hybrid comparison reuse uses Experiment A's component and RRF configuration; candidate sets are exact for paired retrieval/pool conditions.
- Cross-encoder inference cost is excluded because scores were reused; evidence reranking overhead is measured directly.

## Experiment A and B comparison

- Experiment A's cached Dense and Hybrid cross-encoder rankings and scores are the direct cross-encoder baseline.
- Experiment B adaptive candidate pools are not directly compared because Experiment C uses fixed 10/20/50 pools and cached fixed-pool configurations.
- Fixed-pool Experiment C results should be interpreted alongside Experiment B's development-only adaptive measurements.

## Validation and reproduction

- Validation status: **passed**.
- Records: 9000 metric, 9000 ranking, 5400 component-score, and 9000 performance records.
- The report and artifacts preserve run-0001 through run-0008 through SHA-256 checks.

Start a new run:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment_c.py
```

Resume this run:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment_c.py --resume run_0009
```

## Conclusion

The best observed nDCG@10 configuration was `hybrid_pool_50_evidence_aware_full` (nDCG@10 0.704115). Statistical conclusions are limited by the corrected paired tests and the development-only data. No scientific entailment or final held-out conclusion is claimed.
