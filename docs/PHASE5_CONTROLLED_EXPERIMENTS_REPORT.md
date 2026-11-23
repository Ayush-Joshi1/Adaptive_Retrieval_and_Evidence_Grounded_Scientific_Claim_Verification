# Phase 5 - Controlled Experiments

> Controlled official held-out test evaluation. This report distinguishes retrieval relevance from claim verdict correctness.

- Run: `run_0010`
- Protocol freeze timestamp: `2026-11-23T12:58:22.714796+00:00`
- Configuration hash: `05c8c0bf3fddebcb3b43e739760d14770a10eb0c021fd07c6b323d1387511c3e`
- Held-out test claims: 300
- The held-out claims file was first opened only after the frozen protocol record was written.
- Gold labels and evidence were not used for retrieval, reranking, evidence selection, mapping, or verdict inference.

## Research objective

Provide a final, controlled comparison of BM25, Dense, Hybrid, cross-encoder, adaptive retrieval, and evidence-grounded systems on the official SciFact held-out claims.

## Frozen system configurations

| System | Method | Configuration rationale |
|---|---|---|
| bm25_baseline | BM25 | Phase 2 sparse retrieval baseline |
| dense_baseline | Dense | Existing FAISS candidate baseline |
| hybrid_baseline | Hybrid RRF | RRF constant 60 with component depth 100 |
| hybrid_cross_encoder | Hybrid + cross-encoder | Pool-50 reranking baseline from Experiment A |
| dense_adaptive_score_gap | Dense adaptive + reranking | Score-gap threshold 0.10, depths 10/20/50 |
| hybrid_evidence_grounded | Hybrid + CE + evidence | Pool-50 evidence-aware reranking and Phase 4 grounded baseline |

The evidence-grounded system was retained despite development evidence limitations because it adds citation validation and explicit abstention, not because it was assumed superior.

## Retrieval results
Retrieval relevance metrics are **unavailable**: the official held-out claims file contains claim text only and no relevance judgments or cited evidence documents.

## Claim-verification results

- Verdict accuracy: **unavailable** (no official test gold verdict labels)
- Macro precision/recall/F1: **unavailable**
- Abstention rate: 0.466667
- Accuracy on non-abstained predictions: unavailable
- Citation validity/traceability rate: 1.000000
- Evidence sentence coverage: unavailable

## Statistical analysis

- Paired retrieval statistics are **unavailable** because no official held-out relevance judgments or gold labels exist in `claims_test.jsonl`.
- No corrected superiority or equivalence claims are made.

## Efficiency

| System | Mean retrieval (s) | Mean reranking (s) | Mean end-to-end (s) | Mean candidates |
|---|---:|---:|---:|---:|
| bm25_baseline | 0.363151 | 0.000000 | 0.363151 | 50.00 |
| dense_baseline | 0.042233 | 0.000000 | 0.042233 | 50.00 |
| hybrid_baseline | 0.461331 | 0.000000 | 0.461331 | 50.00 |
| hybrid_cross_encoder | 0.570431 | 11.247513 | 11.817944 | 50.00 |
| dense_adaptive_score_gap | 0.042903 | 3.684031 | 3.726934 | 16.40 |
| hybrid_evidence_grounded | 0.382728 | 10.567414 | 23.496942 | 50.00 |

## Validation

- Status: **passed**
- Queries: 300
- Retrieval records: 1800
- Verification records: 300
- Performance records: 1800
- No duplicate query/system records, missing systems, negative timings, or gold/prediction mixing was detected.

## Figures

- `results/runs/run_0010/figures/controlled_system_latency.png`

## Limitations

- The held-out claims file contains no evidence annotations or claim labels, so none could be used after prediction.
- Sentence-index mapping must be interpreted conservatively because official SciFact sentence indices are source-specific.
- The relationship mapper is a rule-based baseline, not validated scientific entailment.
- Confidence is an internal strength heuristic, not calibrated probability.
- Official test retrieval relevance derives document IDs in the official evidence annotations.
- The evidence-grounded verdict classifier may abstain where the baseline relationship signal is weak.

## Reproduction

```powershell
.\.venv\Scripts\python.exe experiments\run_phase5_controlled.py
```

Resume a completed/partial run:

```powershell
.\.venv\Scripts\python.exe experiments\run_phase5_controlled.py --resume run_0010
```
