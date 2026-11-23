# Phase 3 Coverage Audit

This audit verifies that the completed Phase 3 roadmap is represented in implementation and reports without rerunning Experiments A-C.

## Experiment coverage

| Requirement | Status | Evidence |
|---|---|---|
| Candidate pools 10, 20, 50 | Complete | Experiment A and C fixed-pool runs; Experiment B fixed depths |
| Dense retrieval | Complete | Run 0006, 0008, and 0009 |
| Hybrid retrieval | Complete | Run 0006, 0008, and 0009 |
| Cross-encoder reranking | Complete | Run 0006 and 0009 |
| Adaptive K: 10, 20, 50 | Complete | Run 0008 |
| Evidence-aware reranking | Complete | Run 0009 |
| Signal ablations | Complete | Run 0009: cross-encoder + lexical, cross-encoder + scientific, full combination |
| Paired statistics and confidence intervals | Complete | 10,000 paired bootstrap resamples and Holm-Bonferroni analyses |
| Reusable cached scores | Complete | Experiment C reused run 0006 rankings/scores |
| Phase 3 reports | Complete | Reports for A, B, C, and final summary |

## Findings

- Experiment A established fixed-pool baseline and cross-encoder reranking.
- Experiment B established fixed/adaptive retrieval efficiency and quality trade-offs.
- Experiment C established a document-level evidence-aware reranking baseline with 30 fixed configurations.
- No Phase 3 result establishes sentence-level evidence grounding or scientific entailment.
- The held-out official test claims remain outside the Phase 3 evaluation protocol.

## Historical integrity

The preserved run directories are `run_0001` through `run_0009`. Phase 3 report files are retained and were not overwritten to create this audit.
