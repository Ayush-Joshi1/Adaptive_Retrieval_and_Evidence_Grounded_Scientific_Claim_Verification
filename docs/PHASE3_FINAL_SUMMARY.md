# Phase 3 Final Summary

## Scope

- Experiment A evaluated cross-encoder reranking over fixed Dense and Hybrid pools.
- Experiment B evaluated fixed and adaptive Dense/Hybrid retrieval.
- Experiment C evaluated evidence-aware reranking against cached Experiment A rankings.

## Phase 3 coverage

- Candidate pool sizes: 10, 20, 50.
- Retrieval methods: Dense and Hybrid.
- Reranking strategies: original retrieval, cross-encoder-only, evidence-aware reranking, and evidence-signal ablations.
- Adaptive retrieval: included in Experiment B fixed/adaptive depth comparisons.

## Conclusions

- See `docs/PHASE3_REPORT.md`, `docs/PHASE3_EXPERIMENT_B_REPORT.md`, and `docs/PHASE3_EXPERIMENT_C_REPORT.md`.
- Statistical significance must be read from Holm-corrected per-query analyses, not from development means alone.
- Experiment C had 2 significant comparisons among 150 Holm-corrected tests, both evidence-vs-original improvements outside nDCG@10; no evidence-aware-vs-cross-encoder nDCG@10 comparison was significant.
- Experiment C is document-level evidence-related reranking only, not sentence-level evidence grounding.

## Remaining limitations

- Official held-out test claims remain untouched.
- Evidence weights are preconfigured and not calibrated on the relevance judgments.
- Computational conclusions use development-only CPU measurements.

## Completion assessment

Phase 3 requirements are addressed across Experiments A-C. Held-out test evaluation and the evidence-grounding pipeline are outside this phase.
