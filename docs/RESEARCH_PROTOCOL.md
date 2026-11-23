# Research Protocol

## 1. Research motivation

The project studies the interaction between retrieval depth, reranking, and evidence-grounded claim verification in scientific fact checking. The central objective is to separate retrieval quality from downstream generation quality while maintaining rigorous evaluation standards.

## 2. Main research question

How do retrieval depth and reranking affect evidence-grounded scientific claim verification, and which measured gains are attributable to retrieval rather than generation behavior?

## 3. Sub-research questions

1. Does deeper retrieval improve the probability that a relevant scientific abstract appears in the top-K set?
2. Does reranking improve the ordering of candidate documents without changing the underlying candidate pool?
3. Are observed verdict improvements associated with stronger retrieval and evidence grounding?
4. Do sentence-level evidence representations alter verdicts compared with full-document prompts?

## 4. Dataset and split strategy

The dataset is BEIR SciFact, using the official SciFact claim-release files and BEIR qrels. The split mapping is not assumed to be identical across sources. The project must use the original SciFact split files to identify train, dev and test sets.

Key findings from Phase 1:

- BEIR `train` qrels match official SciFact `claims_train.jsonl`.
- BEIR `test` qrels match official SciFact `claims_dev.jsonl`.
- Official SciFact `claims_test.jsonl` has no overlap with BEIR queries or qrels and must remain untouched for final evaluation.

Therefore, the project should use the official SciFact dev claims for tuning and the official SciFact test claims for the final held-out evaluation. The historical 50-query sample is preserved as an exploratory benchmark but must not be treated as a final test set.

## 5. Existing baseline architecture

The current baseline architecture remains:

- Query and corpus encoded with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS `IndexFlatIP` for exact nearest-neighbor retrieval
- Retrieval depths K = 1, 3, 5, 10
- Optional cross-encoder reranking over FAISS top-10 candidates
- Gemini generation for verdict classification

Phase 2 adds a retrieval-only comparison using the same official SciFact dev split and BEIR qrels:

- Sparse retrieval via BM25 (`rank-bm25` if available, fallback implementation otherwise)
- Dense retrieval via normalized MiniLM + FAISS `IndexFlatIP`
- Hybrid retrieval via reciprocal rank fusion (RRF)

This phase does not introduce generation or cross-encoder reranking.

## 6. Evaluation metrics and exact definitions

For retrieval, the project uses the following definitions.

- Hit@K: 1 if at least one relevant document appears in the first K retrieved results, otherwise 0.
- Precision@K: relevant retrieved documents in the first K divided by K.
- Recall@K: relevant retrieved documents in the first K divided by all relevant documents for that query. If a query has no qrels, recall is treated as 0.0 for the metric summary.
- MRR@K: reciprocal rank of the first relevant document within K.
- MAP@K: average precision over the retrieved prefix K.
- nDCG@K: discounted cumulative gain normalized by the ideal ordering under the available relevance scale.

For classification, the project supports the labels `SUPPORT`, `CONTRADICT`, and `NOT ENOUGH INFO`.

- Accuracy: fraction of correct predictions
- Macro precision: mean per-class precision
- Macro recall: mean per-class recall
- Macro F1: mean per-class F1
- Confusion matrix: row-normalized or raw counts by actual class and predicted class, depending on the reporting context
- Per-class metrics: precision, recall, F1 and support for each class

## 7. Experimental controls

The project must apply the following controls:

- No tuning on official test labels or qrels
- No model selection using BEIR test qrels
- No data leakage across train/dev/test splits
- Stable random seed for any sampling or evaluation support
- Unique run IDs for each experiment
- Raw outputs preserved separately from derived analysis files

## 8. Reproducibility requirements

Every run must record:

- dataset name and split
- source/revision of data where known
- query IDs and corpus identifiers
- embedding model ID and revision
- reranker model ID and revision
- retrieval depth
- random seed
- Python version
- dependency versions
- device and timestamps
- git commit hash when available
- metrics and generation settings where relevant

Result directories must be unique and must not overwrite prior experiment outputs silently.

## 9. Statistical analysis plan

Phase 1 prepares reusable statistical functions for:

- query-level paired bootstrap confidence intervals
- paired comparison of retrieval systems
- McNemar tests for paired classification outcomes

Inferential comparisons are only valid when the results are matched and complete. If the data are partial, unpaired, or missing, the result must be reported as a limitation rather than inferred as a valid significance test.

## 10. Known limitations

- The repository historically treated the BEIR qrels test split as a development benchmark, which is a validity issue.
- Official SciFact test labels are not yet used in the current phase and remain reserved.
- The project has no final held-out evaluation manifest yet.
- Historical outputs remain preserved; no generation calls were executed in this phase.
- The project still requires a careful run registry to prevent accidental overwrites and to keep raw outputs traceable.
