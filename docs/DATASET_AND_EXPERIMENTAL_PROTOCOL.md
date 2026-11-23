# Dataset and Experimental Protocol

## Dataset

The project uses BEIR SciFact and the official SciFact claim-release files. SciFact contains scientific claims, a biomedical/clinical document corpus, and relevance or entailment annotations for supported development data.

The project distinguishes source-provided split organization from BEIR query/qrel organization:

| Source | Split use |
|---|---|
| `claims_train.jsonl` | Official training claims |
| `claims_dev.jsonl` | Official development claims |
| BEIR `qrels train` | Verified training relevance judgments |
| BEIR `qrels test` | Verified development relevance judgments |
| `claims_test.jsonl` | Official held-out claim file |

## Query and document organization

- Queries are claim IDs with claim text.
- Corpus documents use corpus/document IDs.
- Retrieval output preserves document IDs, ranks, and retrieval scores.
- Reranking adds cross-encoder scores and remapped ranks.
- Evidence grounding adds sentence indices and exact sentence text.

## Study protocol

The development study uses 300 official development claims. BEIR `qrels test` is treated as development relevance judgment data because it maps to those claims.

Evaluation use cases include:

- Retrieval comparison
- Reranker candidate-pool comparison
- Adaptive stopping strategy evaluation
- Evidence-aware reranking ablations
- Controlled protocol freeze

Study relevance judgments are not used by adaptive stopping rules or evidence thresholds in the finalized protocol.

## Held-out protocol

The official held-out `claims_test.jsonl` file must not be opened until a protocol is frozen. It must not be used for:

- Threshold calibration
- Model selection
- Configuration tuning
- Evidence selector threshold selection
- Verdict threshold selection
- Selective example reporting

Predictions are generated before evaluation gold data are loaded. In this project snapshot, the held-out file contains claim text but no gold evidence, relevance judgments, or verdict labels.

## Data leakage prevention

The repository maintains the following controls:

- Study claim and qrels loading
- Explicit held-out split separation
- Frozen configuration hashes
- Timestamped protocol freeze records
- Raw prediction outputs separate from derived metrics
- Historical run preservation
- Exclusion of test-accessing dataset tests

## Frozen configurations

The controlled evaluation uses:

- BM25 baseline
- Dense baseline
- Hybrid RRF baseline with constant 60
- Hybrid cross-encoder reranking
- Dense adaptive score-gap strategy
- Hybrid evidence-grounded system

The configuration hash is recorded in `run_0010`.

## Metrics

### Retrieval

- Hit@K
- Precision@K
- Recall@K
- MRR@K
- MAP@K
- nDCG@K

### Verification

- Verdict accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision, recall, F1
- Confusion matrix
- Citation validity
- Citation traceability
- Evidence coverage
- Abstention rate
- Non-abstained accuracy
- Coverage-risk analysis

Verification and document retrieval metrics must be reported separately.

## Statistical testing

Paired comparisons use:

- Same-query per-system observations
- Two-sided Wilcoxon signed-rank tests where appropriate
- 10,000 paired bootstrap resamples
- 95% confidence intervals
- Holm-Bonferroni correction within each declared comparison family
- Paired effect sizes where defined

Adjusted p-values below 0.05 indicate statistical significance within that family. They do not establish practical importance or equivalence.

## Held-out limitation

The official held-out claims file cannot support:

- Retrieval accuracy
- Verdict accuracy
- Macro or per-class F1
- Evidence sentence coverage
- Gold-standard confusion matrices
- Statistical superiority claims

It can support pipeline-execution observations such as candidate counts, system latency, successful predictions, abstention behavior, and citation-validity checks against retrieved source text. These observations must not be presented as predictive accuracy.
