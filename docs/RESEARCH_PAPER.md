# Adaptive Retrieval and Evidence-Grounded Scientific Claim Verification

## Abstract

This project examines CPU-compatible retrieval and reranking methods for scientific claim verification using BEIR SciFact. It compares sparse BM25 retrieval, normalized dense embedding retrieval, reciprocal-rank-fusion hybrid retrieval, cross-encoder reranking, adaptive candidate-depth retrieval, and document-level evidence-aware reranking. It also implements a modular evidence-grounding pipeline that extracts claim-relevant sentences, maps them to support/refutation/neutral/insufficient relationship classes, validates citations against source corpus text, and abstains under weak or conflicting evidence.

On the official 300-query development split, Hybrid retrieval achieved the strongest Phase 2 retrieval result at K=10, while BM25 remained a meaningful sparse baseline and Dense retrieval was competitive. Cross-encoder reranking produced numerical improvements in several candidate-pool configurations, but corrected statistical analyses did not establish broad superiority over the corresponding baselines. Adaptive Dense retrieval reduced candidate processing and latency with nearly unchanged nDCG@10 in the evaluated comparison. The highest observed Evidence-Aware Experiment C development nDCG@10 was 0.704115 for Hybrid Pool-50; this numerical improvement was not significant after correction in the evidence-versus-cross-encoder nDCG@10 comparison.

A frozen Phase 5 protocol processed 300 official held-out claim texts. The evidence-grounded system abstained for 46.67% of queries and recorded 23.50 seconds mean end-to-end latency. The held-out claims file contains no relevance judgments, evidence annotations, or verdict labels, so held-out retrieval, verdict accuracy, evidence coverage, and statistical superiority cannot be reported. The run measures pipeline execution and observable behavior rather than predictive accuracy.

**Keywords:** scientific fact verification, SciFact, BM25, dense retrieval, hybrid retrieval, reciprocal rank fusion, cross-encoder reranking, adaptive retrieval, evidence grounding, abstention.

## 1. Introduction

Scientific claim verification is difficult because relevant documents are not automatically decisive evidence. Retrieval systems rank documents by lexical or semantic similarity, but a claim-verifier must also distinguish supporting, contradicting, neutral, and insufficient information. Document-level ranking therefore cannot replace evidence grounding.

This project is organized into six phases: split integrity and infrastructure; retrieval comparison; reranking and adaptive retrieval; evidence-grounded engineering; frozen held-out execution; and research/portfolio documentation.

## 2. Research Problem and Motivation

The central problem is:

> Given a scientific claim and candidate scientific documents, can the system identify exact evidence, classify its relationship to the claim, validate its source, and abstain when evidence is inadequate?

The project does not assume that a retriever, reranker, or evidence extractor performs entailment. It separates:

- Document retrieval relevance
- Sentence-level claim relevance
- Support/refutation relationship classification
- Citation validity
- Final claim-verdict correctness

## 3. Research Objectives and Contributions

The objectives are to:

1. Compare BM25, Dense, and Hybrid retrieval on a verified 300-query development set.
2. Reuse candidate rankings and cross-encoder scores efficiently.
3. Evaluate adaptive candidate-depth selection.
4. Implement evidence selection and source-traceable citation validation.
5. Provide configurable abstention rather than forced verdicts.
6. Freeze and execute a final CPU-only controlled protocol.
7. Report negative, inconclusive, and unavailable results honestly.

Contributions include the modular evidence package, exact corpus citation validation, adaptive and evidence-aware interfaces, frozen run metadata, and a Phase 5 report that distinguishes executable pipeline behavior from unavailable gold-standard evaluation.

## 4. Related Work

SciFact is a scientific fact verification benchmark where claims are paired with corpus abstracts and evidence-level labels. BEIR provides retrieval formulations and qrels over the underlying scientific corpus. BM25 represents sparse lexical retrieval, dense sentence embeddings support semantic nearest-neighbor retrieval, reciprocal rank fusion combines independent rankings, and cross-encoders rank claim-document pairs jointly.

The repository does not fabricate external bibliographic claims. A full publication submission should add peer-reviewed references and their exact editions, but the implementation claims in this paper are grounded in the project reports and source code.

## 5. Dataset Description

BEIR SciFact contains 809 official training claims, 300 official development claims, and 300 official held-out claims. The BEIR `qrels test` split aligns with the official development claims, while `claims_test.jsonl` is outside the BEIR query/qrel mapping.

The document corpus is a JSONL collection with document IDs, titles, and abstracts. Development retrieval uses BM25, a 384-dimensional MiniLM embedding model, and FAISS `IndexFlatIP`.

## 6. Methodology

### 6.1 BM25

BM25 uses tokenized abstract text with k1=1.5 and b=0.75. It serves as a sparse lexical baseline and may use the rank-BM25 implementation or the project fallback.

### 6.2 Dense retrieval

Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2` and L2-normalized embeddings. FAISS `IndexFlatIP` computes inner-product similarity, equivalent to cosine similarity for normalized vectors.

### 6.3 Hybrid retrieval

Hybrid retrieval combines BM25 and Dense rankings using reciprocal rank fusion:

\[
RRF(d)=\sum_{r \in R} \frac{1}{c+r(d)}
\]

The frozen development and Phase 5 parameter is c=60, with component depth 100 for the Phase 5 Hybrid systems.

### 6.4 Cross-encoder reranking

The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`, evaluated on CPU with batch size 16. Candidate pools of 10, 20, and 50 are scored and ranked.

### 6.5 Adaptive retrieval

Adaptive Dense retrieval begins at 10 candidates. A score-gap strategy stops at 10 when the normalized leading-score gap reaches 0.10; otherwise it may expand to 20 or 50. Confidence and score-gap modes are retained as development comparisons.

### 6.6 Evidence grounding

Evidence selection:

1. Split original abstract text into indexed sentences.
2. Encode the claim and sentences using the existing embedding path.
3. Rank normalized claim-sentence relevance.
4. Limit sentences per document and globally.
5. Deduplicate text while preserving source traceability.

Claim-evidence mapping is a transparent rule-based baseline using overlap, support terminology, contradiction markers, and antonym markers. It is not validated semantic entailment.

Citation validation checks:

- Document existence
- Sentence-index validity
- Exact normalized sentence-text match
- Source metadata consistency
- Duplicate citation handling

Verdict classification returns SUPPORTS, REFUTES, or NOT_ENOUGH_INFO. Abstention applies for weak evidence, unresolved conflict, low confidence, or invalid decisive citations.

Confidence is a configurable strength heuristic rather than a calibrated probability.

## 7. Experimental Setup

Phase 2 uses the official 300-query development claims and BEIR `qrels test`, mapped to development claims. Phase 3 uses the same development split for fixed/adaptive and evidence-aware experiments. Phase 5 freezes six systems and processes 300 official held-out claim texts.

The Phase 5 freeze timestamp is 2026-11-23T12:58:22.714796+00:00 and the frozen configuration hash is `05c8c0bf3fddebcb3b43e739760d14770a10eb0c021fd07c6b323d1387511c3e`. The held-out claims file was not opened until the freeze record existed.

## 8. Results

### 8.1 Phase 2 Retrieval

| Approach | Hit@10 | Recall@10 | MRR@10 | MAP@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.7367 | 0.7229 | 0.5169 | 0.5015 | 0.5585 |
| Dense | 0.7967 | 0.7867 | 0.5979 | 0.5887 | 0.6402 |
| Hybrid | 0.8167 | 0.8011 | 0.5988 | 0.5893 | 0.6442 |

RRF sensitivity achieved Hit@10=0.840 at c=10.

### 8.2 Phase 3 Reranking and Adaptive Retrieval

The cross-encoder reranking experiment produced numerical pool-size improvements, but the reported corrected analyses did not establish a broad statistically significant reranking finding.

Dense score-gap adaptive retrieval used 16.93 candidates/query and achieved 0.690694 nDCG@10, nearly unchanged from fixed Top-50 0.690648 in the development comparison. Dense confidence adaptive retrieval used 10 candidates/query and retained 0.683095 nDCG@10.

### 8.3 Evidence-Aware Development Results

The highest observed Experiment C development nDCG@10 was Hybrid Pool-50 full evidence-aware reranking at 0.704115, compared with 0.693866 for the corresponding cross-encoder-only result.

This 0.010249 numerical difference was not significant after the 150-test Holm-Bonferroni development family. It is therefore an observed difference, not a corrected superiority result.

### 8.4 Phase 5 Execution Results

Phase 5 processed 300 held-out claim texts under six frozen systems.

- Evidence-grounded abstention rate: 46.67%
- Evidence-grounded mean end-to-end latency: approximately 23.50 seconds/query
- Evidence selection latency: approximately 12.51 seconds/query
- Citation validity and traceability: 100% among source-matched mappings
- Cross-encoder mean reranking latency: 11.25 seconds/query
- Dense adaptive mean reranking latency: 3.68 seconds/query

These are execution observations. Citation validity means exact source-text/mapping validation, not factual correctness or entailment validity.

## 9. Statistical Analysis

Where development metrics support paired inference, reported analyses use paired bootstrap confidence intervals, Wilcoxon signed-rank tests, effect sizes, and Holm-Bonferroni correction. Non-significance is not evidence of equivalence.

Phase 5 has no valid gold-standard paired inference because the held-out claim file lacks gold relevance judgments and verdict/evidence labels.

## 10. Discussion

Dense and Hybrid retrieval are more effective than BM25 on the Phase 2 development metrics. Hybrid is strongest at K=10, but the dense-hybrid gap is not the basis for a strong corrected superiority claim.

Reranking is useful at shorter and longer pools in selected numerical comparisons, but corrected significance is essential. Adaptive retrieval is a promising efficiency method for Dense Pool-50, with candidate reduction and latency reduction accompanied by nearly unchanged nDCG.

Evidence-aware reranking adds an interpretable evidence signal but remains subject to weight, lexical, and corpus assumptions. Phase 4 is not a validated entailment system.

## 11. Limitations

- Development-set results may not generalize.
- Confidence and evidence thresholds are preconfigured, not calibrated probabilities.
- Document-level evidence matching is not sentence-level entailment.
- Citation validation is not citation or factual verification.
- CPU timing varies with load and runtime state.
- Hybrid Experiment B and C configurations must be interpreted consistently with component-depth differences.
- The official held-out file contains no gold labels.

## 12. Threats to Validity

Threats include split confusion, candidate-set differences, repeated component retrieval, rule-based mapping bias, punctuation-based sentence splitting, missing gold annotations, and CPU timing variability. The protocol and report make these limitations explicit.

## 13. Future Work

Appropriate future work includes externally validated entailment models, calibrated uncertainty, formal citation verification, sentence-level evidence datasets, non-inferiority margins, repeated-run latency confidence intervals, and a truly labeled held-out benchmark.

## 14. Conclusion

The project delivers a complete reproducible retrieval, reranking, adaptive, evidence-grounding, and controlled-execution framework. Measured development results support dense and hybrid retrieval gains over BM25. Adaptive retrieval can reduce candidate processing while preserving observed nDCG in development. Evidence-aware reranking and evidence grounding are promising observed systems, but neither a document lexical baseline nor held-out unlabeled execution establishes validated entailment or predictive superiority.

## 15. References

Repository-linked source material:

- BEIR SciFact data and qrels used by the project.
- `docs/PHASE2_REPORT.md`
- `docs/PHASE3_REPORT.md`
- `docs/PHASE3_EXPERIMENT_B_REPORT.md`
- `docs/PHASE3_EXPERIMENT_C_REPORT.md`
- `docs/PHASE4_EVIDENCE_GROUNDING_REPORT.md`
- `docs/PHASE5_CONTROLLED_EXPERIMENTS_REPORT.md`
- `configs/phase5_frozen_config.yaml`
