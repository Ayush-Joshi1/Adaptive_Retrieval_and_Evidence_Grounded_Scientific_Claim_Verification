# 4–5 Minute Project Demonstration Script

## Suggested screen sequence

1. GitHub repository overview
2. System architecture diagram
3. Retrieval comparison table
4. Reranking and adaptive-efficiency figures
5. Evidence-grounding workflow
6. Controlled-evaluation latency and limitation summary

## Script

### Opening — 30 seconds

"My project is Adaptive Retrieval and Evidence-Grounded Scientific Claim Verification. The central question is not simply whether a retrieval system finds an abstract, but whether a claim-verification system can trace a verdict to relevant source sentences and abstain when evidence is weak or conflicting.

The project progresses from sparse and dense retrieval through reranking and adaptive candidate selection, and finally through document-level evidence grounding."

### Problem and motivation — 45 seconds

"Traditional RAG retrieval returns documents that are semantically or lexically relevant to a query. Relevance does not necessarily establish support or contradiction for a scientific claim. A relevant document may contain contradictory findings, may discuss the claim only partially, or may contain no decidable evidence at all.

The system therefore separates retrieval relevance from relationship classification and requires exact source traceability before a decisive result is reported."

### Architecture — 45 seconds

"The pipeline begins with BM25, Dense, or Hybrid retrieval. Hybrid retrieval uses reciprocal-rank fusion to combine sparse and dense rankings. Candidate depth can be fixed or selected adaptively.

A cross-encoder reranks the candidate set. The evidence-grounding module then extracts claim-relevant sentences, applies a transparent rule-based relationship baseline, validates citations against original corpus text, and classifies the final verdict or abstention."

### Retrieval and reranking — 60 seconds

"On the official 300-query development set, BM25 achieved 0.7367 Hit@10 and 0.5585 nDCG@10. Dense retrieval improved to 0.7967 Hit@10 and 0.6402 nDCG@10, while Hybrid achieved 0.8167 and 0.6442.

Cross-encoder reranking produced numerical improvements for many development pools. However, the requested corrected multiple-comparison analyses did not support a broad statistically significant reranking claim.

Adaptive dense score-gap retrieval processed about 16.93 candidates rather than 50 against fixed Top-50 while retaining nearly the same development nDCG@10. This is an efficiency observation, not proof of equivalence."

### Evidence grounding — 45 seconds

"The evidence-grounding baseline is document-level rather than entailment-based. The evidence-aware development result reached 0.704115 nDCG@10 for Hybrid Pool-50. Its improvement over cross-encoder reranking was not statistically significant after correction.

The system does not claim semantic entailment or sentence-level scientific validation. It validates whether selected text exactly matches a corpus sentence and whether IDs and indices are consistent."

### Controlled evaluation and limitations — 45 seconds

"The final controlled run freeze occurred before any official held-out claims were opened. It processed all 300 held-out claim texts through six systems.

The evidence-grounded system had a 46.67% abstention rate and mean end-to-end latency of about 23.50 seconds per query. Evidence selection added about 12.51 seconds on average.

The held-out file contains claim text but no relevance judgments, evidence annotations, or verdict labels. As a result, held-out retrieval accuracy and claim-verification accuracy cannot be calculated. Citation validity means source-text traceability, not factual correctness."

### Conclusion — 30 seconds

"This project demonstrates a reproducible retrieval-to-evidence pipeline and a scientifically conservative evaluation protocol. Its strongest conclusions concern development retrieval behavior, candidate efficiency, citation validity, and abstention. The held-out test data is insufficient for predictive-accuracy claims.

The final deliverables make the implementation, configuration, figures, reports, and limitations accessible for academic review and portfolio demonstration."
