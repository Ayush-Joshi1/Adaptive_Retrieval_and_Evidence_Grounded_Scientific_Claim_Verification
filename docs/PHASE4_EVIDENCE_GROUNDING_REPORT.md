# Phase 4 - Evidence Grounding

## Research objective

Phase 4 implements document-level evidence grounding for scientific claim verification. It identifies relevant abstract sentences, maps their relationship to a claim, validates citations against the original corpus, and returns an explicit final verdict or abstention.

This implementation does not claim scientific entailment, sentence-level natural-language inference, or citation verification beyond exact source-text and metadata validation.

## Architecture

```text
Claim
  -> Existing Dense/Hybrid or Adaptive retrieval
  -> Optional cross-encoder reranking
  -> Optional evidence-aware reranking
  -> Claim-relevant sentence selection
  -> Baseline relationship mapping
  -> Exact citation validation
  -> Configurable verdict and abstention
  -> Structured result
```

Modules:

- `src/evidence/evidence_selector.py`
- `src/evidence/claim_evidence_mapper.py`
- `src/evidence/citation_validator.py`
- `src/evidence/verdict_classifier.py`
- `src/evidence/evidence_grounding_pipeline.py`
- `configs/phase4_evidence_grounding.yaml`

## Evidence selection

Sentences are split from the original abstract text while retaining exact text and character offsets. The existing normalized embedding API scores claim/sentence relevance. If embeddings are unavailable, a deterministic lexical Jaccard fallback is used.

Selection limits are configurable. It keeps the strongest per-document sentences, ranks globally, deduplicates exact normalized sentence text, and retains traceability to every duplicate source.

## Claim-to-evidence mapping

The baseline mapper does not use gold labels or official evidence annotations at inference time. It classifies:

- `SUPPORTS`: relevant overlap plus support terminology
- `REFUTES`: negation/antonym/conflict markers
- `NEUTRAL`: relevant but undecisive
- `INSUFFICIENT`: weak relevance or missing text

This is a transparent rule-based baseline. Relevance is not treated as entailment.

## Evidence-aware reranking integration

`EvidenceAwareReranker.rerank_with_evidence()` adds optional support and refutation evidence signals to its existing weighted reranking formula. The pipeline also preserves the existing `rerank()` API and existing cross-encoder interface.

Evidence-support weights are configurable in `configs/phase4_evidence_grounding.yaml`; their effects are not calibrated probabilities.

## Citation validation

Citations must satisfy all of:

- Document ID exists in the corpus.
- Sentence index is valid.
- Evidence text exactly matches normalized original sentence text.
- Source metadata is consistent.
- Duplicate document/sentence citations are handled once.

A citation-looking identifier is never sufficient proof.

## Verdict classification

The classifier considers support/refutation strength, source diversity, conflicts, relevance, citation validity, and a configurable confidence threshold.

It returns `NOT_ENOUGH_INFO` for unavailable or weak evidence, unresolved conflicts, low confidence, or invalid decisive citations. Confidence is an uncertainty/strength heuristic, not a calibrated probability.

## Configuration

The separate configuration file controls:

- Sentence limits and relevance threshold
- Mapping thresholds
- Evidence-aware reranking weights
- Abstention/confidence thresholds
- Conflict margin
- Citation rejection behavior
- Dense/Hybrid/adaptive integration options

## Testing

The new test module [tests/test_evidence_grounding.py](../tests/test_evidence_grounding.py) covers:

- Sentence extraction and exact text
- Empty and malformed abstracts
- Multiple documents and duplicate handling
- Supporting, refuting, neutral, and insufficient mappings
- Valid/invalid documents and sentence indices
- Modified text and duplicate citations
- Clear/refuting/conflicting/low-confidence verdicts
- Pipeline success, empty retrieval, and citation failure
- Evidence-aware reranker compatibility

Focused evidence-grounding test status: **18 passed**.
Combined evidence and existing Experiment A/B/C compatibility test status: **47 passed**.

The held-out dataset-validation test is excluded because its existing implementation loads every official claim split. No Phase 4 test reads `claims_test.jsonl`.

## Performance

Evidence selection and relationship mapping are CPU/local operations. The first embedding call reuses the existing project embedder; model loading is process-level. No cross-encoder or embedding model is downloaded by Phase 4 test execution.

Runtime and memory should be measured in a real development-size pipeline run when retrieval and reranking components are supplied.

## Limitations

- No validated entailment model is included.
- Relationship mapping is a lexical/rule-based baseline.
- Scientific terminology overlap is not evidence grounding.
- Confidence is not calibrated.
- Contradictory evidence is preserved, but conflict resolution is conservative.
- Sentence splitting is punctuation-based and may miss some scientific abbreviations.
- Gold labels in the development claim files are not used for inference.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_evidence_grounding.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_experiment_c.py tests/test_experiment_b.py tests/test_experiment_a.py tests/test_phase3_retrieval.py
```

Python API:

```python
from src.evidence import EvidenceGroundingPipeline

pipeline = EvidenceGroundingPipeline(
    corpus,
    retriever=dense_retriever,
    cross_encoder_reranker=cross_encoder,
    evidence_aware_reranker=evidence_aware,
)
result = pipeline.process("claim text", claim_id="claim-id", top_k=50)
```
