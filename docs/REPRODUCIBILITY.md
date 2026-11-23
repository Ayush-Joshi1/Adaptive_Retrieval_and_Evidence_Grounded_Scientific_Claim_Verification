# Reproducibility Guide

## Scope

This guide reproduces development experiments, evidence-grounding tests, and final artifact inspection without rerunning frozen Phase 1–5 experiments.

## Environment

- OS: Windows 10/11
- Python: 3.13 recommended
- CPU execution
- Virtual environment: `.venv`
- Dependency versions: `requirements.txt`

## Installation

```powershell
cd "C:\Users\admin\Desktop\rag project upd"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No GPU is required. Do not add API credentials to source code or commits.

## Data layout

```text
data/
  processed/
    corpus.jsonl
    raw_scifact/
      data/
        claims_train.jsonl
        claims_dev.jsonl
        claims_test.jsonl
  index/
    faiss_index_flat_ip.index
    id_map.json
manifests/
  dataset_manifest.json
configs/
  research_config.yaml
  phase4_evidence_grounding.yaml
  phase5_frozen_config.yaml
```

The official `claims_test.jsonl` must remain excluded from development tuning and test-label-only evaluation.

## Important split mapping

- Official train claims: `claims_train.jsonl`
- Official development claims: `claims_dev.jsonl`
- BEIR `qrels test`: aligned to official development claims
- Official held-out claims: `claims_test.jsonl`

The held-out claims file used in this repository contains claim text only and does not provide relevance judgments, evidence annotations, or verdict labels.

## Dataset preparation

```powershell
.\.venv\Scripts\python.exe src\data\prepare_dataset.py
```

This workflow is data-dependent and may use remote datasets. Existing prepared corpora and indexes are sufficient for the recorded development runs.

## Core tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Held-out claims must not be loaded by broad dataset tests. The following test module intentionally reads every official split and is excluded from final-test automation:

```text
tests/test_dataset_validation.py
```

Relevant controlled tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_phase5_controlled.py tests/test_evidence_grounding.py
```

## Phase 2 canonical reproduction

```powershell
.\.venv\Scripts\python.exe experiments\run_retrieval_comparison.py
.\.venv\Scripts\python.exe scripts\generate_phase2_figures.py
```

Do not overwrite existing runs. The run manager allocates a unique run directory.

## Phase 3 runners

Experiment A:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment_a.py
.\.venv\Scripts\python.exe experiments\run_experiment_a.py --resume run_0006
```

Experiment B:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment_b.py
.\.venv\Scripts\python.exe experiments\run_experiment_b.py --resume run_0008
```

Experiment C:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment_c.py
.\.venv\Scripts\python.exe experiments\run_experiment_c.py --resume run_0009
```

These are not necessary for Phase 6 portfolio generation. Preserve `run_0006`, `run_0008`, and `run_0009`.

## Phase 4 API

```python
from src.evidence import EvidenceGroundingPipeline

pipeline = EvidenceGroundingPipeline(
    corpus,
    retriever=dense_retriever,
    cross_encoder_reranker=cross_encoder,
    evidence_aware_reranker=evidence_aware,
)

result = pipeline.process(
    "A scientific claim",
    claim_id="claim-id",
    top_k=50,
)
```

`EvidenceGroundingPipeline.process()` returns the final verdict, confidence heuristic, evidence lists, citation validation, retrieval/reranking metadata, and latency fields.

## Phase 5 controlled runner

Freeze the protocol:

```powershell
.\.venv\Scripts\python.exe experiments\run_phase5_controlled.py --freeze-only
```

Run or resume the frozen evaluation:

```powershell
.\.venv\Scripts\python.exe experiments\run_phase5_controlled.py
.\.venv\Scripts\python.exe experiments\run_phase5_controlled.py --resume run_0010
```

The freeze record must exist before official held-out claim text is read. Predictions are preserved before gold loading.

## Phase 6 figures

```powershell
.\.venv\Scripts\python.exe scripts\generate_phase6_figures.py
```

Figures are generated only from saved development and run artifacts.

## Output structure

```text
results/runs/run_NNNN/
  config.json
  frozen_config.json
  protocol_freeze.json
  metadata.json
  environment_metadata.json
  aggregate_metrics.json
  per_query_metrics.json
  per_query_rankings.json
  performance_metrics.json
  statistical_tests.json
  validation.json
  figures/
  checkpoints/
```

Some raw checkpoint and per-query files are intentionally ignored by Git.

## Interpreting metrics

- Hit@K: at least one relevant document in the top K.
- Recall@K: retrieved relevant documents divided by query-relevant documents.
- MRR@K: reciprocal rank of the first relevant document.
- MAP@K: mean average precision over the query prefix.
- nDCG@K: graded relevance gain normalized by ideal discounted gain.
- Paired Cohen's dz: continuous paired effect size; use with corrected inference.
- Holm-adjusted p: corrected p-value across a declared comparison family.

## CPU-only considerations

- Cross-encoder reranking is CPU-bound and may take several minutes over 300 queries.
- Embedding and cross-encoder models should be loaded once per process.
- Evidence sentence embedding is bounded by candidate pool and sentence limits.
- Latencies vary with CPU load, cache state, and model/runtime versions.
- The held-out execution is not a gold-standard accuracy evaluation.
