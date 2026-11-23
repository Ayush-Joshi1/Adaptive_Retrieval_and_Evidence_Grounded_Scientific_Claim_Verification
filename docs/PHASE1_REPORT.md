# Phase 1 Report

## Summary

Phase 1 focused on research foundation, evaluation integrity, and reproducibility. The project was inspected in its current state, historical outputs were preserved, split integrity was verified using the official SciFact claim files, and centralized utilities were added for dataset validation, retrieval metrics, classification metrics, run management, and FAISS/index integrity checks.

This phase did not run any Gemini generation experiments and did not modify the historical result files under `results/`.

## Files inspected

- `README.md`
- `requirements.txt`
- `src/data/prepare_dataset.py`
- `src/retrieval/embedder.py`
- `src/retrieval/retriever.py`
- `experiments/`
- `data/processed/`
- `data/index/`
- `results/`
- official raw SciFact files under `data/processed/raw_scifact/data/`

## Files created

- `src/data/dataset_validation.py`
- `src/data/split_manager.py`
- `src/retrieval/index_validation.py`
- `src/evaluation/retrieval_metrics.py`
- `src/evaluation/classification_metrics.py`
- `src/evaluation/statistical_tests.py`
- `src/evaluation/evaluator.py`
- `src/experiments/experiment_config.py`
- `src/experiments/run_manager.py`
- `src/experiments/artifact_manager.py`
- `configs/research_config.yaml`
- `manifests/dataset_manifest.json`
- `manifests/experiment_manifest.json`
- `docs/RESEARCH_PROTOCOL.md`
- `docs/PHASE1_REPORT.md`
- `tests/test_dataset_validation.py`
- `tests/test_retrieval_metrics.py`
- `tests/test_classification_metrics.py`
- `tests/test_index_validation.py`
- `tests/test_run_manager.py`

## Files modified

- `requirements.txt`
- `src/retrieval/retriever.py`
- `src/evaluation/__init__.py`

## Research integrity fixes

- Verified that the historical 50-query sample is not an official development split and must remain a historical exploratory benchmark.
- Added formal split validation that distinguishes official SciFact train/dev/test claim IDs from BEIR qrels.
- Added guardrails against using BEIR test qrels for tuning or final evaluation.
- Added reusable retrieval and classification metrics with explicit treatment of missing qrels and zero-result scenarios.
- Added run management helpers with unique IDs, deduplication, error categories, and retry counters.
- Added FAISS/index consistency validation without overwriting the legacy `.ivf` file.
- Kept the historical index and result files untouched while introducing a safe, accurate lookup path in the retrieval module.

## Tests executed and results

Command run:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_dataset_validation.py tests/test_retrieval_metrics.py tests/test_classification_metrics.py tests/test_index_validation.py tests/test_run_manager.py -q
```

Result:

- 11 passed in 34.22s

## Dataset split findings

Verified using the official SciFact split files and BEIR qrels:

- official `claims_train.jsonl`: 809 query IDs
- official `claims_dev.jsonl`: 300 query IDs
- official `claims_test.jsonl`: 300 query IDs
- BEIR qrels `train`: 809 query IDs
- BEIR qrels `test`: 300 query IDs

Alignment result:

- BEIR `train` qrels match official SciFact train claims
- BEIR `test` qrels match official SciFact dev claims
- official SciFact test claims do not overlap with BEIR queries or qrels

This confirms that the historical project workflow used the BEIR test split as a dev set. This is a valid limitation and not a final evaluation split.

## Remaining issues

- The project still lacks a final, locked-run evaluation manifest for the official test split.
- Historical result files under `results/` remain descriptive and should not be silently replaced.
- No generation experiments were run in Phase 1, so no new leaderboard claims are made.
- The repository still needs a more formal run registry if future generation experiments are added.

## Commands to reproduce Phase 1

```bash
cd "C:\Users\admin\Desktop\rag project upd"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/test_dataset_validation.py tests/test_retrieval_metrics.py tests/test_classification_metrics.py tests/test_index_validation.py tests/test_run_manager.py -q
```

## Completion statement

Phase 1 is complete for the research foundation and reproducibility objectives as defined in the task. The main integrity issue was identified, the split mapping was verified, the project was hardened against test-set leakage in the framework layer, and the relevant deterministic tests pass without running Gemini or altering historical result files.
