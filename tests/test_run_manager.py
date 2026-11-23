from pathlib import Path

from src.experiments.run_manager import deduplicate_records, make_experiment_key, status_record, unique_run_dir


def test_duplicate_experiment_records_are_deduplicated(tmp_path):
    records = [
        {"experiment_key": "exp-1", "status": "success"},
        {"experiment_key": "exp-1", "status": "failure"},
        {"experiment_key": "exp-2", "status": "skipped"},
    ]
    deduped = deduplicate_records(records)
    assert len(deduped) == 2
    assert {row["experiment_key"] for row in deduped} == {"exp-1", "exp-2"}


def test_failed_runs_record_retry_count_and_error_category(tmp_path):
    record = status_record("exp-3", "failed", error_category="api_timeout", retry_count=1)
    assert record["status"] == "failed"
    assert record["error_category"] == "api_timeout"
    assert record["retry_count"] == 1


def test_unique_run_dirs_are_created(tmp_path):
    run_dir_a = unique_run_dir(tmp_path / "runs", prefix="phase1")
    run_dir_b = unique_run_dir(tmp_path / "runs", prefix="phase1")
    assert run_dir_a != run_dir_b
    assert run_dir_a.exists()
    assert run_dir_b.exists()
