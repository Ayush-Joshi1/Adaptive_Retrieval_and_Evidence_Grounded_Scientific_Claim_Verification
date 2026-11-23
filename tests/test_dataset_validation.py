from pathlib import Path

from src.data.dataset_validation import validate_split_alignment, validate_split_integrity


def test_official_splits_align_with_beir_qrels():
    summary = validate_split_integrity()
    assert summary["train_qrel_count"] == summary["train_claim_count"]
    assert summary["test_qrel_count"] == summary["dev_claim_count"]
    assert summary["beir_qrels_train_matches_train_claims"] is True
    assert summary["beir_qrels_test_matches_dev_claims"] is True
    assert validate_split_alignment() is True


def test_official_test_set_is_not_in_beir_queries():
    summary = validate_split_integrity()
    assert summary["official_test_ids_missing_from_beir_queries"]
    assert summary["official_test_ids_in_beir_queries"] == 0
