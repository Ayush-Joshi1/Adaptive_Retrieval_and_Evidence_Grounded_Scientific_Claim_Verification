import hashlib
import json

import pytest

from experiments.run_phase5_controlled import (
    _freeze_payload,
    _infer_claim_label,
    _is_protocol_frozen,
    _test_claim_texts_only,
)


def test_frozen_payload_is_reproducible_and_has_required_systems():
    payload = _freeze_payload()
    canonical = json.dumps(payload, sort_keys=True)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert len(payload["systems"]) == 6
    assert payload["protocol_freeze"]["heldout_test_opened"] is False


def test_protocol_freeze_requires_explicit_freeze_record(tmp_path):
    assert _is_protocol_frozen(tmp_path) is False
    (tmp_path / "protocol_freeze.json").write_text(json.dumps({
        "status": "frozen",
        "heldout_test_opened": False,
        "configuration_hash": "abc",
    }), encoding="utf-8")
    assert _is_protocol_frozen(tmp_path) is True


def test_test_claim_text_loader_excludes_evidence(tmp_path):
    path = tmp_path / "claims_test.jsonl"
    path.write_text(
        json.dumps({"id": 1, "claim": "Claim text", "evidence": {"123": [{"label": "SUPPORT"}]}}) + "\n",
        encoding="utf-8",
    )
    claims = _test_claim_texts_only(path)
    assert claims == [{"query_id": "1", "claim_text": "Claim text"}]
    assert "evidence" not in claims[0]


def test_claim_label_inference_does_not_infer_entailment():
    assert _infer_claim_label({"10": [{"label": "SUPPORT"}]}) == "SUPPORT"
    assert _infer_claim_label({"10": [{"label": "CONTRADICT"}]}) == "CONTRADICT"
    assert _infer_claim_label({}) == "NOT ENOUGH INFO"
    assert _infer_claim_label({"10": [{"label": "SUPPORT"}], "11": [{"label": "CONTRADICT"}]}) == "NOT ENOUGH INFO"
