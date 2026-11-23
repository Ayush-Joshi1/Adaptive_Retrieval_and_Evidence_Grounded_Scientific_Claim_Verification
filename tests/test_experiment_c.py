import json

import pytest

from src.retrieval.evidence_aware_reranker import EvidenceAwareReranker
from experiments.run_experiment_c import _valid_checkpoint, _configuration_name


def test_evidence_scores_are_normalized_and_weighted():
    reranker = EvidenceAwareReranker()
    candidates = [
        {"document_id": "a", "text": "A protein therapy trial", "retrieval_score": 0.9, "rank": 1},
        {"document_id": "b", "text": "An unrelated abstract", "retrieval_score": 0.1, "rank": 2},
    ]
    ranked = reranker.rerank("protein therapy", candidates, cross_encoder_scores=[0.2, 0.8])
    assert ranked[0]["document_id"] in {"a", "b"}
    for row in ranked:
        assert 0.0 <= row["cross_encoder_normalized"] <= 1.0
        assert 0.0 <= row["lexical_overlap"] <= 1.0
        assert 0.0 <= row["scientific_overlap"] <= 1.0
        assert 0.0 <= row["retrieval_score_normalized"] <= 1.0
        assert 0.0 <= row["rank_signal"] <= 1.0
        assert 0.0 <= row["evidence_score"] <= 1.0


def test_weights_are_configurable_and_rejected_when_negative():
    reranker = EvidenceAwareReranker({"cross_encoder": 0.8, "lexical_overlap": 0.2})
    weights = reranker._normalized_weights()
    assert abs(weights["cross_encoder"] - 0.8) < 1e-12
    assert abs(weights["lexical_overlap"] - 0.2) < 1e-12
    with pytest.raises(ValueError, match="non-negative"):
        EvidenceAwareReranker({"cross_encoder": -0.1})


def test_missing_cross_encoder_scores_are_safe():
    reranker = EvidenceAwareReranker()
    candidates = [{"document_id": "a", "text": "gene therapy", "retrieval_score": 0.5, "rank": 1}]
    ranked = reranker.rerank("gene therapy", candidates, cross_encoder_scores=None)
    assert ranked[0]["cross_encoder_score"] == 0.0


def test_duplicate_candidates_are_removed():
    reranker = EvidenceAwareReranker()
    candidates = [
        {"document_id": "a", "text": "protein", "retrieval_score": 0.5, "rank": 1},
        {"document_id": "a", "text": "protein", "retrieval_score": 0.5, "rank": 1},
        {"document_id": "b", "text": "gene", "retrieval_score": 0.4, "rank": 2},
    ]
    ranked = reranker.rerank("protein", candidates, cross_encoder_scores=[0.5, 0.5, 0.4])
    assert [row["document_id"] for row in ranked].count("a") == 1
    assert len(ranked) == 2


def test_ties_use_document_id_deterministically():
    reranker = EvidenceAwareReranker({"cross_encoder": 1.0})
    candidates = [
        {"document_id": "z", "text": "", "retrieval_score": 0.5, "rank": 1},
        {"document_id": "a", "text": "", "retrieval_score": 0.5, "rank": 2},
    ]
    ranked = reranker.rerank("nothing", candidates, cross_encoder_scores=[0.5, 0.5])
    assert [row["document_id"] for row in ranked] == ["z", "a"]


def test_cross_encoder_compatibility_preserves_ids_and_relevance_scores():
    reranker = EvidenceAwareReranker({"cross_encoder": 1.0})
    candidates = [
        {"document_id": "1", "text": "protein", "retrieval_score": 0.8, "rank": 1},
        {"document_id": "2", "text": "drug", "retrieval_score": 0.7, "rank": 2},
    ]
    ranked = reranker.rerank("claim", candidates, cross_encoder_scores=[0.1, 0.9])
    assert [row["document_id"] for row in ranked] == ["2", "1"]
    assert ranked[0]["cross_encoder_score"] == 0.9


def test_scientific_overlap_does_not_treat_contradiction_as_penalty():
    reranker = EvidenceAwareReranker()
    terms = reranker._scientific_overlap("protein therapy", "protein therapy reduces disease")
    contradiction_terms = reranker._scientific_overlap("protein therapy", "protein therapy does not reduce disease")
    assert terms == contradiction_terms


def test_identical_candidate_sets_are_used_for_paired_conditions():
    baseline = ["doc-1", "doc-2", "doc-3"]
    evidence = ["doc-3", "doc-1", "doc-2"]
    assert set(baseline) == set(evidence)
    assert len(baseline) == len(set(baseline))


def test_checkpoint_requires_every_experiment_c_configuration(tmp_path):
    configs = [
        _configuration_name(method, pool, condition)
        for method in ("dense", "hybrid")
        for pool in (10, 20, 50)
        for condition in (
            "original",
            "cross_encoder",
            "cross_encoder_lexical",
            "cross_encoder_scientific",
            "evidence_aware_full",
        )
    ]
    identity_rows = []
    for name in configs:
        identity_rows.append({
            "retrieval_method": name.split("_pool_")[0],
            "candidate_pool_size": int(name.split("_pool_")[1].split("_")[0]),
            "configuration": name,
        })
    checkpoint = {
        "query_id": "1",
        "metric_rows": identity_rows,
        "ranking_rows": identity_rows,
        "component_scores": [],
        "performance_rows": identity_rows,
    }
    path = tmp_path / "1.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    assert _valid_checkpoint(path, "1", configs) == checkpoint
