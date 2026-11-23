import json

from experiments.run_experiment_a import _hydrate_candidates, _ranked_subset
from experiments.run_experiment_a import CANDIDATE_POOLS, _valid_query_checkpoint


def test_hybrid_candidates_are_hydrated_from_shared_corpus():
    candidates = [{"document_id": "doc-1", "rank": 1, "score": 0.5}]
    corpus = {"doc-1": {"corpus_id": "doc-1", "title": "Title", "text": "Abstract"}}

    hydrated = _hydrate_candidates(candidates, corpus)

    assert hydrated[0]["document_id"] == "doc-1"
    assert hydrated[0]["title"] == "Title"
    assert hydrated[0]["text"] == "Abstract"
    assert "text" not in candidates[0]


def test_pool_reranking_excludes_candidates_outside_original_prefix():
    scored = [
        {"document_id": "doc-1", "original_rank": 1, "cross_encoder_score": 0.2},
        {"document_id": "doc-2", "original_rank": 2, "cross_encoder_score": 0.9},
        {"document_id": "doc-11", "original_rank": 11, "cross_encoder_score": 1.0},
    ]

    reranked = _ranked_subset(scored, 10)

    assert [row["document_id"] for row in reranked] == ["doc-2", "doc-1"]
    assert [row["rank"] for row in reranked] == [1, 2]


def test_query_checkpoint_requires_all_configurations_and_matching_candidate_ids(tmp_path):
    rows = []
    rankings = []
    performance = []
    for method in ("dense", "hybrid"):
        for pool_size in CANDIDATE_POOLS:
            for condition in ("baseline", "reranked"):
                identity = {
                    "query_id": "q1",
                    "base_method": method,
                    "candidate_pool_size": pool_size,
                    "condition": condition,
                }
                rows.append(identity)
                performance.append(identity)
                rankings.append({
                    **identity,
                    "rankings": [
                        {"document_id": "doc-1"},
                        {"document_id": "doc-2"},
                    ],
                })
    path = tmp_path / "q1.json"
    checkpoint = {"query_id": "q1", "metrics": rows, "rankings": rankings, "performance": performance}
    path.write_text(json.dumps(checkpoint), encoding="utf-8")

    assert _valid_query_checkpoint(path, "q1") == checkpoint

    checkpoint["metrics"].pop()
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    try:
        _valid_query_checkpoint(path, "q1")
    except ValueError as exc:
        assert "Incomplete metrics checkpoint" in str(exc)
    else:
        raise AssertionError("Incomplete checkpoints must not be resumed")
