from src.retrieval.index_validation import validate_faiss_index


def test_existing_faiss_index_and_id_map_are_consistent():
    report = validate_faiss_index(
        "data/index/faiss_index.ivf",
        "data/index/id_map.json",
        "data/processed/corpus.jsonl",
        expected_dim=384,
        expected_corpus_count=5183,
    )
    assert report["is_valid"] is True
    assert report["metadata"]["index_dimension"] == 384
    assert report["metadata"]["vector_count"] == 5183
    assert report["metadata"]["corpus_count"] == 5183
