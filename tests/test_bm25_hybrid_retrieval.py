from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


class DummyDenseRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=None, query_id=None):
        items = []
        for rank, doc_id in enumerate(self._results[:top_k], start=1):
            items.append({"document_id": str(doc_id), "corpus_id": str(doc_id), "rank": rank, "score": 1.0 / rank})
        return items


def test_bm25_retriever_preserves_document_ids_and_order():
    retriever = BM25Retriever(corpus_path="data/processed/corpus.jsonl", candidate_depth=5)
    result = retriever.retrieve("cell biology", top_k=3, query_id="q1")
    assert result
    assert all("document_id" in row for row in result)
    assert all(row["corpus_id"] == row["document_id"] for row in result)
    assert [row["rank"] for row in result] == [1, 2, 3]
    assert all(row["retriever_name"] == "bm25" for row in result)


def test_hybrid_retriever_rrf_formula():
    bm25 = DummyDenseRetriever(["d1", "d2", "d3"])
    dense = DummyDenseRetriever(["d2", "d1", "d4"])
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_constant=60)
    result = hybrid.retrieve("query", top_k=4, query_id="q1")
    doc_ids = [row["document_id"] for row in result]
    assert doc_ids[0] == "d2" or doc_ids[0] == "d1"
    assert all(row["retriever_name"] == "hybrid" for row in result)
    assert all("rrf_score" in row for row in result)


def test_hybrid_retriever_handles_missing_document_in_one_source():
    bm25 = DummyDenseRetriever(["a", "b"])
    dense = DummyDenseRetriever(["b", "c"])
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_constant=60)
    result = hybrid.retrieve("query", top_k=10, query_id="q1")
    assert {row["document_id"] for row in result} == {"a", "b", "c"}
    assert all("bm25_rank" in row for row in result)
    assert all("dense_rank" in row for row in result)
