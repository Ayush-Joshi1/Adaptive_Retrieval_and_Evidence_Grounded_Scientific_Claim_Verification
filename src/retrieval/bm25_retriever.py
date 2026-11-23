from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency fallback
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "with", "will", "within", "between", "this", "these", "those", "than", "then",
    "through", "under", "over", "after", "before", "because", "but", "into", "more",
    "most", "not", "such", "upon", "their", "there", "these", "which", "where",
}


class _BM25Fallback:
    """Small BM25 implementation used when rank-bm25 is unavailable."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus = [list(doc) for doc in corpus]
        self.doc_count = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(1, self.doc_count)
        self.idf_cache = self._compute_idf()

    def _compute_idf(self):
        df: Dict[str, float] = {}
        for doc in self.corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                df[token] = df.get(token, 0.0) + 1.0
        n = self.doc_count
        idf = {}
        for token, freq in df.items():
            idf[token] = float((n - freq + 0.5) / (freq + 0.5))
        return idf

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        if not query_tokens:
            return [0.0 for _ in self.corpus]
        scores = [0.0 for _ in self.corpus]
        for i, doc_tokens in enumerate(self.corpus):
            doc_len = len(doc_tokens)
            score = 0.0
            for token in set(query_tokens):
                if token not in self.idf_cache:
                    continue
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score += self.idf_cache[token] * numerator / denominator
            scores[i] = score
        return scores


def _tokenize(text: Any, lowercase: bool = True, stopword_handling: bool = False) -> List[str]:
    if text is None:
        return []
    source = str(text)
    if lowercase:
        source = source.lower()
    matches = re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", source)
    if stopword_handling:
        matches = [token for token in matches if token not in STOPWORDS]
    return matches


class BM25Retriever:
    """CPU-compatible BM25 retriever over the SciFact corpus."""

    def __init__(
        self,
        corpus_path: Optional[str | Path] = None,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
        stopword_handling: bool = False,
        tokenization_strategy: str = "simple",
        candidate_depth: int = 100,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.lowercase = lowercase
        self.stopword_handling = stopword_handling
        self.tokenization_strategy = tokenization_strategy
        self.candidate_depth = int(candidate_depth)
        self.corpus_path = Path(corpus_path) if corpus_path is not None else Path(__file__).resolve().parents[2] / "data" / "processed" / "corpus.jsonl"
        self.corpus = self._load_corpus()
        self.corpus_by_id = {str(doc.get("corpus_id")): doc for doc in self.corpus}
        self.id_map = [str(doc.get("corpus_id")) for doc in self.corpus]
        self._query_token_cache: Dict[str, List[str]] = {}
        self._index = self._build_index()

    def _load_corpus(self) -> List[Dict[str, Any]]:
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        docs: List[Dict[str, Any]] = []
        with self.corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                docs.append(doc)
        return docs

    def _build_index(self):
        tokenized_docs = []
        for doc in self.corpus:
            text = doc.get("text") or doc.get("title") or ""
            tokenized_docs.append(_tokenize(text, lowercase=self.lowercase, stopword_handling=self.stopword_handling))
        if BM25Okapi is not None:
            return BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        return _BM25Fallback(tokenized_docs, k1=self.k1, b=self.b)

    def _tokenize_query(self, query: str) -> List[str]:
        key = str(query)
        if key in self._query_token_cache:
            return self._query_token_cache[key]
        tokens = _tokenize(query, lowercase=self.lowercase, stopword_handling=self.stopword_handling)
        self._query_token_cache[key] = tokens
        return tokens

    def retrieve(self, query: str, top_k: Optional[int] = None, query_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.candidate_depth
        if top_k <= 0:
            return []
        if query is None or str(query).strip() == "":
            return []

        tokens = self._tokenize_query(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), self.id_map[item[0]]),
        )

        results: List[Dict[str, Any]] = []
        for rank, (index, score) in enumerate(ranked[:max(1, int(top_k))], start=1):
            doc_id = str(self.id_map[index])
            doc = self.corpus_by_id.get(doc_id)
            if doc is None:
                continue
            result = {
                "query_id": str(query_id) if query_id is not None else "",
                "document_id": doc_id,
                "corpus_id": doc_id,
                "rank": rank,
                "score": float(score),
                "retrieval_score": float(score),
                "similarity_score": float(score),
                "retriever_name": "bm25",
                "title": doc.get("title"),
                "text": doc.get("text"),
            }
            results.append(result)
        return results
