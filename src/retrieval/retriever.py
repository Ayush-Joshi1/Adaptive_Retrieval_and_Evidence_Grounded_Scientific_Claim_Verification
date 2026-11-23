"""Retriever: builds FAISS IndexFlatIP over corpus embeddings and performs queries.

Behavior:
- Loads corpus from `data/processed/corpus.jsonl`.
- Builds embeddings via the embedder and L2-normalizes them.
- Uses FAISS IndexFlatIP so inner product on normalized vectors equals cosine similarity.
- Saves index and id mapping to `data/index/` to avoid recomputation.
"""
import json
import os
from typing import List, Dict, Tuple

import faiss
import numpy as np

from .embedder import encode_texts


DATA_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "index")
os.makedirs(DATA_INDEX_DIR, exist_ok=True)
LEGACY_INDEX_PATH = os.path.join(DATA_INDEX_DIR, "faiss_index.ivf")
INDEX_PATH = os.path.join(DATA_INDEX_DIR, "faiss_index_flat_ip.index")
IDMAP_PATH = os.path.join(DATA_INDEX_DIR, "id_map.json")
CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "processed", "corpus.jsonl")


class Retriever:
    def __init__(self, rebuild: bool = False):
        self.corpus: List[Dict] = []
        self.corpus_by_id: Dict[str, Dict] = {}
        self.id_map: List[str] = []  # position -> corpus_id
        self.index = None
        self.dim = None
        index_candidates = [INDEX_PATH, LEGACY_INDEX_PATH]
        selected_index = None
        for candidate in index_candidates:
            if os.path.exists(candidate) and os.path.exists(IDMAP_PATH):
                selected_index = candidate
                break
        if not rebuild and selected_index is not None:
            try:
                self.index = faiss.read_index(selected_index)
                with open(IDMAP_PATH, "r", encoding="utf-8") as fh:
                    self.id_map = json.load(fh)
                self.corpus = self._load_corpus()
                self.corpus_by_id = {str(doc.get("corpus_id")): doc for doc in self.corpus}
                self.dim = self.index.d
            except Exception:
                self._build_index()
        else:
            self._build_index()

    def _load_corpus(self) -> List[Dict]:
        docs = []
        if not os.path.exists(CORPUS_PATH):
            raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")
        with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                docs.append(json.loads(line))
        return docs

    def _build_index(self):
        # Load corpus and compute embeddings
        self.corpus = self._load_corpus()
        self.corpus_by_id = {str(doc.get("corpus_id")): doc for doc in self.corpus}
        texts = [d.get("text", "") for d in self.corpus]
        corpus_ids = [str(d.get("corpus_id")) for d in self.corpus]

        # Compute embeddings (normalized inside encode_texts)
        embeddings = encode_texts(texts)
        n, dim = embeddings.shape
        self.dim = dim

        # Build FAISS IndexFlatIP for inner-product search on normalized vectors.
        # Keep the historical legacy artifact untouched while writing the accurate flat-index name.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, INDEX_PATH)
        with open(IDMAP_PATH, "w", encoding="utf-8") as fh:
            json.dump(corpus_ids, fh)

        self.index = index
        self.id_map = corpus_ids

    def retrieve(self, query: str, top_k: int = 5, query_id: str | None = None) -> List[Dict]:
        """Retrieve top_k docs for the query.

        Returns list of dicts with keys: corpus_id, title, text, similarity_score
        as well as standardized query/document metadata for retrieval comparison.
        """
        if top_k <= 0:
            return []
        # Encode and normalize query using same embedder
        q_emb = encode_texts([query])  # shape (1, dim)
        if self.index is None:
            raise RuntimeError("Index not initialized")
        D, I = self.index.search(q_emb, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        results = []
        for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
            if idx < 0 or idx >= len(self.id_map):
                continue
            cid = str(self.id_map[idx])
            doc = self.corpus_by_id.get(cid)
            if doc is None:
                continue
            result = {
                "query_id": str(query_id) if query_id is not None else "",
                "document_id": cid,
                "corpus_id": cid,
                "rank": rank,
                "score": float(score),
                "retrieval_score": float(score),
                "similarity_score": float(score),
                "retriever_name": "dense",
                "title": doc.get("title"),
                "text": doc.get("text"),
            }
            results.append(result)
        return results
