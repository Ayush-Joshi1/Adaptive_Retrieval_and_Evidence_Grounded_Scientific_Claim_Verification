"""Simple embedder using sentence-transformers/all-MiniLM-L6-v2.

Provides `encode_texts(texts: List[str]) -> np.ndarray` which returns L2-normalized
embeddings so cosine similarity can be computed as inner product in FAISS IndexFlatIP.

Normalization rationale:
- FAISS IndexFlatIP performs inner-product search. If vectors are L2-normalized,
  inner product equals cosine similarity. Normalizing ensures scores are in [-1,1]
  and comparable across corpus and queries.
"""
import hashlib
import re
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None


def _fallback_encode_texts(texts: List[str]) -> np.ndarray:
    """Create deterministic pseudo-embeddings without external model dependencies.

    The fallback intentionally preserves the public API while allowing retrieval code
    to run in lightweight test environments without downloading the Hugging Face
    model.
    """
    dim = 384
    vectors = []
    for text in texts:
        if text is None:
            text = ""
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        vector = np.zeros(dim, dtype="float32")
        if not tokens:
            vectors.append(vector)
            continue
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % dim
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        vectors.append(vector)
    return np.vstack(vectors) if vectors else np.zeros((0, dim), dtype="float32")


# Load model once at import time to reuse across calls.
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") if SentenceTransformer is not None else None


def encode_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into L2-normalized numpy embeddings.

    Args:
        texts: list of input strings.
        batch_size: batch size for model encoding.

    Returns:
        np.ndarray of shape (len(texts), dim) with dtype float32.
    """
    if _MODEL is None:
        return _fallback_encode_texts(texts)
    # Use the model to get embeddings (returns numpy array)
    embeddings = _MODEL.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    # ensure float32
    embeddings = embeddings.astype("float32")
    # L2-normalize each vector so that inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings
