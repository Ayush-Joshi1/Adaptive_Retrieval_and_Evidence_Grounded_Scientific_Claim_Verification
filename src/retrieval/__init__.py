"""Retrieval package exposing embedder and retriever utilities."""

from .adaptive_retriever import AdaptiveRetriever, ConfidenceAdaptiveStrategy, ScoreGapAdaptiveStrategy
from .bm25_retriever import BM25Retriever
from .cross_encoder_reranker import CrossEncoderReranker
from .embedder import encode_texts
from .evidence_aware_reranker import EvidenceAwareReranker
from .hybrid_retriever import HybridRetriever
from .retriever import Retriever

__all__ = [
    "encode_texts",
    "Retriever",
    "BM25Retriever",
    "HybridRetriever",
    "AdaptiveRetriever",
    "ScoreGapAdaptiveStrategy",
    "ConfidenceAdaptiveStrategy",
    "CrossEncoderReranker",
    "EvidenceAwareReranker",
]
