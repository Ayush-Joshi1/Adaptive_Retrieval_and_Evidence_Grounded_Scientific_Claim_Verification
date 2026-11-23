from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from src.retrieval.embedder import encode_texts
except Exception:  # pragma: no cover - fallback for isolated imports
    encode_texts = None


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|(?=[A-Z]))")


def split_sentences(text: str) -> List[Dict[str, Any]]:
    """Split text while preserving exact sentence text and character offsets."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.strip()
    positions = [match.start() for match in _SENTENCE_BOUNDARY.finditer(text)]
    boundaries = [0] + positions
    if boundaries[-1] == 0:
        boundaries = [0, len(text)]
    elif boundaries[-1] != len(text):
        boundaries.append(len(text))

    sentences: List[Dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        sentence = text[start:end].strip()
        if not sentence:
            continue
        absolute_start = text.find(sentence, start)
        if absolute_start < 0:
            absolute_start = start
        sentences.append({
            "sentence": sentence,
            "sentence_index": index,
            "start_char": absolute_start,
            "end_char": absolute_start + len(sentence),
        })
    return sentences


class EvidenceSelector:
    """Select claim-relevant sentences using the existing CPU sentence embedder."""

    def __init__(
        self,
        max_sentences_per_document: int = 2,
        max_total_sentences: int = 6,
        relevance_threshold: float = 0.25,
        embedding_batch_size: int = 64,
    ) -> None:
        self.max_sentences_per_document = max(1, int(max_sentences_per_document))
        self.max_total_sentences = max(1, int(max_total_sentences))
        self.relevance_threshold = float(relevance_threshold)
        self.embedding_batch_size = max(1, int(embedding_batch_size))

    @staticmethod
    def _document_id(document: Mapping[str, Any]) -> str:
        value = document.get("document_id") or document.get("corpus_id") or document.get("_id")
        return str(value) if value is not None else ""

    @staticmethod
    def _fallback_similarity(claim: str, sentence: str) -> float:
        def tokens(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", str(text).lower()))
        claim_tokens = tokens(claim)
        sentence_tokens = tokens(sentence)
        if not claim_tokens or not sentence_tokens:
            return 0.0
        overlap = len(claim_tokens & sentence_tokens)
        return overlap / max(1, len(claim_tokens | sentence_tokens))

    def select(
        self,
        claim: str,
        documents: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        claim = "" if claim is None else str(claim)
        candidates: List[Dict[str, Any]] = []
        for document_rank, document in enumerate(documents, start=1):
            document_id = self._document_id(document)
            if not document_id:
                continue
            abstract = document.get("text")
            if abstract is None:
                abstract = document.get("abstract")
            if abstract is None:
                abstract = ""
            split = split_sentences(str(abstract))
            for sentence_info in split:
                candidates.append({
                    "document_id": document_id,
                    "document_rank": document_rank,
                    "evidence_sentence": sentence_info["sentence"],
                    "sentence_index": sentence_info["sentence_index"],
                    "start_char": sentence_info["start_char"],
                    "end_char": sentence_info["end_char"],
                })

        if not candidates or not claim.strip():
            return []

        sentences = [item["evidence_sentence"] for item in candidates]
        raw_scores: List[float] = []
        if encode_texts is not None:
            try:
                embeddings = encode_texts([claim] + sentences, batch_size=self.embedding_batch_size)
                claim_embedding = embeddings[0]
                sentence_embeddings = embeddings[1:]
                raw_scores = [float(sentence_embedding @ claim_embedding) for sentence_embedding in sentence_embeddings]
            except Exception:
                raw_scores = []
        if len(raw_scores) != len(candidates):
            raw_scores = [self._fallback_similarity(claim, sentence) for sentence in sentences]

        minimum = min(raw_scores) if raw_scores else 0.0
        maximum = max(raw_scores) if raw_scores else 0.0
        if maximum <= minimum:
            normalized = [0.5 if raw_score > 0 else 0.0 for raw_score in raw_scores]
        else:
            normalized = [(raw_score - minimum) / (maximum - minimum) for raw_score in raw_scores]

        for candidate, raw_score, relevance_score in zip(candidates, raw_scores, normalized):
            candidate["raw_relevance_score"] = float(raw_score)
            candidate["evidence_relevance_score"] = float(relevance_score)

        # Keep the strongest sentence from each document, then rank globally.
        per_document: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            per_document.setdefault(candidate["document_id"], []).append(candidate)
        selected: List[Dict[str, Any]] = []
        for document_candidates in per_document.values():
            document_candidates.sort(
                key=lambda item: (
                    -item["evidence_relevance_score"],
                    item["sentence_index"],
                    item["evidence_sentence"],
                )
            )
            selected.extend(document_candidates[:self.max_sentences_per_document])

        selected.sort(
            key=lambda item: (
                -item["evidence_relevance_score"],
                item["document_rank"],
                item["sentence_index"],
                item["evidence_sentence"],
            )
        )

        deduplicated: Dict[str, Dict[str, Any]] = {}
        for candidate in selected:
            normalized_text = re.sub(r"\s+", " ", candidate["evidence_sentence"]).strip().lower()
            if normalized_text not in deduplicated:
                item = dict(candidate)
                item["source_document_ids"] = [candidate["document_id"]]
                item["source_traceability"] = [{
                    "document_id": candidate["document_id"],
                    "sentence_index": candidate["sentence_index"],
                    "start_char": candidate["start_char"],
                    "end_char": candidate["end_char"],
                }]
                deduplicated[normalized_text] = item
            else:
                item = deduplicated[normalized_text]
                if candidate["document_id"] not in item["source_document_ids"]:
                    item["source_document_ids"].append(candidate["document_id"])
                item["source_traceability"].append({
                    "document_id": candidate["document_id"],
                    "sentence_index": candidate["sentence_index"],
                    "start_char": candidate["start_char"],
                    "end_char": candidate["end_char"],
                })

        ranked = list(deduplicated.values())
        ranked = [item for item in ranked if item["evidence_relevance_score"] >= self.relevance_threshold]
        return ranked[:self.max_total_sentences]
