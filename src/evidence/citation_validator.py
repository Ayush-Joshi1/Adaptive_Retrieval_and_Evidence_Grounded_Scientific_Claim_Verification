from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


class CitationValidator:
    """Validate evidence mappings against exact text in the original corpus."""

    def __init__(self, corpus: Mapping[str, Mapping[str, Any]]) -> None:
        self.corpus = {str(document_id): dict(document) for document_id, document in corpus.items()}

    @staticmethod
    def _normalize_sentence(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def validate(self, mappings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        valid = []
        invalid = []
        seen = set()
        for mapping in mappings:
            document_id = str(mapping.get("document_id") or "")
            sentence = self._normalize_sentence(mapping.get("evidence_sentence"))
            sentence_index = int(mapping.get("sentence_index", 0))
            result: Dict[str, Any] = {
                "document_id": document_id,
                "sentence_index": sentence_index,
                "evidence_sentence": mapping.get("evidence_sentence"),
                "valid": False,
                "reason": None,
            }
            document = self.corpus.get(document_id)
            if document is None:
                result["reason"] = "document_id_not_in_corpus"
                invalid.append(result)
                continue
            abstract = document.get("text") or document.get("abstract") or ""
            if not isinstance(abstract, str):
                abstract = str(abstract)
            # Reproduce sentence splitting conservatively while preserving exact text.
            boundary = re.compile(r"(?<=[.!?])(?:\s+|(?=[A-Z]))")
            positions = [match.start() for match in boundary.finditer(abstract.strip())]
            boundaries = [0] + positions
            if boundaries[-1] == 0:
                boundaries = [0, len(abstract.strip())]
            elif boundaries[-1] != len(abstract.strip()):
                boundaries.append(len(abstract.strip()))
            sentences = []
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
                candidate = abstract.strip()[start:end].strip()
                if candidate:
                    sentences.append((index, candidate))

            if sentence_index < 1 or sentence_index > len(sentences):
                result["reason"] = "sentence_index_out_of_range"
                invalid.append(result)
                continue
            original = next((text for index, text in sentences if index == sentence_index), "")
            if self._normalize_sentence(original) != sentence:
                result["reason"] = "evidence_text_modified"
                invalid.append(result)
                continue

            metadata_match = True
            source_metadata = mapping.get("source_metadata") or {}
            for key in ("corpus_id", "document_id"):
                if key in source_metadata and str(source_metadata[key]) != document_id:
                    metadata_match = False
            if not metadata_match:
                result["reason"] = "source_metadata_mismatch"
                invalid.append(result)
                continue

            citation_key = (document_id, sentence_index)
            if citation_key in seen:
                result["reason"] = "duplicate_citation"
                invalid.append(result)
                continue
            seen.add(citation_key)
            result["valid"] = True
            valid.append(result)
        return {
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "all_valid": bool(valid) and not invalid,
        }
