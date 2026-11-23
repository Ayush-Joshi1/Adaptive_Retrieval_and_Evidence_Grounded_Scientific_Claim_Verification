from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import faiss


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_faiss_index(
    index_path: str | Path,
    id_map_path: str | Path,
    corpus_path: str | Path,
    expected_dim: int | None = None,
    expected_corpus_count: int | None = None,
) -> Dict[str, Any]:
    issues: List[str] = []
    index_file = Path(index_path)
    id_map_file = Path(id_map_path)
    corpus_file = Path(corpus_path)

    if not index_file.exists():
        return {"is_valid": False, "issues": [f"Missing index file: {index_file}"], "metadata": {}}
    if not id_map_file.exists():
        return {"is_valid": False, "issues": [f"Missing ID map: {id_map_file}"], "metadata": {}}
    if not corpus_file.exists():
        return {"is_valid": False, "issues": [f"Missing corpus file: {corpus_file}"], "metadata": {}}

    index = faiss.read_index(str(index_file))
    id_map = json.loads(id_map_file.read_text(encoding="utf-8"))
    corpus_entries = [json.loads(line) for line in corpus_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    actual_dim = int(index.d)
    if expected_dim is not None and actual_dim != expected_dim:
        issues.append(f"Embedding dimension mismatch: expected {expected_dim}, found {actual_dim}.")

    actual_corpus_count = len(corpus_entries)
    if expected_corpus_count is not None and actual_corpus_count != expected_corpus_count:
        issues.append(f"Corpus count mismatch: expected {expected_corpus_count}, found {actual_corpus_count}.")

    if index.ntotal != len(id_map):
        issues.append(f"FAISS vector count ({index.ntotal}) does not match id_map length ({len(id_map)}).")
    if index.ntotal != actual_corpus_count:
        issues.append(f"FAISS vector count ({index.ntotal}) does not match corpus count ({actual_corpus_count}).")
    if len(id_map) != len(set(id_map)):
        issues.append("ID map contains duplicate corpus IDs.")

    corpus_ids = [str(item.get("corpus_id")) for item in corpus_entries]
    if len(corpus_ids) != len(set(corpus_ids)):
        issues.append("Corpus file contains duplicate corpus IDs.")
    if len(id_map) != len(corpus_ids):
        issues.append("ID map length differs from the number of corpus documents.")
    if set(id_map) != set(corpus_ids):
        issues.append("ID map and corpus ID sets are not identical.")

    metadata = {
        "index_path": str(index_file),
        "index_type": str(type(index).__name__),
        "index_dimension": actual_dim,
        "vector_count": int(index.ntotal),
        "id_map_length": len(id_map),
        "corpus_count": actual_corpus_count,
        "id_map_unique": len(id_map) == len(set(id_map)),
        "corpus_id_unique": len(corpus_ids) == len(set(corpus_ids)),
        "index_fingerprint": _hash_file(index_file),
        "id_map_fingerprint": _hash_file(id_map_file),
        "corpus_fingerprint": _hash_file(corpus_file),
    }
    return {"is_valid": not issues, "issues": issues, "metadata": metadata}


def validate_index_consistency(index_path: str | Path, id_map_path: str | Path, corpus_path: str | Path) -> Dict[str, Any]:
    return validate_faiss_index(index_path, id_map_path, corpus_path, expected_dim=384, expected_corpus_count=5183)
