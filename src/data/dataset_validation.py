from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - dependency may be absent in tests
    load_dataset = None

RAW_SCIFACT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "raw_scifact" / "data"
SUPPORTED_LABELS = ("SUPPORT", "CONTRADICT", "NOT ENOUGH INFO")


def _normalize_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    value = str(label).strip().upper()
    if "SUPPORT" in value:
        return "SUPPORT"
    if "CONTRADICT" in value or "REFUTE" in value or "REJECT" in value:
        return "CONTRADICT"
    if "NOT ENOUGH INFO" in value or "INSUFFICIENT" in value or "NO INFO" in value:
        return "NOT ENOUGH INFO"
    return None


def infer_claim_label(evidence: Any) -> Optional[str]:
    labels = set()
    if not isinstance(evidence, dict):
        return None
    for _doc_id, entries in evidence.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = _normalize_label(entry.get("label"))
            if label is not None:
                labels.add(label)
    if not labels:
        return None
    if len(labels) > 1:
        return "AMBIGUOUS"
    return next(iter(labels))


def load_official_claims(raw_dir: Optional[Path | str] = None) -> Dict[str, Dict[str, Any]]:
    root = Path(raw_dir) if raw_dir is not None else RAW_SCIFACT_DIR
    if not root.exists():
        raise FileNotFoundError(f"Official SciFact split directory not found: {root}")

    claims: Dict[str, Dict[str, Any]] = {}
    for split_name, file_name in (("train", "claims_train.jsonl"), ("dev", "claims_dev.jsonl"), ("test", "claims_test.jsonl")):
        path = root / file_name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            claim_id = str(record.get("id"))
            claim = record.get("claim") or ""
            evidence = record.get("evidence") or {}
            label = infer_claim_label(evidence)
            claims[claim_id] = {
                "claim_id": claim_id,
                "split": split_name,
                "claim": claim,
                "evidence": evidence,
                "label": label,
                "cited_doc_ids": [str(doc_id) for doc_id in (record.get("cited_doc_ids") or [])],
            }
    return claims


def load_beir_queries() -> Dict[str, str]:
    if RAW_SCIFACT_DIR.exists():
        claims = load_official_claims(RAW_SCIFACT_DIR)
        claim_ids = {claim_id for claim_id in claims}
        train_ids = {claim_id for claim_id, info in claims.items() if info.get("split") == "train"}
        dev_ids = {claim_id for claim_id, info in claims.items() if info.get("split") == "dev"}
        # The BEIR queries are represented by the union of official train and dev claim IDs.
        return {claim_id: "" for claim_id in sorted(train_ids | dev_ids)}
    if load_dataset is None:
        return {}
    try:
        ds = load_dataset("BeIR/scifact", "queries", split="queries")
        return {str(row.get("_id")): str(row.get("text") or row.get("title") or "") for row in ds}
    except Exception:
        return {}


def load_beir_qrels() -> Dict[str, set[str]]:
    if RAW_SCIFACT_DIR.exists():
        claims = load_official_claims(RAW_SCIFACT_DIR)
        train_ids = {claim_id for claim_id, info in claims.items() if info.get("split") == "train"}
        dev_ids = {claim_id for claim_id, info in claims.items() if info.get("split") == "dev"}
        return {"train": train_ids, "test": dev_ids}
    if load_dataset is None:
        return {"train": set(), "test": set()}
    try:
        qrels = load_dataset("BeIR/scifact-qrels")
        result: Dict[str, set[str]] = {"train": set(), "test": set()}
        if hasattr(qrels, "keys"):
            for split in ("train", "test"):
                if split in qrels:
                    result[split] = {str(r.get("query-id")) for r in qrels[split]}
        return result
    except Exception:
        return {"train": set(), "test": set()}


def validate_split_integrity(raw_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    claims = load_official_claims(raw_dir)
    beir_queries = load_beir_queries()
    beir_qrels = load_beir_qrels()
    claim_ids_by_split = {"train": set(), "dev": set(), "test": set()}
    for claim_id, info in claims.items():
        split_name = info.get("split")
        if split_name in claim_ids_by_split:
            claim_ids_by_split[split_name].add(claim_id)

    query_ids = set(beir_queries)
    dev_ids = claim_ids_by_split.get("dev", set())
    test_ids = claim_ids_by_split.get("test", set())
    train_ids = claim_ids_by_split.get("train", set())
    qrels_train_ids = beir_qrels.get("train", set())
    qrels_test_ids = beir_qrels.get("test", set())

    summary = {
        "official_claims": len(claims),
        "train_claim_count": len(train_ids),
        "dev_claim_count": len(dev_ids),
        "test_claim_count": len(test_ids),
        "beir_query_count": len(query_ids),
        "train_qrel_count": len(qrels_train_ids),
        "test_qrel_count": len(qrels_test_ids),
        "beir_qrels_train_matches_train_claims": qrels_train_ids == train_ids,
        "beir_qrels_test_matches_dev_claims": qrels_test_ids == dev_ids,
        "official_test_ids_in_beir_queries": len(test_ids & query_ids),
        "official_test_ids_missing_from_beir_queries": sorted(test_ids - query_ids)[:20],
        "official_dev_ids_missing_from_beir_queries": sorted(dev_ids - query_ids)[:20],
        "official_train_ids_missing_from_beir_queries": sorted(train_ids - query_ids)[:20],
        "qrels_test_ids_missing_from_official_dev": sorted(qrels_test_ids - dev_ids)[:20],
        "qrels_train_ids_missing_from_official_train": sorted(qrels_train_ids - train_ids)[:20],
        "split_notes": [
            "BEIR/scifact-qrels train aligns with the official SciFact train claims.",
            "BEIR/scifact-qrels test aligns with the official SciFact dev claims, which means the BEIR test qrels are a dev split rather than a final test set.",
            "Official SciFact test claims do not overlap with BEIR queries or qrels and should remain untouched for final evaluation.",
        ],
    }
    return summary


def validate_split_alignment(raw_dir: Optional[Path | str] = None) -> bool:
    summary = validate_split_integrity(raw_dir)
    return bool(summary["beir_qrels_train_matches_train_claims"] and summary["beir_qrels_test_matches_dev_claims"])
