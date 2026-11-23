from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RUN_ROOT = Path(__file__).resolve().parents[2] / "results" / "runs"


def make_experiment_key(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def unique_run_dir(run_root: str | Path = RUN_ROOT, prefix: str = "run") -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate = root / f"{prefix}_{suffix:04d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def protected_write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        suffix = 1
        while True:
            candidate = target.with_name(f"{target.stem}_{suffix}{target.suffix}")
            if not candidate.exists():
                target = candidate
                break
            suffix += 1
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def deduplicate_records(records: Iterable[Dict[str, Any]], key_field: str = "experiment_key") -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = str(record.get(key_field, make_experiment_key(record)))
        deduped.setdefault(key, record)
    return list(deduped.values())


def status_record(experiment_key: str, status: str, *, error_category: Optional[str] = None, retry_count: int = 0, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    record = {
        "experiment_key": experiment_key,
        "status": status,
        "retry_count": int(retry_count),
        "error_category": error_category,
        "details": details or {},
    }
    if status == "failed" and error_category is None:
        record["error_category"] = "unknown"
    return record
