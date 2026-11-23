from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .dataset_validation import validate_split_integrity, load_official_claims


def build_split_manifest(raw_dir: Optional[Path | str] = None, output_path: Optional[Path | str] = None) -> Dict[str, Any]:
    summary = validate_split_integrity(raw_dir)
    manifest = {
        "dataset": "BEIR SciFact / official SciFact claims",
        "integrity_summary": summary,
        "rules": {
            "use_dev_for_tuning": True,
            "reserve_official_test_for_final_evaluation": True,
            "never_use_test_qrels_for_model_selection": True,
        },
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def official_split_ids(raw_dir: Optional[Path | str] = None) -> Dict[str, list[str]]:
    claims = load_official_claims(raw_dir)
    ids_by_split: Dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for claim_id, info in claims.items():
        split = info.get("split")
        if split in ids_by_split:
            ids_by_split[split].add(claim_id)
    return {split: sorted(values) for split, values in ids_by_split.items()}
