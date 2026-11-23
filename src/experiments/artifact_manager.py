from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


class ArtifactManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def raw_path(self, artifact_name: str) -> Path:
        return self.root / artifact_name

    def unique_path(self, artifact_name: str) -> Path:
        target = self.root / artifact_name
        if not target.exists():
            return target
        stem = target.stem
        suffix = target.suffix
        index = 1
        while True:
            candidate = self.root / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def write_json(self, artifact_name: str, payload: Dict[str, Any]) -> Path:
        path = self.unique_path(artifact_name)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_lines(self, artifact_name: str, rows: Iterable[Dict[str, Any]]) -> Path:
        path = self.unique_path(artifact_name)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True))
                fh.write("\n")
        return path
