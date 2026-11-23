from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is optional for minimal installs
    yaml = None


@dataclass
class ExperimentConfig:
    dataset_name: str = "BEIR SciFact"
    split: str = "official_dev_for_tuning"
    dataset_source: str = "BeIR/scifact, BeIR/scifact-qrels, official SciFact claim files"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "not_pinned"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str = "not_pinned"
    retrieval_depths: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    random_seed: int = 42
    python_version: str = sys.version.split()[0]
    device: str = "CPU"
    run_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit_hash: Optional[str] = None
    generator_model: str = "gemini-2.5-flash-lite"
    generator_settings: Dict[str, Any] = field(default_factory=dict)
    metric_definitions: Dict[str, str] = field(default_factory=dict)

    def build_run_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def to_manifest(self) -> Dict[str, Any]:
        manifest = asdict(self)
        manifest["run_id"] = self.build_run_id()
        manifest["run_timestamp"] = self.run_timestamp
        return manifest


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    yaml_path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read experiment YAML configuration files.")
    with yaml_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        metric_definitions={
            "hit_at_k": "1 if any relevant document appears in the first K retrieved results; otherwise 0",
            "precision_at_k": "relevant retrieved within top-K divided by K",
            "recall_at_k": "relevant retrieved within top-K divided by all relevant documents for the query",
            "mrr_at_k": "reciprocal of the first relevant rank within K",
            "map_at_k": "average precision over the first K results",
            "ndcg_at_k": "discounted cumulative gain normalized by the ideal ranking under the same relevance scale",
        },
        generator_settings={"temperature": 0.0, "max_output_tokens": 256},
    )
