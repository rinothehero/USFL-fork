from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ALLOWED_FRAMEWORKS = {"sfl", "gas", "multisfl"}


@dataclass(frozen=True)
class ExperimentSpec:
    raw: Dict[str, Any]

    @property
    def framework(self) -> str:
        return str(self.raw["framework"]).lower()

    @property
    def method(self) -> str:
        return str(self.raw.get("method", ""))

    @property
    def common(self) -> Dict[str, Any]:
        return self.raw["common"]

    @property
    def execution(self) -> Dict[str, Any]:
        return self.raw["execution"]

    @property
    def framework_overrides(self) -> Dict[str, Any]:
        return self.raw.get("framework_overrides", {})


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "YAML spec requires PyYAML. Install with `pip install pyyaml` or use JSON."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Spec at {path} must be a mapping object")
    return payload


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Spec at {path} must be a mapping object")
    return payload


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _defaults() -> Dict[str, Any]:
    return {
        "framework": "sfl",
        "method": "sfl",
        "common": {
            "dataset": "cifar10",
            "model": "resnet18_flex",
            "seed": 42,
            "rounds": 300,
            "total_clients": 100,
            "clients_per_round": 10,
            "local_epochs": 5,
            "client_batch_size": 50,
            "server_batch_size": 500,
            "learning_rate": 0.001,
            "momentum": 0.0,
            "device": "cuda",
            "split_layer": "layer1.1.bn2",
            "distribution": {
                "mode": "uniform",
                "dirichlet_alpha": 0.3,
                "labels_per_client": 2,
                "min_require_size": 10,
            },
            "drift": {
                "enabled": True,
                "sample_interval": 1,
            },
        },
        "execution": {
            "mode": "run",
            "python": "python",
            "cwd": None,
            "env": {},
            "command": None,
            "raw_result_path": None,
            "raw_result_glob": None,
            "normalized_output": None,
        },
        "framework_overrides": {},
    }


def _validate(payload: Dict[str, Any]) -> None:
    framework = str(payload.get("framework", "")).lower()
    if framework not in ALLOWED_FRAMEWORKS:
        raise ValueError(
            f"Unsupported framework '{framework}'. Expected one of {sorted(ALLOWED_FRAMEWORKS)}"
        )

    common = payload.get("common", {})
    required_common = [
        "dataset",
        "model",
        "seed",
        "rounds",
        "total_clients",
        "clients_per_round",
        "local_epochs",
        "client_batch_size",
        "server_batch_size",
        "learning_rate",
        "momentum",
        "device",
        "split_layer",
        "distribution",
        "drift",
    ]
    missing_common = [k for k in required_common if k not in common]
    if missing_common:
        raise ValueError(f"Missing common keys: {missing_common}")

    execution = payload.get("execution", {})
    mode = execution.get("mode", "run")
    if mode not in {"run", "normalize_only"}:
        raise ValueError("execution.mode must be 'run' or 'normalize_only'")

    if mode == "normalize_only" and not execution.get("raw_result_path"):
        raise ValueError("execution.raw_result_path is required for normalize_only mode")


def load_spec(path: str | Path) -> ExperimentSpec:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spec not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        user_payload = _load_yaml(p)
    elif suffix == ".json":
        user_payload = _load_json(p)
    else:
        raise ValueError(f"Unsupported spec extension: {suffix}")

    merged = _deep_merge(_defaults(), user_payload)
    _validate(merged)
    return ExperimentSpec(raw=merged)


def resolve_path(path_or_none: Optional[str], repo_root: Path) -> Optional[Path]:
    if path_or_none is None:
        return None
    p = Path(path_or_none)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()
