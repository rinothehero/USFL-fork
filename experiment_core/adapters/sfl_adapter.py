from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import FrameworkAdapter
from ..normalization import normalize_sfl


class SFLAdapter(FrameworkAdapter):
    name = "sfl"

    def default_cwd(self, repo_root: Path) -> Path:
        return repo_root / "sfl_framework-fork-feature-training-tracker"

    def build_env(self, spec: Dict[str, Any]) -> Dict[str, str]:
        env = dict(spec.get("execution", {}).get("env", {}))
        return {str(k): str(v) for k, v in env.items()}

    def build_command(self, spec: Dict[str, Any], repo_root: Path) -> list[str]:
        execution = spec["execution"]
        if execution.get("command"):
            return [str(x) for x in execution["command"]]

        common = spec["common"]
        dist = common["distribution"]

        method = str(spec.get("method", "sfl"))
        selector_default = "usfl" if method == "usfl" else "uniform"
        batch_size = (
            int(common["server_batch_size"]) if method == "usfl" else int(common["client_batch_size"])
        )

        cmd = [
            str(execution.get("python", "python")),
            "server/main.py",
            "--dataset", str(common["dataset"]),
            "--model", str(common["model"]),
            "--method", method,
            "--criterion", "ce",
            "--optimizer", "sgd",
            "--learning_rate", str(common["learning_rate"]),
            "--momentum", str(common["momentum"]),
            "--local-epochs", str(common["local_epochs"]),
            "--global-round", str(common["rounds"]),
            "--batch-size", str(batch_size),
            "--device", str(common["device"]),
            "--selector", str(spec.get("framework_overrides", {}).get("selector", selector_default)),
            "--aggregator", str(spec.get("framework_overrides", {}).get("aggregator", "fedavg")),
            "--distributer", str(dist["mode"]),
            "--num-clients", str(common["total_clients"]),
            "--num-clients-per-round", str(common["clients_per_round"]),
            "--split-strategy", "layer_name",
            "--split-layer", str(common["split_layer"]),
            "--seed", str(common["seed"]),
        ]

        if dist.get("dirichlet_alpha") is not None:
            cmd.extend(["--dirichlet-alpha", str(dist["dirichlet_alpha"])])
        if dist.get("labels_per_client") is not None:
            cmd.extend(["--labels-per-client", str(dist["labels_per_client"])])
        if dist.get("min_require_size") is not None:
            cmd.extend(["--min-require-size", str(dist["min_require_size"])])

        if common.get("drift", {}).get("enabled", False):
            cmd.append("--enable-drift-measurement")
            cmd.extend([
                "--drift-sample-interval",
                str(common.get("drift", {}).get("sample_interval", 1)),
            ])

        sfl_overrides = spec.get("framework_overrides", {}).get("sfl_args", {})
        for k, v in sfl_overrides.items():
            flag = k if str(k).startswith("-") else f"--{k}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            elif isinstance(v, (list, tuple)):
                cmd.append(flag)
                cmd.extend([str(x) for x in v])
            else:
                cmd.extend([flag, str(v)])

        extra = spec.get("framework_overrides", {}).get("extra_cli", [])
        cmd.extend([str(x) for x in extra])

        return cmd

    def find_latest_raw_result(
        self,
        spec: Dict[str, Any],
        repo_root: Path,
        started_epoch: float,
    ) -> Optional[Path]:
        execution = spec["execution"]
        explicit = execution.get("raw_result_path")
        if explicit:
            p = Path(explicit)
            return p if p.is_absolute() else (repo_root / p).resolve()

        cwd = Path(execution.get("cwd") or self.default_cwd(repo_root))
        pattern = execution.get("raw_result_glob") or "result-*.json"
        return self._newest_matching(cwd, pattern, started_epoch)

    def normalize(self, raw_payload: Dict[str, Any], raw_path: Path, run_meta: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_sfl(raw_payload, raw_path, run_meta)
