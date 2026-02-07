from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import FrameworkAdapter
from ..normalization import normalize_gas


def _b(value: bool) -> str:
    return "true" if value else "false"


# Unified g_oracle_mode → GAS g_measure_mode
_ORACLE_MODE_TO_GAS = {"global": "strict", "individual": "realistic"}


# GAS only supports these models.  Map unified spec names → GAS internal names.
_GAS_MODEL_MAP: Dict[str, str] = {
    "resnet18": "resnet18",
    "resnet18_flex": "resnet18",
    "resnet18_image_style": "resnet18",
    "alexnet": "alexnet",
}

# Map unified distribution mode → GAS internal boolean flags
_GAS_DISTRIBUTION_MAP: Dict[str, Dict[str, str]] = {
    "uniform": {"GAS_IID": "true", "GAS_DIRICHLET": "false", "GAS_LABEL_DIRICHLET": "false"},
    "iid": {"GAS_IID": "true", "GAS_DIRICHLET": "false", "GAS_LABEL_DIRICHLET": "false"},
    "dirichlet": {"GAS_IID": "false", "GAS_DIRICHLET": "true", "GAS_LABEL_DIRICHLET": "false"},
    "shard_dirichlet": {"GAS_IID": "false", "GAS_DIRICHLET": "false", "GAS_LABEL_DIRICHLET": "true"},
    "label": {"GAS_IID": "false", "GAS_DIRICHLET": "false", "GAS_LABEL_DIRICHLET": "false"},
}


class GASAdapter(FrameworkAdapter):
    name = "gas"

    def default_cwd(self, repo_root: Path) -> Path:
        return repo_root / "GAS_implementation"

    def build_env(self, spec: Dict[str, Any]) -> Dict[str, str]:
        common = spec["common"]
        dist = common["distribution"]

        # Model mapping with validation
        model_name = str(common["model"]).lower()
        gas_model = _GAS_MODEL_MAP.get(model_name)
        if gas_model is None:
            supported = ", ".join(sorted(_GAS_MODEL_MAP.keys()))
            raise ValueError(
                f"GAS does not support model '{common['model']}'. "
                f"Supported models: {supported}"
            )

        # Distribution mode mapping
        dist_mode = str(dist.get("mode", "shard_dirichlet")).lower()
        dist_flags = _GAS_DISTRIBUTION_MAP.get(dist_mode, {})

        env = {
            "GAS_DATASET": str(common["dataset"]),
            "GAS_MODEL": gas_model,
            "GAS_BATCH_SIZE": str(common["client_batch_size"]),
            "GAS_LABELS_PER_CLIENT": str(dist.get("labels_per_client", 2)),
            "GAS_DIRICHLET_ALPHA": str(dist.get("dirichlet_alpha", 0.3)),
            "GAS_MIN_REQUIRE_SIZE": str(dist.get("min_require_size", 10)),
            "GAS_GLOBAL_EPOCHS": str(common["rounds"]),
            "GAS_LOCAL_EPOCHS": str(common["local_epochs"]),
            "GAS_TOTAL_CLIENTS": str(common["total_clients"]),
            "GAS_CLIENTS_PER_ROUND": str(common["clients_per_round"]),
            "GAS_LR": str(common["learning_rate"]),
            "GAS_MOMENTUM": str(common["momentum"]),
            "GAS_SPLIT_LAYER": str(common["split_layer"]),
            "GAS_SEED": str(common["seed"]),
            "GAS_DRIFT_MEASUREMENT": _b(bool(common.get("drift", {}).get("enabled", False))),
            "GAS_DRIFT_SAMPLE_INTERVAL": str(common.get("drift", {}).get("sample_interval", 1)),
            "GAS_G_MEASUREMENT": _b(bool(common.get("enable_g_measurement", False))),
            "GAS_G_MEASUREMENT_ACCUMULATION": str(common.get("g_measurement_mode", "single")),
            "GAS_G_MEASUREMENT_K": str(common.get("g_measurement_k", 5)),
            "GAS_USE_VARIANCE_G": _b(bool(common.get("use_variance_g", False))),
            "GAS_G_MEASURE_FREQUENCY": str(common.get("g_measure_frequency", 10)),
            "GAS_G_MEASURE_MODE": _ORACLE_MODE_TO_GAS.get(
                str(common.get("g_oracle_mode", "global")), "strict"
            ),
            "GAS_CLIP_GRAD": _b(bool(common.get("clip_grad", False))),
            "GAS_CLIP_GRAD_MAX_NORM": str(common.get("clip_grad_max_norm", 10.0)),
            "GAS_WEIGHT_DECAY": str(common.get("weight_decay", 0.0)),
            "GAS_USE_SFL_TRANSFORM": _b(bool(common.get("use_sfl_transform", False))),
            "GAS_USE_TORCHVISION_INIT": _b(bool(common.get("use_torchvision_init", False))),
            "GAS_USE_FULL_EPOCHS": _b(bool(common.get("use_full_epochs", False))),
        }

        # Apply distribution mode flags
        env.update(dist_flags)

        # Result output directory (from batch_runner)
        result_output_dir = spec.get("execution", {}).get("result_output_dir", "")
        if result_output_dir:
            env["GAS_RESULT_OUTPUT_DIR"] = result_output_dir

        gas_overrides = spec.get("framework_overrides", {}).get("gas_env", {})
        for k, v in gas_overrides.items():
            env[str(k)] = str(v)

        custom_env = spec.get("execution", {}).get("env", {})
        for k, v in custom_env.items():
            env[str(k)] = str(v)

        return env

    def build_command(self, spec: Dict[str, Any], repo_root: Path) -> list[str]:
        execution = spec["execution"]
        if execution.get("command"):
            return [str(x) for x in execution["command"]]

        cmd = [str(execution.get("python", "python")), "GAS_main.py"]

        # Preserve existing GAS behavior: first positional arg toggles variance G mode
        use_variance_g = bool(spec.get("common", {}).get("use_variance_g", False))
        cmd.append("true" if use_variance_g else "false")

        gas_flags = spec.get("framework_overrides", {}).get("gas_flags", [])
        cmd.extend([str(x) for x in gas_flags])
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

        # Check unified result_output_dir first
        result_output_dir = execution.get("result_output_dir", "")
        if result_output_dir:
            d = Path(result_output_dir)
            if not d.is_absolute():
                d = (repo_root / d).resolve()
            found = self._newest_matching(d, "results_gas_*.json", started_epoch)
            if found:
                return found

        cwd = Path(execution.get("cwd") or self.default_cwd(repo_root))
        pattern = execution.get("raw_result_glob") or "results/results_gas_*.json"
        return self._newest_matching(cwd, pattern, started_epoch)

    def normalize(self, raw_payload: Dict[str, Any], raw_path: Path, run_meta: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_gas(raw_payload, raw_path, run_meta)
