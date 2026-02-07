from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import FrameworkAdapter
from ..normalization import normalize_multisfl


def _str_bool(v: bool) -> str:
    return "true" if v else "false"


# Unified g_oracle_mode → MultiSFL oracle_mode
_ORACLE_MODE_TO_MSFL = {"global": "master", "individual": "branch"}


class MultiSFLAdapter(FrameworkAdapter):
    name = "multisfl"

    def default_cwd(self, repo_root: Path) -> Path:
        return repo_root / "multisfl_implementation"

    def build_env(self, spec: Dict[str, Any]) -> Dict[str, str]:
        env = dict(spec.get("execution", {}).get("env", {}))
        return {str(k): str(v) for k, v in env.items()}

    def build_command(self, spec: Dict[str, Any], repo_root: Path) -> list[str]:
        execution = spec["execution"]
        if execution.get("command"):
            return [str(x) for x in execution["command"]]

        common = spec["common"]
        dist = common["distribution"]
        overrides = spec.get("framework_overrides", {})

        cmd = [
            str(execution.get("python", "python")),
            "run_multisfl.py",
            "--dataset", str(common["dataset"]),
            "--model_type", str(common["model"]),
            "--partition", str(dist.get("mode", "shard_dirichlet")),
            "--alpha_dirichlet", str(dist.get("dirichlet_alpha", 0.3)),
            "--shards", str(dist.get("labels_per_client", 2)),
            "--min_samples_per_client", str(dist.get("min_require_size", 10)),
            "--rounds", str(common["rounds"]),
            "--num_clients", str(common["total_clients"]),
            "--n_main", str(common["clients_per_round"]),
            "--batch_size", str(common["client_batch_size"]),
            "--local_steps", str(common["local_epochs"]),
            "--lr_client", str(common["learning_rate"]),
            "--lr_server", str(overrides.get("lr_server", common["learning_rate"])),
            "--momentum", str(common["momentum"]),
            "--seed", str(common["seed"]),
            "--device", str(common["device"]),
            "--split_layer", str(common["split_layer"]),
            "--enable_drift_measurement", _str_bool(bool(common.get("drift", {}).get("enabled", False))),
            "--drift_sample_interval", str(common.get("drift", {}).get("sample_interval", 1)),
            "--enable_g_measurement", _str_bool(bool(common.get("enable_g_measurement", False))),
            "--g_measure_frequency", str(common.get("g_measure_frequency", 10)),
            "--g_measurement_mode", str(common.get("g_measurement_mode", "single")),
            "--g_measurement_k", str(common.get("g_measurement_k", 5)),
            "--use_variance_g", _str_bool(bool(common.get("use_variance_g", False))),
            "--oracle_mode", _ORACLE_MODE_TO_MSFL.get(
                str(common.get("g_oracle_mode", "global")), "master"
            ),
            # Master pull
            "--alpha_master_pull", str(overrides.get("alpha_master_pull", 0.1)),
            # Sampling proportion scheduler
            "--p_update", str(overrides.get("p_update", "abs_ratio")),
            "--p0", str(overrides.get("p0", 0.01)),
            "--p_min", str(overrides.get("p_min", 0.01)),
            "--p_max", str(overrides.get("p_max", 0.5)),
            "--gamma", str(overrides.get("gamma", 0.5)),
            "--delta_clip", str(overrides.get("delta_clip", 0.2)),
            "--eps", str(overrides.get("eps", 1e-12)),
            # Replay
            "--replay_budget_mode", str(overrides.get("replay_budget_mode", "local_dataset")),
            "--replay_min_total", str(overrides.get("replay_min_total", 0)),
            "--max_assistant_trials", str(overrides.get("max_assistant_trials", 20)),
            # Gradient clipping (from common)
            "--clip_grad", _str_bool(bool(common.get("clip_grad", False))),
            "--clip_grad_max_norm", str(common.get("clip_grad_max_norm", 10.0)),
            # Transforms & init (from common)
            "--use_sfl_transform", _str_bool(bool(common.get("use_sfl_transform", False))),
            "--use_torchvision_init", _str_bool(bool(common.get("use_torchvision_init", False))),
        ]

        # Result output directory (from batch_runner)
        result_output_dir = execution.get("result_output_dir", "")
        if result_output_dir:
            cmd.extend(["--result_output_dir", result_output_dir])

        if overrides.get("branches") is not None:
            cmd.extend(["--branches", str(overrides["branches"])])
        if common.get("use_full_epochs", False):
            cmd.append("--use-full-epochs")

        extra = overrides.get("extra_cli", [])
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

        # Check unified result_output_dir first
        result_output_dir = execution.get("result_output_dir", "")
        if result_output_dir:
            d = Path(result_output_dir)
            if not d.is_absolute():
                d = (repo_root / d).resolve()
            found = self._newest_matching(d, "results_multisfl_*.json", started_epoch)
            if found:
                return found

        cwd = Path(execution.get("cwd") or self.default_cwd(repo_root))
        pattern = execution.get("raw_result_glob") or "results/results_multisfl_*.json"
        return self._newest_matching(cwd, pattern, started_epoch)

    def normalize(self, raw_payload: Dict[str, Any], raw_path: Path, run_meta: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_multisfl(raw_payload, raw_path, run_meta)
