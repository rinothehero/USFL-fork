from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import FrameworkAdapter
from ..normalization import normalize_sfl


class SFLAdapter(FrameworkAdapter):
    """
    SFL/USFL/SCAFFOLD adapter.

    Delegates to experiment_core.sfl_runner which calls simulation.py's
    run_simulation() directly (in-process asyncio), matching how the
    framework is designed to run.
    """
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

        # _spec_path is injected by run_experiment.py before reaching here.
        spec_path = execution.get("_spec_path", "spec.json")

        # Use absolute script path (not -m) because runner.py sets CWD to the
        # SFL framework directory, where experiment_core isn't importable.
        # sfl_runner.py handles its own sys.path setup internally.
        sfl_runner_script = repo_root / "experiment_core" / "sfl_runner.py"

        cmd = [
            str(execution.get("python", "python")),
            str(sfl_runner_script),
            "--spec", str(spec_path),
            "--repo-root", str(repo_root),
        ]

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
