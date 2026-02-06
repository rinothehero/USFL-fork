from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import get_adapter
from .normalization import load_raw_result
from .spec import ExperimentSpec, resolve_path


@dataclass
class RunOutcome:
    raw_result_path: Path
    normalized_result_path: Path
    exit_code: int
    command: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_normalized_path(raw_result_path: Path) -> Path:
    return raw_result_path.with_name(f"{raw_result_path.stem}.normalized.json")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_experiment(spec: ExperimentSpec, repo_root: Path) -> RunOutcome:
    adapter = get_adapter(spec.framework)
    execution = spec.execution

    mode = execution.get("mode", "run")

    cwd = resolve_path(execution.get("cwd"), repo_root) or adapter.default_cwd(repo_root)
    cwd = cwd.resolve()

    command = adapter.build_command(spec.raw, repo_root)
    env = os.environ.copy()
    env.update(adapter.build_env(spec.raw))

    started_epoch = time.time()
    started_at = _utc_now_iso()

    exit_code = 0
    if mode == "run":
        proc = subprocess.run(
            command, cwd=str(cwd), env=env, check=False,
            capture_output=True, text=True,
        )
        exit_code = int(proc.returncode)
        if exit_code != 0:
            err_detail = []
            if proc.stdout:
                err_detail.append(f"STDOUT (last 2000 chars):\n{proc.stdout[-2000:]}")
            if proc.stderr:
                err_detail.append(f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}")
            detail_str = "\n".join(err_detail) if err_detail else "(no output captured)"
            raise RuntimeError(
                f"Experiment command failed with exit code {exit_code}:\n"
                f"  Command: {' '.join(command)}\n"
                f"  CWD: {cwd}\n"
                f"{detail_str}"
            )

    explicit_raw = resolve_path(execution.get("raw_result_path"), repo_root)
    raw_result_path = explicit_raw
    if raw_result_path is None:
        raw_result_path = adapter.find_latest_raw_result(spec.raw, repo_root, started_epoch)

    if raw_result_path is None:
        raise FileNotFoundError(
            "Could not locate raw result file. Set execution.raw_result_path explicitly."
        )

    raw_result_path = raw_result_path.resolve()
    raw_payload = load_raw_result(raw_result_path)

    run_meta = {
        "framework": spec.framework,
        "method": spec.method,
        "mode": mode,
        "command": command,
        "cwd": str(cwd),
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "exit_code": exit_code,
    }

    normalized_payload = adapter.normalize(raw_payload, raw_result_path, run_meta)

    normalized_path = resolve_path(execution.get("normalized_output"), repo_root)
    if normalized_path is None:
        normalized_path = _default_normalized_path(raw_result_path)

    _write_json(normalized_path, normalized_payload)

    return RunOutcome(
        raw_result_path=raw_result_path,
        normalized_result_path=normalized_path,
        exit_code=exit_code,
        command=command,
    )
