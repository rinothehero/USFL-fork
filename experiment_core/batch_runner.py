"""
Batch experiment runner: 하나의 공통 config로 여러 기법을 병렬 실행.

사용법:
    python -m experiment_core.batch_runner --spec batch_spec.json --repo-root .

batch_spec.json 예시:
{
  "output_dir": "results/compare_20260207",
  "common": {
    "dataset": "cifar10",
    "model": "resnet18_flex",
    "rounds": 100,
    "total_clients": 100,
    "clients_per_round": 10,
    "local_epochs": 5,
    "learning_rate": 0.001,
    "split_layer": "layer1.1.bn2",
    "distribution": {
      "mode": "shard_dirichlet",
      "dirichlet_alpha": 0.3,
      "labels_per_client": 2
    },
    "drift": {"enabled": true}
  },
  "experiments": [
    {"name": "sfl_baseline", "framework": "sfl", "method": "sfl", "gpu": 0},
    {"name": "usfl_full",    "framework": "sfl", "method": "usfl", "gpu": 1,
     "overrides": {"sfl_args": {"gradient-shuffle": true}}},
    {"name": "gas",          "framework": "gas", "gpu": 2},
    {"name": "multisfl_3br", "framework": "multisfl", "gpu": 3,
     "overrides": {"branches": 3, "lr_server": 0.01}}
  ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _build_single_spec(
    common: Dict[str, Any],
    exp: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """batch spec의 한 experiment 항목 → 단일 experiment spec으로 변환."""
    name = exp.get("name", exp.get("framework", "unknown"))
    framework = exp["framework"]
    method = exp.get("method", framework)

    # normalized 결과를 output_dir에 모으기
    norm_path = str(output_dir / f"{name}.normalized.json")

    spec: Dict[str, Any] = {
        "framework": framework,
        "method": method,
        "common": dict(common),
        "execution": {
            "mode": "run",
            "normalized_output": norm_path,
        },
        "framework_overrides": exp.get("overrides", {}),
    }

    # experiment-level common overrides (batch_size 등 실험별로 다를 수 있음)
    if "common" in exp:
        spec["common"] = _deep_merge(spec["common"], exp["common"])

    return spec


def _run_one(
    spec: Dict[str, Any],
    name: str,
    gpu: Optional[int],
    repo_root: Path,
    log_dir: Path,
) -> subprocess.Popen:
    """단일 실험을 subprocess로 시작. Popen 객체 반환."""
    # spec을 임시 파일로 저장
    spec_path = log_dir / f"{name}.spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "experiment_core.run_experiment",
        "--spec", str(spec_path),
        "--repo-root", str(repo_root),
    ]

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_file = log_dir / f"{name}.log"
    fh = open(log_file, "w", encoding="utf-8")

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=fh, stderr=subprocess.STDOUT,
        cwd=str(repo_root),
    )

    return proc


def run_batch(batch_spec_path: str, repo_root: Path) -> None:
    with open(batch_spec_path, "r", encoding="utf-8") as f:
        batch = json.load(f)

    common = batch.get("common", {})
    experiments: List[Dict[str, Any]] = batch["experiments"]

    # output directory
    timestamp = _utc_now_str()
    output_dir = Path(batch.get("output_dir", f"results/batch_{timestamp}"))
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # batch spec 자체도 output_dir에 복사
    shutil.copy2(batch_spec_path, output_dir / "batch_spec.json")

    print(f"[batch_runner] Output: {output_dir}")
    print(f"[batch_runner] Experiments: {len(experiments)}")
    print()

    # 각 실험 시작
    procs: List[Dict[str, Any]] = []
    for exp in experiments:
        name = exp.get("name", exp.get("framework", "unknown"))
        gpu = exp.get("gpu")
        spec = _build_single_spec(common, exp, output_dir)

        proc = _run_one(spec, name, gpu, repo_root, log_dir)
        gpu_str = f"GPU {gpu}" if gpu is not None else "CPU"
        print(f"  [{name}] started (PID={proc.pid}, {gpu_str})")
        procs.append({"name": name, "proc": proc, "gpu": gpu, "start": time.time()})

    print()
    print("[batch_runner] Waiting for all experiments to finish...")
    print()

    # 완료 대기
    results = []
    for item in procs:
        proc = item["proc"]
        proc.wait()
        elapsed = time.time() - item["start"]
        status = "OK" if proc.returncode == 0 else f"FAIL (exit={proc.returncode})"
        results.append({
            "name": item["name"],
            "gpu": item["gpu"],
            "exit_code": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
        })
        print(f"  [{item['name']}] {status}  ({elapsed:.0f}s)")

    # summary 저장
    summary = {
        "timestamp": timestamp,
        "batch_spec": str(batch_spec_path),
        "output_dir": str(output_dir),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"[batch_runner] Done. Results in: {output_dir}/")
    print(f"  summary.json          - 실행 결과 요약")
    print(f"  *.normalized.json     - 통일된 실험 결과")
    print(f"  logs/*.log            - 각 실험 stdout/stderr")

    # 실패가 있으면 exit code 1
    if any(r["exit_code"] != 0 for r in results):
        failed = [r["name"] for r in results if r["exit_code"] != 0]
        print(f"\n  WARNING: {len(failed)} experiment(s) failed: {', '.join(failed)}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple experiments in parallel with shared config",
    )
    parser.add_argument("--spec", required=True, help="Path to batch spec JSON")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    args = parser.parse_args()

    run_batch(args.spec, Path(args.repo_root).resolve())


if __name__ == "__main__":
    main()
