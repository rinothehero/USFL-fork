from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_experiment
from .spec import load_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for SFL/GAS/MultiSFL",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to experiment spec JSON/YAML",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current working directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec)
    # Inject resolved spec file path so adapters (e.g. SFL) can reference
    # the on-disk spec when building subprocess commands.
    spec.raw.setdefault("execution", {})["_spec_path"] = str(
        Path(args.spec).resolve()
    )
    spec.raw["execution"]["_repo_root"] = str(Path(args.repo_root).resolve())
    outcome = run_experiment(spec, Path(args.repo_root).resolve())

    print("[experiment_core] done")
    print(f"framework: {spec.framework}")
    print(f"method: {spec.method}")
    print(f"raw_result: {outcome.raw_result_path}")
    print(f"normalized_result: {outcome.normalized_result_path}")
    print(f"exit_code: {outcome.exit_code}")


if __name__ == "__main__":
    main()
