"""
CLI entry point for the unified simulation framework.

Usage:
    python -m unified_simulation_framework.entry \
        --method sfl \
        --sfl-args "-d cifar10 -m resnet18_flex -M sfl -le 5 -gr 10 ..."

    Or from Python:
        from unified_simulation_framework.entry import run
        results = run(method="sfl", sfl_args=["-d", "cifar10", ...])
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import UnifiedConfig, parse_unified_config, SFL_DIR
from .client_ops import RoundResult
from .trainer import SimTrainer


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results: List[RoundResult], output_dir: str, method: str):
    """Save round results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"unified_{method}_results.json")

    data = {
        "method": method,
        "rounds": [
            {
                "round": r.round_number,
                "accuracy": r.accuracy,
                "loss": r.loss,
                **r.metrics,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


def run(
    method: str,
    sfl_args: List[str],
    extra_args: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> List[RoundResult]:
    """
    Run a complete experiment.

    Args:
        method: 'sfl', 'usfl', 'scaffold_sfl', 'mix2sfl', 'gas', 'multisfl'
        sfl_args: CLI args in SFL format (short flags)
        extra_args: Method-specific overrides
        output_dir: Where to save results (optional)
    """
    # SFL framework expects CWD to be its directory for dataset/model paths
    original_cwd = os.getcwd()
    os.chdir(SFL_DIR)

    try:
        config = parse_unified_config(method, sfl_args, extra_args)

        if config.seed is not None:
            set_seed(config.seed)

        print(f"[unified] Method: {method}")
        print(f"[unified] Dataset: {config.dataset}, Model: {config.model_name}")
        print(f"[unified] Rounds: {config.global_round}, Clients: {config.num_clients}")
        print(f"[unified] Device: {config.device}")
        print()

        trainer = SimTrainer(config)
        results = trainer.train()

        if output_dir:
            save_results(results, output_dir, method)

        return results
    finally:
        os.chdir(original_cwd)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified SFL Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic SFL on CIFAR-10
  python -m unified_simulation_framework.entry \\
      --method sfl \\
      --sfl-args "-d cifar10 -m resnet18_flex -M sfl -le 5 -gr 10 -bs 50 \\
                  -nc 100 -ncpr 10 -ss layer_name -sl layer2 -sd 42 \\
                  -distr shard_dirichlet -diri-alpha 0.3 -lpc 2 \\
                  -o sgd -lr 0.001 -s uniform -aggr fedavg -c ce \\
                  -de cuda -sma false -nnf"
        """,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["sfl", "usfl", "scaffold_sfl", "mix2sfl", "gas", "multisfl"],
    )
    parser.add_argument(
        "--sfl-args",
        required=True,
        type=str,
        help="SFL-format CLI args as a single quoted string",
    )
    parser.add_argument("--output-dir", type=str, default=None)

    # GAS-specific
    parser.add_argument("--gas-generate-features", action="store_true")
    parser.add_argument("--gas-logit-adjustment", action="store_true", default=True)
    parser.add_argument("--gas-v-test", action="store_true")

    # MultiSFL-specific
    parser.add_argument("--multisfl-branches", type=int, default=3)
    parser.add_argument("--multisfl-alpha-master-pull", type=float, default=0.1)
    parser.add_argument("--multisfl-p-update", type=str, default="abs_ratio")

    args = parser.parse_args()
    sfl_args = args.sfl_args.split()

    extra_args = {}
    for k, v in vars(args).items():
        if k.startswith("gas_") or k.startswith("multisfl_"):
            extra_args[k] = v

    run(
        method=args.method,
        sfl_args=sfl_args,
        extra_args=extra_args if extra_args else None,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
