"""
CLI entry point and result saving.

Usage:
    # 1. Pre-compute schedule (once)
    python -m sfl_sim.prepare --config experiments/sfl_cifar10.json

    # 2. Train (reuse same schedule for all methods)
    python -m sfl_sim --config experiments/sfl_cifar10.json --schedule-dir schedules/seed42_cifar10_100c_shard_dirichlet_a0.3_lpc2/
    python -m sfl_sim --config experiments/usfl_cifar10.json --schedule-dir schedules/seed42_cifar10_100c_shard_dirichlet_a0.3_lpc2/
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch

from .config import Config, parse_args
from .client_ops import RoundResult


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_schedule_dir(schedule_dir: str, config: Config):
    """
    Load pre-computed data_map, schedule, and meta from directory.
    Auto-fills config fields from meta.json.

    Returns:
        (client_data_masks, selection_schedule)
    """
    sdir = Path(schedule_dir)
    if not sdir.exists():
        raise FileNotFoundError(f"Schedule directory not found: {sdir}")

    # Load meta.json and fill config
    meta_path = sdir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {sdir}")
    with open(meta_path) as f:
        meta = json.load(f)

    # Auto-fill config from meta (schedule defines these, not experiment config)
    config.dataset = meta["dataset"]
    config.num_clients = meta["num_clients"]
    config.num_clients_per_round = meta["clients_per_round"]
    config.global_round = meta["rounds"]
    config.distribution = meta["distribution"]
    config.dirichlet_alpha = meta["alpha"]
    config.labels_per_client = meta["labels_per_client"]
    config.selector = meta["selector"]

    # Load data_map.json
    data_map_path = sdir / "data_map.json"
    if not data_map_path.exists():
        raise FileNotFoundError(f"data_map.json not found in {sdir}")
    with open(data_map_path) as f:
        raw_map = json.load(f)
    client_data_masks = {int(k): v for k, v in raw_map.items()}

    # Load schedule.json
    schedule_path = sdir / "schedule.json"
    if not schedule_path.exists():
        raise FileNotFoundError(f"schedule.json not found in {sdir}")
    with open(schedule_path) as f:
        raw_schedule = json.load(f)

    selection_schedule = []
    for r in range(1, config.global_round + 1):
        key = str(r)
        if key not in raw_schedule:
            raise ValueError(
                f"Schedule has {len(raw_schedule)} rounds but need {config.global_round}. "
                f"Re-run: python -m sfl_sim.prepare -gr {config.global_round}"
            )
        selection_schedule.append(raw_schedule[key])

    print(f"[sfl_sim] Loaded schedule from {sdir}/", flush=True)
    print(f"  {meta['dataset']}, {meta['num_clients']} clients, "
          f"{meta['rounds']} rounds, selector={meta['selector']}", flush=True)

    return client_data_masks, selection_schedule


_METHOD_FIELDS = {
    "usfl": [
        "balancing_strategy", "balancing_target",
        "gradient_shuffle", "gradient_shuffle_strategy",
        "gradient_average_weight", "adaptive_alpha_beta",
        "use_dynamic_batch_scheduler",
    ],
    "mix2sfl": [
        "mix2sfl_beta_alpha", "mix2sfl_smashmix_ratio",
    ],
    "multisfl": [
        "num_branches", "alpha_master_pull",
        "p_update", "p0", "p_min", "p_max", "p_eps", "p_delta_clip",
        "gamma", "replay_min_total", "max_assistant_trials", "replay_budget_mode",
    ],
}


def _get_method_config(config: Config) -> dict:
    """Extract method-specific config fields."""
    field_names = _METHOD_FIELDS.get(config.method, [])
    return {name: getattr(config, name) for name in field_names}


def save_results(
    results: List[RoundResult],
    config: Config,
    output_dir: str,
    started_at: str,
):
    """Save results in unified JSON format."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{config.method}_{config.dataset}_{timestamp}.json"

    payload = {
        "meta": {
            "framework": "sfl_sim",
            "method": config.method,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_rounds": config.global_round,
            "seed": config.seed,
            "schedule_dir": config.schedule_dir,
        },
        "config": {
            "dataset": config.dataset,
            "model": config.model,
            "split_layer": config.split_layer,
            "num_clients": config.num_clients,
            "clients_per_round": config.num_clients_per_round,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "optimizer": config.optimizer,
            "distribution": config.distribution,
            "dirichlet_alpha": config.dirichlet_alpha,
            "labels_per_client": config.labels_per_client,
            "selector": config.selector,
            "aggregator": config.aggregator,
            "exhaustion_policy": config.exhaustion_policy,
            "server_training_mode": config.server_training_mode,
            "scale_client_grad": config.scale_client_grad,
        },
        "method_config": _get_method_config(config),
        "rounds": [
            {
                "round": r.round_number,
                "accuracy": r.accuracy,
                "loss": r.loss,
                "time": r.metrics.get("round_time", 0.0),
                "selected_clients": r.metrics.get("selected_clients", []),
                "metrics": {
                    k: v for k, v in r.metrics.items()
                    if k not in ("round_time", "selected_clients")
                },
            }
            for r in results
        ],
        "summary": _compute_summary(results),
    }

    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {path}", flush=True)
    return path


def _compute_summary(results: List[RoundResult]) -> dict:
    """Compute summary statistics."""
    if not results:
        return {}

    accuracies = [r.accuracy for r in results]
    best_acc = max(accuracies)
    best_round = accuracies.index(best_acc) + 1

    return {
        "best_accuracy": best_acc,
        "best_round": best_round,
        "final_accuracy": accuracies[-1],
    }


def run(config: Config) -> List[RoundResult]:
    """Run a complete experiment."""
    from .trainer import SimTrainer
    from .methods import get_method_hook

    set_seed(config.seed)

    # Load pre-computed schedule first (fills config from meta.json)
    if not config.schedule_dir:
        raise ValueError(
            "Missing --schedule-dir. Generate a schedule first:\n"
            "  python -m sfl_sim.prepare -d cifar10 -nc 100 -ncpr 10 -gr 100 "
            "--selector usfl --seed 42\n"
            "Then pass the output directory via --schedule-dir"
        )
    client_data_masks, selection_schedule = _load_schedule_dir(
        config.schedule_dir, config
    )

    # Print config after schedule loads (meta.json may override rounds, dataset, etc.)
    print(f"[sfl_sim] Method: {config.method}", flush=True)
    print(f"[sfl_sim] Dataset: {config.dataset}, Model: {config.model}", flush=True)
    print(f"[sfl_sim] Rounds: {config.global_round}, Clients: {config.num_clients}", flush=True)
    print(f"[sfl_sim] Device: {config.device}", flush=True)
    print(flush=True)

    # Build trainer
    from collections import defaultdict
    trainer = SimTrainer.__new__(SimTrainer)
    trainer.config = config
    trainer.device = torch.device(config.device)
    trainer.rng = np.random.RandomState(config.seed)
    trainer._callbacks = defaultdict(list)
    trainer.client_data_masks = client_data_masks
    trainer.selection_schedule = selection_schedule
    trainer.all_client_ids = list(range(config.num_clients))

    # Load data
    from .data import load_dataset, get_testloader
    trainer.trainset, testset, trainer.num_classes = load_dataset(
        config.dataset, data_dir="./data"
    )
    trainer.testloader = get_testloader(testset, batch_size=config.batch_size)

    # Create model
    from .models import create_model
    trainer.model = create_model(
        config.model, trainer.num_classes, config.split_layer, config.dataset
    )
    trainer.model.to(trainer.device)

    # Create hook
    hook = get_method_hook(config.method, config, trainer)
    trainer.hook = hook

    return trainer.train()


def main():
    """CLI entry point."""
    config = parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    results = run(config)
    save_results(results, config, config.result_dir, started_at)


if __name__ == "__main__":
    main()
