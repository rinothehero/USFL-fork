"""
CLI entry point and result saving.

Usage:
    python -m sfl_sim -d cifar10 -m resnet18 -M sfl -le 5 -gr 100 ...
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
            "optimizer": config.optimizer,
            "distribution": config.distribution,
            "dirichlet_alpha": config.dirichlet_alpha,
            "labels_per_client": config.labels_per_client,
            "selector": config.selector,
            "aggregator": config.aggregator,
        },
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

    print(f"[sfl_sim] Method: {config.method}", flush=True)
    print(f"[sfl_sim] Dataset: {config.dataset}, Model: {config.model}", flush=True)
    print(f"[sfl_sim] Rounds: {config.global_round}, Clients: {config.num_clients}", flush=True)
    print(f"[sfl_sim] Device: {config.device}", flush=True)
    print(flush=True)

    # Build trainer first, then hook (hook needs trainer reference)
    trainer = SimTrainer.__new__(SimTrainer)
    trainer.config = config
    trainer.device = torch.device(config.device)
    trainer.rng = np.random.RandomState(config.seed)

    # Load data
    from .data import load_dataset, distribute, get_testloader
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

    # Distribute data
    trainer.client_data_masks = distribute(
        trainer.trainset,
        config.num_clients,
        config.distribution,
        alpha=config.dirichlet_alpha,
        labels_per_client=config.labels_per_client,
        min_require_size=config.min_require_size,
        seed=config.seed,
    )
    trainer.all_client_ids = list(range(config.num_clients))

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
