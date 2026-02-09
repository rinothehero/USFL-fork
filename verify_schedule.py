"""
Verify that generate_schedule.py produces a distribution identical to the
actual SFL framework's ShardDirichletDistributer.

Run on a machine with torchvision + the SFL framework available:
    python verify_schedule.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    seed = 42
    alpha = 0.3
    labels_per_client = 2
    num_clients = 100
    batch_size = 50
    schedule_path = "experiment_configs/client_schedule.json"

    # Read min_require_size from common.json (same as generate_spec.py default)
    common_path = Path("experiment_configs/common.json")
    if common_path.exists():
        with open(common_path) as f:
            common_cfg = json.load(f)
        min_require_size = common_cfg.get("min_require_size", 10)
    else:
        min_require_size = 10
    print(f"[0] min_require_size={min_require_size} (must match framework)")

    # ── 1. Load real CIFAR-10 targets ──
    from generate_schedule import _distribute_once
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root="./data", train=True, download=True)
    targets = np.array(ds.targets)
    num_classes = len(set(targets.tolist()))
    print(f"[1] CIFAR-10 loaded: {len(targets)} samples, {num_classes} classes")

    for retry in range(10000):
        prng = np.random.default_rng(seed + retry)
        idx_gen, classes_gen = _distribute_once(
            targets, num_clients, num_classes, labels_per_client, alpha, prng
        )
        min_size = min(len(v) for v in idx_gen)
        if min_size >= min_require_size:
            print(f"[2] generate_schedule distribution: success on retry {retry}, min_size={min_size}")
            break

    # ── 3. Run actual framework's distributer ──
    sys.path.insert(0, "sfl_framework-fork-feature-training-tracker/server")
    from modules.trainer.distributer.shard_dirichlet_distributer import ShardDirichletDistributer

    class FakeConfig:
        def __init__(self):
            self.labels_per_client = labels_per_client
            self.dirichlet_alpha = alpha
            self.seed = seed
            self.min_require_size = min_require_size
            self.delete_fraction_of_data = False
            self.batch_size = batch_size

    distributer = ShardDirichletDistributer(FakeConfig())
    clients_list = [str(i) for i in range(num_clients)]
    idx_fw = distributer.distribute(ds, clients_list)
    print(f"[3] Framework distribution: {len(idx_fw)} clients")

    # ── 4. Compare distributions ──
    match = True
    for cid in range(num_clients):
        gen_set = sorted(idx_gen[cid])
        fw_set = sorted(idx_fw[cid])
        if gen_set != fw_set:
            match = False
            gen_labels = Counter(int(targets[i]) for i in idx_gen[cid])
            fw_labels = Counter(int(targets[i]) for i in idx_fw[cid])
            print(f"  [MISMATCH] Client {cid}: "
                  f"gen={len(gen_set)} samples {dict(gen_labels)}, "
                  f"fw={len(fw_set)} samples {dict(fw_labels)}")
            if cid >= 5:
                print(f"  ... (stopping after 5 mismatches)")
                break

    if match:
        print(f"[4] PASS: All {num_clients} clients have identical distributions")
    else:
        print(f"[4] FAIL: Distributions differ")
        return

    # ── 5. Verify schedule class coverage ──
    with open(schedule_path) as f:
        schedule = json.load(f)

    # Build per-client label sets from the verified distribution
    client_labels = {}
    for cid in range(num_clients):
        client_labels[cid] = set(int(targets[i]) for i in idx_gen[cid])

    all_covered = True
    for r, clients in enumerate(schedule):
        round_labels = set()
        for cid in clients:
            round_labels.update(client_labels[cid])
        if len(round_labels) < num_classes:
            missing = set(range(num_classes)) - round_labels
            print(f"  [FAIL] Round {r+1}: missing classes {missing}, clients={clients}")
            all_covered = False

    if all_covered:
        print(f"[5] PASS: All {len(schedule)} rounds cover all {num_classes} classes")
    else:
        print(f"[5] FAIL: Some rounds have missing classes")

    print("\n=== Verification complete ===")


if __name__ == "__main__":
    main()
