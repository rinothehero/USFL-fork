import argparse
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
from datetime import datetime

from multisfl.config import MultiSFLConfig
from multisfl.models import get_split_models
from multisfl.data import (
    SyntheticImageDataset,
    CIFAR10Dataset,
    FashionMNISTDataset,
    partition_iid,
    partition_dirichlet,
    partition_shard_dirichlet,
    print_partition_stats,
    get_cifar10_test_loader,
    get_fmnist_test_loader,
    get_synthetic_test_loader,
)
from multisfl.client import Client
from multisfl.servers import FedServer, MainServer, BranchClientState, BranchServerState
from multisfl.replay import ScoreVectorTracker, KnowledgeRequestPlanner
from multisfl.scheduler import SamplingProportionScheduler
from multisfl.trainer import MultiSFLTrainer
from multisfl.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MultiSFL: Multi-model Split Federated Learning"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "cifar10", "fmnist"],
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "shard_dirichlet"],
    )
    parser.add_argument("--alpha_dirichlet", type=float, default=0.3)
    parser.add_argument(
        "--shards",
        type=int,
        default=2,
        help="Number of classes per client for shard_dirichlet",
    )
    parser.add_argument("--data_root", type=str, default="./data")

    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--n_main", type=int, default=10)
    parser.add_argument("--branches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--local_steps", type=int, default=5)

    parser.add_argument("--lr_client", type=float, default=0.01)
    parser.add_argument("--lr_server", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--alpha_master_pull", type=float, default=0.1)

    parser.add_argument("--p0", type=float, default=0.01)
    parser.add_argument("--p_min", type=float, default=0.01)
    parser.add_argument("--p_max", type=float, default=0.5)
    parser.add_argument(
        "--p_update",
        type=str,
        default="abs_ratio",
        choices=["paper", "abs_ratio", "one_plus_delta"],
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta_clip", type=float, default=0.2)

    parser.add_argument(
        "--replay_budget_mode",
        type=str,
        default="local_dataset",
        choices=["batch", "local_dataset"],
    )
    parser.add_argument("--replay_min_total", type=int, default=0)
    parser.add_argument("--max_assistant_trials", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--synthetic_train_size", type=int, default=5000)
    parser.add_argument("--synthetic_test_size", type=int, default=1000)

    parser.add_argument(
        "--model_type",
        type=str,
        default="simple",
        choices=[
            "simple",
            "alexnet",
            "alexnet_light",
            "resnet18",
            "resnet18_light",
            "resnet18_flex",
            "resnet18_image_style",
        ],
    )
    parser.add_argument("--split_layer", type=str, default=None)

    # G Measurement
    parser.add_argument(
        "--enable_g_measurement",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable G measurement",
    )
    parser.add_argument(
        "--g_measure_frequency",
        type=int,
        default=10,
        help="Frequency of G measurement (every N rounds)",
    )
    parser.add_argument(
        "--use_variance_g",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use variance-based G metrics (SFL-style)",
    )
    parser.add_argument(
        "--use_sfl_transform",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use SFL-style ToTensor-only transforms",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("MultiSFL Configuration")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 70)

    cfg = MultiSFLConfig(
        num_rounds=args.rounds,
        num_clients_total=args.num_clients,
        n_main_clients_per_round=args.n_main,
        num_branches=args.branches if args.branches else args.n_main,
        dataset=args.dataset,
        num_classes=args.num_classes,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        local_steps=args.local_steps,
        lr_client=args.lr_client,
        lr_server=args.lr_server,
        momentum=args.momentum,
        alpha_master_pull=args.alpha_master_pull,
        gamma=args.gamma,
        p0=args.p0,
        p_min=args.p_min,
        p_max=args.p_max,
        p_update=args.p_update,
        delta_clip=args.delta_clip,
        max_assistant_trials_per_branch=args.max_assistant_trials,
        replay_budget_mode=args.replay_budget_mode,
        replay_min_total=args.replay_min_total,
        model_type=args.model_type,
        split_layer=args.split_layer,
        enable_g_measurement=(args.enable_g_measurement.lower() == "true"),
        g_measure_frequency=args.g_measure_frequency,
        use_variance_g=(args.use_variance_g.lower() == "true"),
        use_sfl_transform=(args.use_sfl_transform.lower() == "true"),
    )

    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "cifar10":
        train_dataset = CIFAR10Dataset(
            root=args.data_root,
            train=True,
            augment=True,
            download=True,
            use_sfl_transform=cfg.use_sfl_transform,
        )
        test_loader = get_cifar10_test_loader(
            batch_size=128, root=args.data_root, use_sfl_transform=cfg.use_sfl_transform
        )
    elif args.dataset == "fmnist":
        train_dataset = FashionMNISTDataset(
            root=args.data_root, train=True, augment=True, download=True
        )
        test_loader = get_fmnist_test_loader(batch_size=128, root=args.data_root)
    else:
        train_dataset = SyntheticImageDataset(
            n=args.synthetic_train_size, num_classes=cfg.num_classes, seed=cfg.seed
        )
        test_loader = get_synthetic_test_loader(
            n=args.synthetic_test_size,
            num_classes=cfg.num_classes,
            batch_size=128,
            seed=cfg.seed + 1,
        )

    print(f"Partitioning data: {args.partition}")
    if args.partition == "dirichlet":
        client_datas = partition_dirichlet(
            train_dataset,
            num_clients=cfg.num_clients_total,
            num_classes=cfg.num_classes,
            alpha=args.alpha_dirichlet,
            seed=cfg.seed,
        )
    elif args.partition == "shard_dirichlet":
        client_datas = partition_shard_dirichlet(
            train_dataset,
            num_clients=cfg.num_clients_total,
            num_classes=cfg.num_classes,
            shards=args.shards,
            alpha=args.alpha_dirichlet,
            seed=cfg.seed,
        )
    else:
        client_datas = partition_iid(
            train_dataset,
            num_clients=cfg.num_clients_total,
            num_classes=cfg.num_classes,
            seed=cfg.seed,
        )

    print_partition_stats(client_datas, num_classes=cfg.num_classes)

    clients = [
        Client(
            cd.client_id,
            cd.dataset,
            cfg.num_classes,
            class_to_indices=cd.class_to_indices,
            device=cfg.device,
        )
        for cd in client_datas
    ]

    B = cfg.num_branches or cfg.n_main_clients_per_round
    print(f"\nInitializing {B} branch models ({cfg.model_type})...")

    # [CRITICAL FIX] Initialize ONE base model pair, then deepcopy for each branch.
    # If we initialize new models inside the loop, they start with DIFFERENT random weights.
    # Averaging them (FedAvg) would destroy the features.
    wc_init, ws_init = get_split_models(
        model_type=cfg.model_type,
        dataset=args.dataset,
        num_classes=cfg.num_classes,
        split_layer=cfg.split_layer,
    )
    # Ensure they are on CPU first to save GPU memory during copy if needed,
    # but typically fine to init on device if memory allows.
    # Let's move to device after copy to be safe? No, let's keep it simple.

    branch_client_states = []
    branch_server_states = []

    for _ in range(B):
        # Create deep copy of the initialized weights
        wc = copy.deepcopy(wc_init).to(cfg.device)
        ws = copy.deepcopy(ws_init).to(cfg.device)

        opt_c = optim.SGD(wc.parameters(), lr=cfg.lr_client, momentum=cfg.momentum)
        opt_s = optim.SGD(ws.parameters(), lr=cfg.lr_server, momentum=cfg.momentum)

        branch_client_states.append(BranchClientState(model=wc, optimizer=opt_c))
        branch_server_states.append(BranchServerState(model=ws, optimizer=opt_s))

    fed = FedServer(
        branch_client_states, alpha=cfg.alpha_master_pull, device=cfg.device
    )
    main_server = MainServer(
        branch_server_states, alpha=cfg.alpha_master_pull, device=cfg.device
    )

    score_tracker = ScoreVectorTracker(
        num_branches=B, num_classes=cfg.num_classes, gamma=cfg.gamma
    )
    planner = KnowledgeRequestPlanner(num_classes=cfg.num_classes)
    scheduler = SamplingProportionScheduler(
        p0=cfg.p0,
        p_min=cfg.p_min,
        p_max=cfg.p_max,
        eps=cfg.eps,
        mode=cfg.p_update,
        delta_clip=cfg.delta_clip,
        verbose=True,
    )

    trainer = MultiSFLTrainer(
        cfg=cfg,
        clients=clients,
        fed=fed,
        main=main_server,
        score_tracker=score_tracker,
        planner=planner,
        scheduler=scheduler,
        test_loader=test_loader,
    )

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    stats = trainer.run()

    print("\n" + "=" * 70)
    print("Training Complete - Summary")
    print("=" * 70)
    print(
        f"{'Round':>6} {'Acc':>8} {'p_r':>10} {'Requested':>10} {'Collected':>10} {'||grad_f||':>12}"
    )
    print("-" * 70)
    for s in stats:
        print(
            f"{s.round_idx:>6} {s.acc:>8.4f} {s.p_r:>10.6f} {s.requested:>10} {s.collected:>10} {s.mean_grad_f_main_norm:>12.6f}"
        )

    final_acc = stats[-1].acc if stats else 0.0
    best_acc = max(s.acc for s in stats) if stats else 0.0
    total_requested = sum(s.requested for s in stats)
    total_collected = sum(s.collected for s in stats)

    print("-" * 70)
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Total replay requested: {total_requested}, collected: {total_collected}")
    print("=" * 70)

    # Save results to JSON
    results = {
        "config": vars(args),
        "timestamp": datetime.now().isoformat(),
        "rounds": [],
    }

    g_summary = trainer.g_system.get_summary() if trainer.g_system else {}
    g_map = {}
    if g_summary:
        for i, r_idx in enumerate(g_summary["rounds"]):
            g_map[r_idx] = {
                "client_g": g_summary["client_g"][i],
                "client_g_rel": g_summary["client_g_rel"][i],
                "server_g": g_summary["server_g"][i],
                "server_g_rel": g_summary["server_g_rel"][i],
            }

    for s in stats:
        round_data = {
            "round": s.round_idx,
            "accuracy": s.acc,
            "p_r": s.p_r,
            "fgn_r": s.fgn_r,
            "requested": s.requested,
            "collected": s.collected,
            "trials": s.trials,
            "mean_grad_f_main_norm": s.mean_grad_f_main_norm,
            "mean_client_update_norm": s.mean_client_update_norm,
            "mean_server_update_norm": s.mean_server_update_norm,
        }

        # Match 0-based index for G map
        g_idx = s.round_idx - 1
        if g_idx in g_map:
            round_data.update(g_map[g_idx])

        results["rounds"].append(round_data)

    results["summary"] = {
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "total_requested": total_requested,
        "total_collected": total_collected,
    }

    os.makedirs("results", exist_ok=True)
    filename = f"results/results_multisfl_{args.dataset}_{args.partition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
