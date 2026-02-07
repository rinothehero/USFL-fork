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
from multisfl.models import get_split_models, load_torchvision_resnet18_init
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


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


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
    parser.add_argument(
        "--min_samples_per_client",
        type=int,
        default=10,
        help="Minimum samples per client for partitioning",
    )
    parser.add_argument(
        "--use-full-epochs",
        action="store_true",
        help="Use full epochs (dataset iteration) instead of fixed steps for local training",
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
    parser.add_argument("--eps", type=float, default=1e-12)
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
    parser.add_argument("--result_output_dir", type=str, default="results",
                        help="Directory for result JSON files")
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
        "--g_measurement_mode",
        type=str,
        default="single",
        choices=["single", "k_batch", "accumulated"],
        help="G measurement mode: 'single' (1-step), 'k_batch' (first K batches), or 'accumulated' (full round average)",
    )
    parser.add_argument(
        "--g_measurement_k",
        type=int,
        default=5,
        help="Number of batches for k_batch mode (default: 5)",
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
    parser.add_argument(
        "--use_torchvision_init",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use torchvision ResNet18 initialization (for image_style models)",
    )
    parser.add_argument(
        "--oracle_mode",
        type=str,
        default="master",
        choices=["master", "branch"],
        help="Oracle computation mode (master or branch average)",
    )

    parser.add_argument(
        "--clip_grad",
        type=str_to_bool,
        default=False,
        help="Enable gradient clipping for client/server updates",
    )
    parser.add_argument(
        "--clip_grad_max_norm",
        type=float,
        default=10.0,
        help="Max norm for gradient clipping",
    )

    # Drift Measurement
    parser.add_argument(
        "--enable_drift_measurement",
        type=str_to_bool,
        default=False,
        help="Enable SCAFFOLD-style drift measurement",
    )
    parser.add_argument(
        "--drift_sample_interval",
        type=int,
        default=1,
        help="Sample drift every N steps (1 = every step)",
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

    num_classes = args.num_classes
    device = args.device

    cfg = MultiSFLConfig(
        num_rounds=args.rounds,
        num_clients_total=args.num_clients,
        n_main_clients_per_round=args.n_main,
        num_branches=args.branches,
        dataset=args.dataset,
        num_classes=num_classes,
        device=device,
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
        eps=args.eps,
        replay_budget_mode=args.replay_budget_mode,
        replay_min_total=args.replay_min_total,
        max_assistant_trials_per_branch=args.max_assistant_trials,
        p_update=args.p_update,
        delta_clip=args.delta_clip,
        model_type=args.model_type,
        split_layer=args.split_layer,
        enable_g_measurement=str_to_bool(args.enable_g_measurement),
        g_measure_frequency=args.g_measure_frequency,
        g_measurement_mode=args.g_measurement_mode,
        g_measurement_k=args.g_measurement_k,
        use_variance_g=str_to_bool(args.use_variance_g),
        use_sfl_transform=str_to_bool(args.use_sfl_transform),
        use_torchvision_init=str_to_bool(args.use_torchvision_init),
        oracle_mode=args.oracle_mode,
        clip_grad=args.clip_grad,
        clip_grad_max_norm=args.clip_grad_max_norm,
        min_samples_per_client=args.min_samples_per_client,
        use_full_epochs=args.use_full_epochs,
        enable_drift_measurement=args.enable_drift_measurement,
        drift_sample_interval=args.drift_sample_interval,
    )

    # Data Partitioning
    if args.dataset == "synthetic":
        train_data = SyntheticImageDataset(
            args.synthetic_train_size, args.num_classes, seed=args.seed
        )
        test_data = SyntheticImageDataset(
            args.synthetic_test_size, args.num_classes, seed=args.seed + 1
        )
        test_loader = get_synthetic_test_loader(test_data, args.batch_size)
    elif args.dataset == "cifar10":
        train_data = CIFAR10Dataset(
            root=args.data_root,
            train=True,
            augment=True,
            use_sfl_transform=args.use_sfl_transform,
        )
        test_loader = get_cifar10_test_loader(
            root=args.data_root,
            batch_size=args.batch_size,
            use_sfl_transform=args.use_sfl_transform,
        )
    elif args.dataset == "fmnist":
        train_data = FashionMNISTDataset(root=args.data_root, train=True, augment=True)
        test_loader = get_fmnist_test_loader(
            root=args.data_root, batch_size=args.batch_size
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Partitioning data: {args.partition}")
    if args.partition == "iid":
        client_data_list = partition_iid(
            train_data, args.num_clients, args.num_classes, seed=args.seed
        )
    elif args.partition == "dirichlet":
        client_data_list = partition_dirichlet(
            train_data,
            args.num_clients,
            args.num_classes,
            alpha=args.alpha_dirichlet,
            seed=args.seed,
            min_samples_per_client=args.min_samples_per_client,
        )
    elif args.partition == "shard_dirichlet":
        client_data_list = partition_shard_dirichlet(
            train_data,
            args.num_clients,
            args.num_classes,
            alpha=args.alpha_dirichlet,
            shards=args.shards,
            seed=args.seed,
            min_samples_per_client=args.min_samples_per_client,
        )
    else:
        raise ValueError(f"Unknown partition method: {args.partition}")

    print_partition_stats(client_data_list)

    # Create Client instances
    clients = []
    for cid, client_data in enumerate(client_data_list):
        client = Client(
            client_id=cid,
            dataset=client_data.dataset,
            num_classes=num_classes,
            class_to_indices=client_data.class_to_indices,
            device=device,
        )
        clients.append(client)

    # Initialize Replay Components
    num_branches = cfg.num_branches or cfg.n_main_clients_per_round
    score_tracker = ScoreVectorTracker(
        num_branches=num_branches,
        num_classes=cfg.num_classes,
        gamma=cfg.gamma,
    )

    planner = KnowledgeRequestPlanner(
        num_classes=cfg.num_classes,
    )

    scheduler = SamplingProportionScheduler(
        p0=cfg.p0,
        p_min=cfg.p_min,
        p_max=cfg.p_max,
        mode=cfg.p_update,
        delta_clip=cfg.delta_clip,
        eps=cfg.eps,
    )

    # Initialize Models & Servers
    wc_init, ws_init = get_split_models(
        cfg.model_type, args.dataset, cfg.num_classes, cfg.split_layer
    )

    if cfg.use_torchvision_init and cfg.model_type == "resnet18_image_style":
        print("[Init] Loading torchvision ResNet18 weights...")
        wc_init, ws_init = load_torchvision_resnet18_init(
            wc_init, ws_init, split_layer=cfg.split_layer or "layer1", image_style=True
        )

    branch_client_states = []
    branch_server_states = []

    for b in range(num_branches):
        wc = copy.deepcopy(wc_init).to(device)
        ws = copy.deepcopy(ws_init).to(device)

        opt_c = optim.SGD(wc.parameters(), lr=cfg.lr_client, momentum=cfg.momentum)
        opt_s = optim.SGD(ws.parameters(), lr=cfg.lr_server, momentum=cfg.momentum)

        branch_client_states.append(BranchClientState(wc, opt_c))
        branch_server_states.append(BranchServerState(ws, opt_s))

    fed = FedServer(branch_client_states, cfg.alpha_master_pull, device=device)
    main_server = MainServer(
        branch_server_states,
        cfg.alpha_master_pull,
        device=device,
        clip_grad=cfg.clip_grad,
        clip_grad_max_norm=cfg.clip_grad_max_norm,
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

    g_summary = trainer.g_system.get_summary() if trainer.g_system else {}
    g_measurements = trainer.g_system.get_all_measurements() if trainer.g_system else []

    # Save results to JSON
    results = {
        "config": vars(args),
        "timestamp": datetime.now().isoformat(),
        "rounds": [],
        "g_measurements": g_measurements,
    }
    g_map = {}
    if g_summary:
        for i, r_idx in enumerate(g_summary["rounds"]):
            g_map[r_idx] = {
                "client_g": g_summary["client_g"][i],
                "client_g_rel": g_summary["client_g_rel"][i],
                "server_g": g_summary["server_g"][i],
                "server_g_rel": g_summary["server_g_rel"][i],
                "variance_client_g": g_summary["variance_client_g"][i],
                "variance_client_g_rel": g_summary["variance_client_g_rel"][i],
                "variance_server_g": g_summary["variance_server_g"][i],
                "variance_server_g_rel": g_summary["variance_server_g_rel"][i],
            }

    g_detail_map = {}
    for measurement in g_measurements:
        g_detail_map[measurement["round"]] = measurement

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
        if g_idx in g_detail_map:
            round_data["per_client_g"] = g_detail_map[g_idx].get("per_client_g", {})
            round_data["per_branch_server_g"] = g_detail_map[g_idx].get(
                "per_branch_server_g", {}
            )

        results["rounds"].append(round_data)

    results["summary"] = {
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "total_requested": total_requested,
        "total_collected": total_collected,
    }

    # Add drift measurement results
    if trainer.drift_tracker is not None:
        results["drift_history"] = trainer.drift_tracker.get_history()
        print(f"\n[Drift Measurement] Final G_drift values: {results['drift_history']['G_drift'][-5:]}")

    os.makedirs(args.result_output_dir, exist_ok=True)
    filename = os.path.join(args.result_output_dir, f"results_multisfl_{args.dataset}_{args.partition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
