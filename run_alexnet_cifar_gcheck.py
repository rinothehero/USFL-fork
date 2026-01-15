#!/usr/bin/env python3
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Experiment:
    name: str
    kind: str
    env: Dict[str, str]
    command: List[str]


def run_experiment(exp: Experiment) -> None:
    env = os.environ.copy()
    env.update(exp.env)
    print("=" * 70)
    print(f"Running {exp.kind}: {exp.name}")
    print("Command:", " ".join(exp.command))
    if exp.env:
        print("Env:")
        for key, value in exp.env.items():
            print(f"  {key}={value}")
    print("=" * 70)
    subprocess.run(exp.command, check=True, env=env)


def gas_env(
    *,
    batch_size: int,
    labels_per_client: int,
    dirichlet_alpha: float,
    split_layer: str,
    total_clients: int,
    clients_per_round: int,
    global_epochs: int,
    local_epochs: int,
    min_require_size: int,
) -> Dict[str, str]:
    return {
        "GAS_DATASET": "cifar10",
        "GAS_MODEL": "alexnet",
        "GAS_BATCH_SIZE": str(batch_size),
        "GAS_LABELS_PER_CLIENT": str(labels_per_client),
        "GAS_DIRICHLET_ALPHA": str(dirichlet_alpha),
        "GAS_SPLIT_LAYER": split_layer,
        "GAS_TOTAL_CLIENTS": str(total_clients),
        "GAS_CLIENTS_PER_ROUND": str(clients_per_round),
        "GAS_GLOBAL_EPOCHS": str(global_epochs),
        "GAS_LOCAL_EPOCHS": str(local_epochs),
        "GAS_MIN_REQUIRE_SIZE": str(min_require_size),
        "GAS_LR": "0.001",
        "GAS_MOMENTUM": "0.0",
    }


def multisfl_command(
    *,
    batch_size: int,
    split_layer: str,
    labels_per_client: int,
    dirichlet_alpha: float,
    total_clients: int,
    clients_per_round: int,
    rounds: int,
    local_steps: int,
    min_samples: int,
    device: str,
) -> List[str]:
    return [
        "python",
        "multisfl_implementation/run_multisfl.py",
        "--dataset",
        "cifar10",
        "--partition",
        "shard_dirichlet",
        "--shards",
        str(labels_per_client),
        "--alpha_dirichlet",
        str(dirichlet_alpha),
        "--batch_size",
        str(batch_size),
        "--rounds",
        str(rounds),
        "--num_clients",
        str(total_clients),
        "--n_main",
        str(clients_per_round),
        "--local_steps",
        str(local_steps),
        "--lr_client",
        "0.001",
        "--lr_server",
        "0.001",
        "--momentum",
        "0.0",
        "--model_type",
        "alexnet",
        "--split_layer",
        split_layer,
        "--min_samples_per_client",
        str(min_samples),
        "--enable_g_measurement",
        "true",
        "--g_measure_frequency",
        "1",
        "--use_variance_g",
        "false",
        "--oracle_mode",
        "master",
        "--seed",
        "42",
        "--device",
        device,
    ]


def usfl_command(
    *,
    split_layer: str,
    batch_size: int,
    labels_per_client: int,
    dirichlet_alpha: float,
    total_clients: int,
    clients_per_round: int,
    global_rounds: int,
    local_epochs: int,
    min_require_size: int,
    port: int,
    device: str,
) -> List[str]:
    return [
        "python",
        "-c",
        (
            "import sys,asyncio; "
            "sys.path.insert(0,'/home/xsailor6/jhkang/usfl/sfl_framework-fork-feature-training-tracker'); "
            "import simulation as sim; "
            f"port='{port}'; "
            "server_args=["
            f"'-H','0.0.0.0','-p',port,"
            f"'-d','cifar10','-m','alexnet','-M','usfl',"
            f"'-le','{local_epochs}','-gr','{global_rounds}',"
            f"'-de','{device}','-nc','{total_clients}','-ncpr','{clients_per_round}',"
            f"'-distr','shard_dirichlet','-lpc','{labels_per_client}',"
            f"'-diri-alpha','{dirichlet_alpha}','-bs','{batch_size}',"
            f"'-ss','layer_name','-sl','{split_layer}',"
            "'-o','sgd','-lr','0.001','-mt','0.0','-s','usfl','-aggr','fedavg',"
            f"'-mrs','{min_require_size}','--enable-g-measurement','--diagnostic-rounds','1'"
            "];"
            f"client_args=[[ '-cid',str(i),'-d','{device}', '-su',f'localhost:{port}'] for i in range({total_clients})];"
            "asyncio.run(sim.run_simulation(server_args, client_args))"
        ),
    ]


def main() -> None:
    total_clients = 10
    clients_per_round = 5
    global_rounds = 2
    local_epochs = 1
    local_steps = 1
    min_require_size = 10
    labels_per_client = 8
    dirichlet_alpha = 0.9

    gas_batch_size = 50
    multisfl_batch_size = 50
    usfl_batch_size = 500

    device = "cpu"

    experiments: List[Experiment] = [
        Experiment(
            name="USFL AlexNet CIFAR (conv.2)",
            kind="USFL",
            env={},
            command=usfl_command(
                split_layer="conv.2",
                batch_size=usfl_batch_size,
                labels_per_client=labels_per_client,
                dirichlet_alpha=dirichlet_alpha,
                total_clients=total_clients,
                clients_per_round=clients_per_round,
                global_rounds=global_rounds,
                local_epochs=local_epochs,
                min_require_size=min_require_size,
                port=3000,
                device=device,
            ),
        ),
        Experiment(
            name="USFL AlexNet CIFAR (conv.5)",
            kind="USFL",
            env={},
            command=usfl_command(
                split_layer="conv.5",
                batch_size=usfl_batch_size,
                labels_per_client=labels_per_client,
                dirichlet_alpha=dirichlet_alpha,
                total_clients=total_clients,
                clients_per_round=clients_per_round,
                global_rounds=global_rounds,
                local_epochs=local_epochs,
                min_require_size=min_require_size,
                port=3001,
                device=device,
            ),
        ),
        Experiment(
            name="GAS AlexNet CIFAR (conv.2)",
            kind="GAS",
            env=gas_env(
                batch_size=gas_batch_size,
                labels_per_client=labels_per_client,
                dirichlet_alpha=dirichlet_alpha,
                split_layer="conv.2",
                total_clients=total_clients,
                clients_per_round=clients_per_round,
                global_epochs=global_rounds,
                local_epochs=local_epochs,
                min_require_size=min_require_size,
            ),
            command=[
                "python",
                "GAS_implementation/GAS_main.py",
                "true",
                "--sfl-transform",
            ],
        ),
        Experiment(
            name="GAS AlexNet CIFAR (conv.5)",
            kind="GAS",
            env=gas_env(
                batch_size=gas_batch_size,
                labels_per_client=labels_per_client,
                dirichlet_alpha=dirichlet_alpha,
                split_layer="conv.5",
                total_clients=total_clients,
                clients_per_round=clients_per_round,
                global_epochs=global_rounds,
                local_epochs=local_epochs,
                min_require_size=min_require_size,
            ),
            command=[
                "python",
                "GAS_implementation/GAS_main.py",
                "true",
                "--sfl-transform",
            ],
        ),
        Experiment(
            name="MultiSFL AlexNet CIFAR (conv.2)",
            kind="MultiSFL",
            env={},
            command=multisfl_command(
                batch_size=multisfl_batch_size,
                split_layer="conv.2",
                labels_per_client=labels_per_client,
                dirichlet_alpha=dirichlet_alpha,
                total_clients=total_clients,
                clients_per_round=clients_per_round,
                rounds=global_rounds,
                local_steps=local_steps,
                min_samples=min_require_size,
                device=device,
            ),
        ),
        Experiment(
            name="MultiSFL AlexNet CIFAR (conv.5)",
            kind="MultiSFL",
            env={},
            command=multisfl_command(
                batch_size=multisfl_batch_size,
                split_layer="conv.5",
                labels_per_client=labels_per_client,
                dirichlet_alpha=dirichlet_alpha,
                total_clients=total_clients,
                clients_per_round=clients_per_round,
                rounds=global_rounds,
                local_steps=local_steps,
                min_samples=min_require_size,
                device=device,
            ),
        ),
    ]

    for exp in experiments:
        run_experiment(exp)


if __name__ == "__main__":
    main()
