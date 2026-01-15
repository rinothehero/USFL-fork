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
    dataset: str,
    model: str,
    batch_size: int,
    labels_per_client: int,
    dirichlet_alpha: float,
    split_layer: str,
) -> Dict[str, str]:
    return {
        "GAS_DATASET": dataset,
        "GAS_MODEL": model,
        "GAS_BATCH_SIZE": str(batch_size),
        "GAS_LABELS_PER_CLIENT": str(labels_per_client),
        "GAS_DIRICHLET_ALPHA": str(dirichlet_alpha),
        "GAS_SPLIT_LAYER": split_layer,
        "GAS_TOTAL_CLIENTS": "100",
        "GAS_CLIENTS_PER_ROUND": "10",
        "GAS_GLOBAL_EPOCHS": "300",
        "GAS_LOCAL_EPOCHS": "5",
        "GAS_LR": "0.001",
        "GAS_MOMENTUM": "0.0",
        "GAS_MIN_REQUIRE_SIZE": "10",
    }


def multisfl_command(
    *,
    dataset: str,
    model_type: str,
    batch_size: int,
    split_layer: str,
    labels_per_client: int,
    dirichlet_alpha: float,
) -> List[str]:
    return [
        "python",
        "multisfl_implementation/run_multisfl.py",
        "--dataset",
        dataset,
        "--partition",
        "shard_dirichlet",
        "--shards",
        str(labels_per_client),
        "--alpha_dirichlet",
        str(dirichlet_alpha),
        "--batch_size",
        str(batch_size),
        "--rounds",
        "300",
        "--num_clients",
        "100",
        "--n_main",
        "10",
        "--local_steps",
        "5",
        "--lr_client",
        "0.001",
        "--lr_server",
        "0.001",
        "--momentum",
        "0.0",
        "--model_type",
        model_type,
        "--split_layer",
        split_layer,
        "--min_samples_per_client",
        "10",
        "--enable_g_measurement",
        "true",
        "--g_measure_frequency",
        "10",
        "--use_variance_g",
        "true",
        "--oracle_mode",
        "branch",
        "--seed",
        "42",
        "--device",
        "cuda",
    ]


def main() -> None:
    experiments: List[Experiment] = [
        Experiment(
            name="GAS-A (cifar10, resnet18, layer1.1.bn2)",
            kind="GAS",
            env=gas_env(
                dataset="cifar10",
                model="resnet18",
                batch_size=50,
                labels_per_client=2,
                dirichlet_alpha=0.3,
                split_layer="layer1.1.bn2",
            ),
            command=[
                "python",
                "GAS_implementation/GAS_main.py",
                "true",
                "--sfl-transform",
            ],
        ),
        Experiment(
            name="MultiSFL-A (cifar10, resnet18_image_style, layer1.1.bn2)",
            kind="MultiSFL",
            env={},
            command=multisfl_command(
                dataset="cifar10",
                model_type="resnet18_image_style",
                batch_size=50,
                split_layer="layer1.1.bn2",
                labels_per_client=2,
                dirichlet_alpha=0.3,
            ),
        ),
        Experiment(
            name="GAS-O (cifar10, resnet18, layer1.0.bn1)",
            kind="GAS",
            env=gas_env(
                dataset="cifar10",
                model="resnet18",
                batch_size=50,
                labels_per_client=2,
                dirichlet_alpha=0.3,
                split_layer="layer1.0.bn1",
            ),
            command=[
                "python",
                "GAS_implementation/GAS_main.py",
                "true",
                "--sfl-transform",
            ],
        ),
        Experiment(
            name="MultiSFL-O (cifar10, resnet18_image_style, layer1.0.bn1)",
            kind="MultiSFL",
            env={},
            command=multisfl_command(
                dataset="cifar10",
                model_type="resnet18_image_style",
                batch_size=50,
                split_layer="layer1.0.bn1",
                labels_per_client=2,
                dirichlet_alpha=0.3,
            ),
        ),
        Experiment(
            name="GAS-B (fmnist, resnet18, layer2.1.bn2)",
            kind="GAS",
            env=gas_env(
                dataset="cifar10",
                model="resnet18",
                batch_size=50,
                labels_per_client=2,
                dirichlet_alpha=0.3,
                split_layer="layer2.1.bn2",
            ),
            command=[
                "python",
                "GAS_implementation/GAS_main.py",
                "true",
                "--sfl-transform",
            ],
        ),
    
        Experiment(
            name="MultiSFL-B (cifar10, resnet18_image_style, layer2.1.bn2)",
            kind="MultiSFL",
            env={},
            command=multisfl_command(
                dataset="cifar10",
                model_type="resnet18_image_style",
                batch_size=50,
                split_layer="layer2.1.bn2",
                labels_per_client=2,
                dirichlet_alpha=0.3,
            ),
        ),
    ]

    for exp in experiments:
        run_experiment(exp)


if __name__ == "__main__":
    main()
