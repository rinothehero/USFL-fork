import asyncio
from typing import TYPE_CHECKING

from server_args import parse_args

from server import Server

if TYPE_CHECKING:
    from server_args import Config

import torch


def _validate_selector(config: "Config"):
    if config.selector == "missing_class":
        if config.num_missing_class is None:
            raise ValueError("num_missing_class is required for missing_class selector")

        if config.distributer not in ["label", "label_dirichlet"]:
            raise ValueError(
                "distributer must be 'label' or 'label_dirichlet' for missing_class selector"
            )


def _validate_workload(config: "Config"):
    nlp_datasets = [
        "cola",
        "sst2",
        "mrpc",
        "sts-b",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "wnli",
        "ax",
    ]
    vision_datasets = ["cifar10", "cifar100", "mnist", "fmnist"]

    if config.dataset in nlp_datasets:
        return config.model == "distilbert"
    elif config.dataset in vision_datasets:
        if config.model == "lenet" and config.dataset not in ["mnist", "fmnist"]:
            return False

        return config.model in [
            "vgg11",
            "resnet18",
            "tiny_vgg11",
            "alexnet",
            "alexnet_scala",
            "mobilenet",
            "lenet",
        ]

    return False


def _validate_device(config: "Config"):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is not available on this system")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device is not available on this system")


def _validate_distributer(config: "Config"):
    if config.distributer == "dirichlet":
        if config.dirichlet_alpha is None:
            raise ValueError("Dirichlet alpha is required for dirichlet distributer")

    if config.distributer == "label":
        if config.labels_per_client is None:
            raise ValueError(
                "labels_per_client alpha is required for label distributer"
            )


def _validate_glue_parameters(config: "Config"):
    nlp_datasets = [
        "cola",
        "sst2",
        "mrpc",
        "sts-b",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "wnli",
        "ax",
    ]

    if config.dataset in nlp_datasets:
        if config.glue_tokenizer is None:
            raise ValueError("Tokenizer is required for GLUE datasets")
        if config.glue_max_seq_length is None:
            raise ValueError("Max sequence length is required for GLUE datasets")


def _validate_sfl_parameters(config: "Config"):
    if config.method in ["sfl", "sfl-u"]:
        if config.split_strategy is None:
            raise ValueError(
                "Split strategy is required when using the 'sfl' or 'sfl-u' method"
            )

        if config.method == "sfl" and len(config.split_ratio) != 1:
            raise ValueError(
                f"The 'sfl' method requires a split_ratio with exactly 1 value (have {len(config.split_ratio)})"
            )
        elif config.method == "sfl-u" and len(config.split_ratio) != 2:
            raise ValueError(
                f"The 'sfl-u' method requires a split_ratio with exactly 2 values (have {len(config.split_ratio)})"
            )


def _validate_fitfl_parameters(config: "Config"):
    if config.method == "fitfl":
        if config.max_pruning_ratio is None:
            raise ValueError("Maximum pruning ratio is required for fitfl method")
        if config.min_pruning_ratio is None:
            raise ValueError("Minimum pruning ratio is required for fitfl method")
        if config.pruning_decision_threshold is None:
            raise ValueError("Pruning decision threshold is required for fitfl method")
        if config.pruning_interval is None:
            raise ValueError("Pruning interval is required for fitfl method")
        if config.min_pruning_accuracy is None:
            raise ValueError("Minimum pruning accuracy is required for fitfl method")
        if config.ema_threshold is None:
            raise ValueError("EMA threshold is required for fitfl method")

        if config.dataset not in ["cifar10", "fmnist", "mnist"]:
            raise ValueError(
                "Dataset must be either 'cifar10' or 'fmnist' or 'mnist' for fitfl method"
            )

        if config.model not in [
            "tiny_vgg11",
            "resnet18",
            "alexnet",
        ]:
            raise ValueError(
                "Model must be either 'tiny_vgg11' or 'resnet18' or 'alexnet' for fitfl method"
            )


def _validate_prunefl_parameters(config: "Config"):
    if config.method == "prunefl":
        if config.initial_pruning_ratio is None:
            raise ValueError("Pruning ratio is required for prunefl method")

        if config.initial_pruning_device_id is None:
            raise ValueError("Initial pruning device id is required for prunefl method")

        if config.initial_pruning_epoch is None:
            raise ValueError("Initial pruning epoch is required for prunefl method")

        if config.dataset not in ["cifar10", "fmnist", "mnist"]:
            raise ValueError(
                "Dataset must be either 'cifar10' or 'fmnist' or 'mnist' for prunefl method"
            )

        if config.model not in [
            "tiny_vgg11",
            "resnet18",
            "alexnet",
        ]:
            raise ValueError(
                "Model must be either 'tiny_vgg11' or 'resnet18' or 'alexnet' for prunefl method"
            )


def _validate_fedsparsify_parameters(config: "Config"):
    if config.method == "fedsparsify":
        if config.target_pruning_ratio is None:
            raise ValueError("Target pruning ratio is required for fedsparsify method")

        if config.pruning_interval is None:
            raise ValueError("Pruning interval is required for fedsparsify method")

        if config.pruning_control_variable is None:
            raise ValueError(
                "Pruning control variable is required for fedsparsify method"
            )

        if config.dataset not in ["cifar10", "fmnist", "mnist"]:
            raise ValueError(
                "Dataset must be either 'cifar10' or 'fmnist' or 'mnist' for fedsparsify method"
            )

        if config.model not in [
            "tiny_vgg11",
            "resnet18",
            "alexnet",
        ]:
            raise ValueError(
                "Model must be either 'tiny_vgg11' or 'resnet18' or 'alexnet' for fedsparsify method"
            )


def _validate_nestfl_parameters(config: "Config"):
    if config.method == "nestfl":
        if config.model not in ["tiny_vgg11", "resnet18", "alexnet"]:
            raise ValueError(
                "Model must be either 'tiny_vgg11' or 'resnet18' or 'alexnet' for nestfl method"
            )

        if config.dataset not in ["cifar10", "fmnist", "mnist"]:
            raise ValueError(
                "Dataset must be either 'cifar10' or 'fmnist' or 'mnist' for nestfl method"
            )


def _validate_scala_parameters(config: "Config"):
    if config.method == "scala":
        if config.criterion not in ["ce"]:
            raise ValueError("Criterion must be 'ce' for scala method")

        if config.split_strategy is None:
            raise ValueError("Split strategy is required when using the 'scala' method")

        if len(config.split_ratio) != 1:
            raise ValueError(
                f"The 'scala' method requires a split_ratio with exactly 1 value (have {len(config.split_ratio)})"
            )


def _validate_scala_parameters(config: "Config"):
    if config.method == "scala":
        if config.criterion not in ["ce"]:
            raise ValueError("Criterion must be 'ce' for scala method")


def _validate_usfl_freshness_parameters(config: "Config"):
    """Validate USFL freshness scoring configuration dependencies"""
    if config.use_fresh_scoring:
        # Fresh scoring requires cumulative usage tracking
        if not config.use_cumulative_usage:
            raise ValueError(
                "Configuration Error: 'use_fresh_scoring' requires 'use_cumulative_usage' to be enabled.\n"
                "Please add '-ucu' flag or set use_cumulative_usage=True.\n"
                "Freshness scoring needs historical usage data to function correctly."
            )

        # Validate freshness_decay_rate range
        if hasattr(config, 'freshness_decay_rate') and config.freshness_decay_rate is not None:
            if not (0.0 <= config.freshness_decay_rate <= 1.0):
                raise ValueError(
                    f"Configuration Error: 'freshness_decay_rate' must be between 0.0 and 1.0, "
                    f"got {config.freshness_decay_rate}"
                )

        print(f"[INFO] Freshness scoring enabled with decay rate: "
              f"{getattr(config, 'freshness_decay_rate', 0.5)}")


def _validate_usfl_data_replication(config: "Config"):
    """Validate USFL data replication configuration"""
    use_replication = getattr(config, 'use_data_replication', False)

    if use_replication:
        print(f"[INFO] Data replication enabled: using max-based augmentation instead of min-based trimming")
        print(f"[INFO] Clients will over-sample their data to balance class distribution")


def validate_config(config: "Config"):
    if not _validate_workload(config):
        raise ValueError(f"Workload is not valid ({config.model}-{config.dataset})")

    _validate_selector(config)
    _validate_device(config)
    _validate_glue_parameters(config)
    _validate_distributer(config)

    _validate_sfl_parameters(config)
    _validate_fitfl_parameters(config)
    _validate_prunefl_parameters(config)
    _validate_fedsparsify_parameters(config)
    _validate_nestfl_parameters(config)
    _validate_scala_parameters(config)
    _validate_usfl_freshness_parameters(config)
    _validate_usfl_data_replication(config)


def main(config: "Config"):
    validate_config(config)
    server_instance = Server(config)
    asyncio.run(server_instance.run())


def simulation_main(config: "Config"):
    validate_config(config)
    server_instance = Server(config)
    asyncio.run(server_instance.run())


if __name__ == "__main__":
    config = parse_args()
    main(config)
