import argparse
import re
from dataclasses import dataclass
from typing import Optional


def validate_device(value):
    if value in ["cpu", "mps", "cuda"]:
        return value
    if re.match(r"^cuda:\d+$", value):
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid device: '{value}'. Must be 'cpu', 'mps', or 'cuda:{int}'."
    )


def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    elif value.lower() in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


@dataclass
class Config:
    # ==================
    #       SERVER
    # ==================
    host: str  # Server host address for incoming connections.
    port: int  # Port number for server connections.

    networking_fairness: bool  # Whether to use networking fairness.
    # When set to true, the server will send data to clients in a round-robin manner,
    # whereas when set to false, the server will send data to clients concurrently.

    # ==================
    #      WORKLOAD
    # ==================
    dataset: str  # Dataset to use for training (e.g., 'cifar10', 'cola').
    model: str  # Model architecture to use (e.g., 'resnet18', 'vgg11', 'tiny_vgg11', 'distilbert').

    # ==================
    #     TRAINING SETTINGS
    # ==================
    seed: int
    use_dynamic_batch_scheduler: bool
    use_fresh_scoring: bool
    freshness_decay_rate: float
    method: str
    """
    Training method:
        - 'fl'   : Federated Learning.
        - 'sfl'  : Split Federated Learning (Client -> Server training flow).
        - 'sfl-u': Split Federated Learning with label distribution privacy (Client -> Server -> Client training flow).
        - 'fitfl': FitFL-Accurate and Fast Federated Learning by Hybrid Scaling
        - 'mix2sfl': Mix2SFL (SmashMix + GradMix on Parallel SL)
    """
    criterion: str  # Loss function (e.g., 'ce' for CrossEntropy, 'mse' for Mean Squared Error).
    optimizer: str  # Optimizer to use (e.g., 'sgd', 'adam').
    learning_rate: float  # Learning rate for the optimizer.
    server_learning_rate: float  # Server-side learning rate (default: learning_rate * num_clients_per_round).
    momentum: float  # Momentum factor for the optimizer (used in optimizers like SGD).
    local_epochs: int  # Number of local epochs per client.
    global_round: int  # Number of global rounds to be executed.
    batch_size: int  # Batch size used for training.
    device: str  # Compute device (e.g., 'cpu', 'cuda').
    selector: str  # Client selection strategy (e.g., 'uniform').
    aggregator: str  # Aggregation method for client updates (e.g., 'fedavg').
    distributer: str  # Data distribution strategy (e.g., 'uniform', 'dirichlet').
    num_clients: int  # Total number of clients participating.
    num_clients_per_round: int  # Number of clients selected per round.
    round_duration: int  # Duration (in seconds) for each round.
    delete_fraction_of_data: bool  # When set to true, it randomly selects half of the labels from the entire set and removes half of the data belonging to those selected classes.

    # ==================
    #       PATHS
    # ==================
    model_path: str  # File path to save/load the model.
    dataset_path: str  # File path to the dataset location.

    # ==================
    #  METHOD OPTIONS
    # ==================
    # SFL
    server_model_aggregation: bool  # Whether to aggregate model on server.
    # SFL, SFL-U, SCALA
    split_strategy: str  # Model splitting strategy.
    split_layer: str  # Layer name for model splitting.
    # FitFL
    max_pruning_ratio: int  # Maximum pruning ratio.
    # FitFL
    min_pruning_ratio: int  # Minimum pruning ratio.
    # FitFL
    pruning_decision_threshold: float  # Pruning decision threshold.
    # FitFL, FedSparsify
    pruning_interval: int  # Pruning interval.
    # FitFL
    min_pruning_accuracy: float  # Minimum pruning accuracy.
    # FitFL
    ema_threshold: float  # EMA threshold.
    # FitFL
    num_extra_epoch: int
    # PruneFL
    initial_pruning_ratio: int  # Pruning ratio.
    # PruneFL
    initial_pruning_device_id: int  # Initial pruning device id.
    # PruneFL
    initial_pruning_epoch: int  # Initial pruning epoch.
    # FedSparsify
    target_pruning_ratio: int  # Pruning ratio.
    # FedSparsify
    pruning_control_variable: (
        float  # Control variable for pruning in FedSparsify method.
    )
    # FedProx
    prox_mu: float  # Proximal term constant.
    # SCALA
    enable_concatenation: bool  # Whether to enable concatenation of activations.
    # SCALA
    enable_logit_adjustment: bool  # Whether to enable logit adjustment.
    # USFL
    gradient_shuffle: bool
    gradient_shuffle_strategy: str
    gradient_shuffle_target: str
    gradient_average_weight: float
    adaptive_alpha_beta: float  # Sensitivity coefficient for adaptive alpha strategy
    use_additional_epoch: bool
    use_cumulative_usage: bool
    usage_decay_factor: float
    use_data_replication: (
        bool  # Use data replication (max-based) instead of trimming (min-based)
    )
    balancing_strategy: str  # "trimming" | "replication" | "target" (hybrid)
    balancing_target: str  # For target strategy: "mean" | "median" | fixed number
    # Mix2SFL
    mix2sfl_smashmix_enabled: bool
    mix2sfl_smashmix_ns_ratio: float
    mix2sfl_smashmix_lambda_dist: str
    mix2sfl_smashmix_beta_alpha: float
    mix2sfl_gradmix_enabled: bool
    mix2sfl_gradmix_phi: float
    mix2sfl_gradmix_reduce: str
    mix2sfl_gradmix_cprime_selection: str

    # ==================
    #  GLUE DATASET OPTIONS
    # ==================
    glue_tokenizer: str  # Tokenizer used for GLUE tasks (e.g., 'bert-base-uncased').
    glue_max_seq_length: int  # Maximum sequence length for tokenization in GLUE tasks.

    # ==================
    #  DIRICHLET DISTRIBUTER OPTIONS
    # ==================
    dirichlet_alpha: (
        float  # Alpha parameter for Dirichlet distribution in data splitting.
    )

    # ==================
    #  LABEL DISTRIBUTER OPTIONS
    # ==================
    labels_per_client: int

    # ==================
    #  SHARD DIRICHLET DISTRIBUTER OPTIONS
    # ==================
    min_require_size: (
        int  # Minimum required samples per client for shard_dirichlet distributer
    )

    # ==================
    #  G MEASUREMENT OPTIONS
    # ==================
    enable_g_measurement: bool  # Enable gradient dissimilarity (G) measurement
    diagnostic_rounds: (
        str  # Comma-separated list of rounds to run G measurement (e.g., "1,3,5")
    )
    use_variance_g: bool
    oracle_batch_size: Optional[int]
    g_measurement_mode: str  # "single" (1-step) | "k_batch" (first K batches) | "accumulated" (full round average)
    g_measurement_k: int  # Number of batches to collect in k_batch mode (default: 5)

    # ==================
    #  DRIFT MEASUREMENT OPTIONS (SCAFFOLD-style)
    # ==================
    enable_drift_measurement: bool  # Enable client drift measurement
    drift_sample_interval: int  # Measure drift every n steps (1 = every step, default: 1)

    # ==================
    #  MISSING CLASS SELECTOR OPTIONS
    # ==================
    num_missing_class: int

    # ==================
    #  RATIO SPLIT STRATEGY OPTIONS
    # ==================
    split_ratio: list[float]
    """
    Split ratio for model partitioning:
        - For 'sfl' method: Must have 1 value (model split into 2 parts).
        - For 'sfl-u' method: Must have 2 values (model split into 3 parts).
    """


def parse_args(custom_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-sd",
        "--seed",
        help="Defines random seed for reproducibility",
        action="store",
        type=int,
        dest="seed",
        default=42,
        required=False,
    )

    parser.add_argument(
        "-H",
        "--host",
        help="Defines server host",
        action="store",
        type=str,
        dest="host",
        default="0.0.0.0",
        required=False,
    )

    parser.add_argument(
        "-p",
        "--port",
        help="Defines server port",
        action="store",
        type=int,
        dest="port",
        default=1000,
        required=False,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="Defines dataset to use",
        action="store",
        type=str,
        dest="dataset",
        choices=[
            "cifar10",
            "mnist",
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
            "cifar100",
            "mnist",
            "fmnist",
        ],
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Defines model to use",
        action="store",
        type=str,
        dest="model",
        choices=[
            "resnet18",
            "resnet18_cifar",
            "lenet",
            "vgg11",
            "distilbert",
            "tiny_vgg11",
            "alexnet",
            "alexnet_scala",
            "mobilenet",
        ],
        required=True,
    )

    parser.add_argument(
        "-M",
        "--method",
        help="Defines method to use",
        action="store",
        type=str,
        dest="method",
        choices=[
            "sfl",
            "fl",
            "sfl-u",
            "fitfl",
            "prunefl",
            "fedsparsify",
            "nestfl",
            "fedprox",
            "scala",
            "usfl",
            "fedcbs",
            "sflprox",
            "cl",
            "mix2sfl",
            "scaffold_sfl",
        ],
        required=True,
    )

    parser.add_argument(
        "-c",
        "--criterion",
        help="Defines criterion to use",
        action="store",
        type=str,
        dest="criterion",
        choices=["ce", "mse", "bce"],
        default="ce",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        help="Defines optimizer to use",
        action="store",
        type=str,
        dest="optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adam",
        required=False,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Defines learning rate",
        action="store",
        type=float,
        dest="learning_rate",
        default=0.001,
        required=False,
    )

    parser.add_argument(
        "-slr",
        "--server_learning_rate",
        help="Server model learning rate (default: learning_rate * num_clients_per_round)",
        action="store",
        type=float,
        dest="server_learning_rate",
        default=None,
        required=False,
    )

    parser.add_argument(
        "-mt",
        "--momentum",
        help="Defines momentum",
        action="store",
        type=float,
        dest="momentum",
        default=0.9,
        required=False,
    )

    parser.add_argument(
        "-le",
        "--local-epochs",
        help="Defines epochs",
        action="store",
        type=int,
        dest="local_epochs",
        required=True,
    )

    parser.add_argument(
        "-de",
        "--device",
        help="Defines device to use",
        action="store",
        type=validate_device,
        dest="device",
        default="cpu",
        required=False,
    )

    parser.add_argument(
        "-glue-tz",
        "--glue-tokenizer",
        help="Defines tokenizer to use",
        action="store",
        type=str,
        dest="glue_tokenizer",
        choices=["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
        default="distilbert-base-uncased",
        required=False,
    )

    parser.add_argument(
        "-gl-msl",
        "--glue-max-seq-length",
        help="Defines max sequence length",
        action="store",
        type=int,
        dest="glue_max_seq_length",
        default=128,
        required=False,
    )

    parser.add_argument(
        "-m-path",
        "--model-path",
        help="Defines path to model",
        action="store",
        type=str,
        dest="model_path",
        default="./.models",
        required=False,
    )

    parser.add_argument(
        "-d-path",
        "--dataset-path",
        help="Defines path to dataset",
        action="store",
        type=str,
        dest="dataset_path",
        default="./.datasets",
        required=False,
    )

    parser.add_argument(
        "-gr",
        "--global-round",
        help="Defines global round",
        action="store",
        type=int,
        dest="global_round",
        required=True,
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        help="Defines batch size",
        action="store",
        type=int,
        dest="batch_size",
        default=64,
        required=False,
    )

    parser.add_argument(
        "-s",
        "--selector",
        help="Defines selector to use",
        action="store",
        type=str,
        dest="selector",
        choices=["uniform", "usfl", "missing_class", "fedcbs"],
        default="uniform",
        required=False,
    )

    parser.add_argument(
        "-aggr",
        "--aggregator",
        help="Defines aggregator to use",
        action="store",
        type=str,
        dest="aggregator",
        choices=["fedavg", "fitfl", "usfl"],
        default="fedavg",
        required=False,
    )

    parser.add_argument(
        "-nc",
        "--num-clients",
        help="Defines number of clients",
        action="store",
        type=int,
        dest="num_clients",
        required=True,
    )

    parser.add_argument(
        "-ncpr",
        "--num-clients-per-round",
        help="Defines number of clients per round",
        action="store",
        type=int,
        dest="num_clients_per_round",
        required=True,
    )

    parser.add_argument(
        "-distr",
        "--distributer",
        help="Defines distributer to use",
        action="store",
        type=str,
        dest="distributer",
        choices=["uniform", "iid", "dirichlet", "label", "label_dirichlet", "shard_dirichlet"],
        default="dirichlet",
        required=False,
    )

    parser.add_argument(
        "-diri-alpha",
        "--dirichlet-alpha",
        help="Defines alpha for dirichlet distributer",
        action="store",
        type=float,
        dest="dirichlet_alpha",
        default=0.5,
        required=False,
    )

    parser.add_argument(
        "-mrs",
        "--min-require-size",
        help="Minimum required samples per client for shard_dirichlet distributer. Default: max(10, batch_size//4)",
        action="store",
        type=int,
        dest="min_require_size",
        default=None,
        required=False,
    )

    parser.add_argument(
        "-rd",
        "--round-duration",
        help="Defines round duration (in seconds)",
        action="store",
        type=int,
        dest="round_duration",
        default=999999999999,
        required=False,
    )

    parser.add_argument(
        "-ss",
        "--split-strategy",
        help="Defines split strategy to use",
        action="store",
        type=str,
        dest="split_strategy",
        choices=["ratio_param", "ratio_layer", "layer_name"],
        required=False,
    )

    parser.add_argument(
        "-sl",
        "--split-layer",
        help="Defines split layer name for layer_name strategy",
        action="store",
        type=str,
        dest="split_layer",
        default="",
        required=False,
    )

    parser.add_argument(
        "-sr",
        "--split-ratio",
        help="Defines split ratio",
        action="store",
        type=float,
        nargs="+",
        dest="split_ratio",
        default=[],
        required=False,
    )

    parser.add_argument(
        "-max-p",
        "--max-pruning-ratio",
        help="Defines max pruning ratio",
        action="store",
        type=int,
        dest="max_pruning_ratio",
        default=1,
        required=False,
    )

    parser.add_argument(
        "-min-p",
        "--min-pruning-ratio",
        help="Defines min pruning ratio",
        action="store",
        type=int,
        dest="min_pruning_ratio",
        default=0,
        required=False,
    )

    parser.add_argument(
        "-p-thres",
        "--pruning-threshold",
        help="Defines pruning threshold",
        action="store",
        type=int,
        dest="pruning_decision_threshold",
        default=0.03,
        required=False,
    )

    parser.add_argument(
        "-min-p-acc",
        "--min-pruning-accuracy",
        help="Defines min pruning accuracy",
        action="store",
        type=float,
        dest="min_pruning_accuracy",
        default=0.1,
        required=False,
    )

    parser.add_argument(
        "-pi",
        "--pruning-interval",
        help="Defines pruning interval",
        action="store",
        type=int,
        dest="pruning_interval",
        default=20,
        required=False,
    )

    parser.add_argument(
        "-e-thres",
        "--ema-threshold",
        help="Defines EMA threshold",
        action="store",
        type=float,
        dest="ema_threshold",
        default=0.05,
        required=False,
    )

    parser.add_argument(
        "-ipr",
        "--initial-pruning-ratio",
        help="Defines initial pruning ratio",
        action="store",
        type=float,
        dest="initial_pruning_ratio",
        default=0.2,
        required=False,
    )

    parser.add_argument(
        "-ipd",
        "--initial-pruning-device-id",
        help="Defines initial pruning device id",
        action="store",
        type=int,
        dest="initial_pruning_device_id",
        default=0,
        required=False,
    )

    parser.add_argument(
        "-ipe",
        "--initial-pruning-epoch",
        help="Defines initial pruning epoch",
        action="store",
        type=int,
        dest="initial_pruning_epoch",
        default=5,
        required=False,
    )

    parser.add_argument(
        "-pcv",
        "--pruning-control-variable",
        help="Defines control variable for pruning in FedSparsify method",
        action="store",
        type=float,
        dest="pruning_control_variable",
        default=3,
        required=False,
    )

    parser.add_argument(
        "-tpr",
        "--target-pruning-ratio",
        help="Defines target pruning ratio",
        action="store",
        type=float,
        dest="target_pruning_ratio",
        default=0.2,
        required=False,
    )

    parser.add_argument(
        "-ne",
        "--num-extra-epoch",
        help="Defines maximum number of extra epoch",
        action="store",
        type=int,
        dest="num_extra_epoch",
        default=99999999,
        required=False,
    )

    parser.add_argument(
        "-lpc",
        "--labels-per-client",
        help="Defines maximum number of labels per client",
        action="store",
        type=int,
        dest="labels_per_client",
        default=2,
        required=False,
    )

    parser.add_argument(
        "-pm",
        "--prox-mu",
        help="Defines proximal term constant",
        action="store",
        type=float,
        dest="prox_mu",
        default=1.0,
        required=False,
    )

    parser.add_argument(
        "-sma",
        "--server-model-aggregation",
        help="Defines whether to aggregate model on server",
        action="store",
        type=str_to_bool,
        default=False,
        dest="server_model_aggregation",
    )

    parser.add_argument(
        "-df",
        "--delete-fraction-of-data",
        help="Defines whether to delete fraction of data",
        action="store",
        type=str_to_bool,
        dest="delete_fraction_of_data",
        default=False,
        required=False,
    )

    parser.add_argument(
        "-nf",
        "--networking-fairness",
        help="Enable networking fairness (default: enabled)",
        action="store_true",
        dest="networking_fairness",
    )

    parser.add_argument(
        "-nnf",
        "--no-networking-fairness",
        help="Disable networking fairness",
        action="store_false",
        dest="networking_fairness",
    )

    parser.add_argument(
        "-nmc",
        "--num-missing-class",
        help="Defines number of missing class",
        action="store",
        type=int,
        dest="num_missing_class",
        default=0,
    )
    parser.add_argument(
        "-ec",
        "--enable-concatenation",
        help="Enable concatenation of activations (default: enabled)",
        action="store_true",
        dest="enable_concatenation",
    )

    parser.add_argument(
        "-nec",
        "--no-enable-concatenation",
        help="Disable concatenation of activations",
        action="store_false",
        dest="enable_concatenation",
    )

    parser.add_argument(
        "-ela",
        "--enable-logit-adjustment",
        help="Enable logit adjustment (default: enabled)",
        action="store_true",
        dest="enable_logit_adjustment",
    )

    parser.add_argument(
        "-nela",
        "--no-enable-logit-adjustment",
        help="Disable logit adjustment",
        action="store_false",
        dest="enable_logit_adjustment",
    )

    parser.add_argument(
        "-gs",
        "--gradieng-shuffle",
        help="Enable gradieng shuffle (default: enabled)",
        action="store_true",
        dest="gradient_shuffle",
    )

    parser.add_argument(
        "-ngs",
        "--no-gradieng-shuffle",
        help="Disable gradieng shuffle (default: enabled)",
        action="store_false",
        dest="gradient_shuffle",
    )

    parser.add_argument(
        "-gss",
        "--gradient-shuffle-strategy",
        help="Defines split strategy to use",
        action="store",
        type=str,
        dest="gradient_shuffle_strategy",
        choices=["random", "inplace", "average", "average_adaptive_alpha"],
        required=False,
    )

    parser.add_argument(
        "-gst",
        "--gradient-shuffle-target",
        help="Defines shuffle target for tuple gradients (default: all)",
        action="store",
        type=str,
        dest="gradient_shuffle_target",
        choices=["all", "activation_only"],
        default="all",
        required=False,
    )

    parser.add_argument(
        "-gaw",
        "--gradient-average-weight",
        help="Defines weight for average gradient shuffle strategy (default: 0.5)",
        action="store",
        type=float,
        dest="gradient_average_weight",
        default=0.5,
        required=False,
    )

    parser.add_argument(
        "-aab",
        "--adaptive-alpha-beta",
        help="Sensitivity coefficient (beta) for adaptive alpha strategy (default: 2.0)",
        action="store",
        type=float,
        dest="adaptive_alpha_beta",
        default=2.0,
        required=False,
    )

    parser.add_argument(
        "-uae",
        "--use-additional-epoch",
        help="Enable using additional epochs to compensate for removed data in USFL",
        action="store_true",
        dest="use_additional_epoch",
    )

    parser.add_argument(
        "-ucu",
        "--use-cumulative-usage",
        help="Enable using cumulative usage with time decay for client selection in USFL",
        action="store_true",
        dest="use_cumulative_usage",
    )
    parser.add_argument(
        "-udf",
        "--usage-decay-factor",
        help="Usage decay factor for cumulative usage (default: 0.99)",
        action="store",
        type=float,
        dest="usage_decay_factor",
    )

    parser.add_argument(
        "-udbs",
        "--use-dynamic-batch-scheduler",
        help="Enable the dynamic batch scheduler to utilize all client data without waste.",
        action="store_true",
        dest="use_dynamic_batch_scheduler",
    )
    parser.add_argument(
        "-ufs",
        "--use-fresh-scoring",
        help="Enable the Top-n + Freshness scoring algorithm for client selection.",
        action="store_true",
        dest="use_fresh_scoring",
    )
    parser.add_argument(
        "-udr",
        "--use-data-replication",
        help="Use data replication (max-based augmentation) instead of trimming (min-based). When enabled, trimming is disabled.",
        action="store_true",
        dest="use_data_replication",
    )
    parser.add_argument(
        "-fdr",
        "--freshness-decay-rate",
        help="Decay rate for freshness score calculation (default: 0.5)",
        action="store",
        type=float,
        dest="freshness_decay_rate",
    )
    parser.add_argument(
        "-bstrat",
        "--balancing-strategy",
        help="Data balancing strategy: trimming (min-based), replication (max-based), or target (hybrid)",
        action="store",
        type=str,
        dest="balancing_strategy",
        choices=["trimming", "replication", "target"],
        default="trimming",
        required=False,
    )
    parser.add_argument(
        "-btarget",
        "--balancing-target",
        help="Target for 'target' balancing strategy: 'mean', 'median', or a fixed number (e.g., '100')",
        action="store",
        type=str,
        dest="balancing_target",
        default="mean",
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-smashmix-enabled",
        help="Enable SmashMix for mix2sfl",
        action="store",
        type=str_to_bool,
        dest="mix2sfl_smashmix_enabled",
        default=True,
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-smashmix-ns-ratio",
        help="SmashMix partner ratio per step (ns = floor(|C| * ratio))",
        action="store",
        type=float,
        dest="mix2sfl_smashmix_ns_ratio",
        default=0.2,
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-smashmix-lambda-dist",
        help="Lambda distribution for SmashMix (uniform or beta)",
        action="store",
        type=str,
        dest="mix2sfl_smashmix_lambda_dist",
        default="uniform",
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-smashmix-beta-alpha",
        help="Alpha for beta distribution when SmashMix lambda_dist=beta",
        action="store",
        type=float,
        dest="mix2sfl_smashmix_beta_alpha",
        default=1.0,
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-gradmix-enabled",
        help="Enable GradMix for mix2sfl",
        action="store",
        type=str_to_bool,
        dest="mix2sfl_gradmix_enabled",
        default=True,
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-gradmix-phi",
        help="GradMix ratio phi (|C'|/|C|)",
        action="store",
        type=float,
        dest="mix2sfl_gradmix_phi",
        default=0.5,
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-gradmix-reduce",
        help="GradMix reduction for C' (sum or mean)",
        action="store",
        type=str,
        dest="mix2sfl_gradmix_reduce",
        default="sum",
        required=False,
    )

    parser.add_argument(
        "--mix2sfl-gradmix-cprime-selection",
        help="GradMix C' selection strategy",
        action="store",
        type=str,
        dest="mix2sfl_gradmix_cprime_selection",
        default="random_each_step",
        required=False,
    )

    # G Measurement Options
    parser.add_argument(
        "--enable-g-measurement",
        help="Enable gradient dissimilarity (G) measurement",
        action="store_true",
        dest="enable_g_measurement",
    )
    parser.add_argument(
        "--diagnostic-rounds",
        help="Comma-separated list of rounds to run G measurement (e.g., '1,3,5')",
        type=str,
        default="1,3,5",
        dest="diagnostic_rounds",
    )
    parser.add_argument(
        "--use-variance-g",
        help="Use variance-based G measurement",
        action="store_true",
        dest="use_variance_g",
    )
    parser.add_argument(
        "--oracle-batch-size",
        help="Batch size for oracle gradient calculation (default: training batch size)",
        action="store",
        type=int,
        dest="oracle_batch_size",
        default=None,
    )
    parser.add_argument(
        "--g-measurement-mode",
        help="G measurement mode: 'single' (1-step) | 'k_batch' (first K batches) | 'accumulated' (full round average)",
        action="store",
        type=str,
        dest="g_measurement_mode",
        choices=["single", "k_batch", "accumulated"],
        default="single",
    )
    parser.add_argument(
        "--g-measurement-k",
        help="Number of batches to collect in k_batch mode (default: 5)",
        action="store",
        type=int,
        dest="g_measurement_k",
        default=5,
    )

    # Drift Measurement Options (SCAFFOLD-style)
    parser.add_argument(
        "--enable-drift-measurement",
        help="Enable client drift measurement (SCAFFOLD-style)",
        action="store_true",
        dest="enable_drift_measurement",
    )
    parser.add_argument(
        "--drift-sample-interval",
        help="Measure drift every n steps (1 = every step, default: 1)",
        action="store",
        type=int,
        dest="drift_sample_interval",
        default=1,
    )

    parser.set_defaults(networking_fairness=True)

    try:
        args = parser.parse_args(args=custom_args)
    except:
        args, _ = parser.parse_known_args(args=custom_args)

    return Config(**vars(args))
