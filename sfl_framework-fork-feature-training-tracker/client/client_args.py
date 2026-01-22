import argparse
import re
from dataclasses import dataclass


@dataclass
class ServerConfig:
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
    delete_fraction_of_data: bool
    gradient_shuffle: bool
    gradient_shuffle_strategy: str
    gradient_shuffle_target: str
    gradient_average_weight: float
    adaptive_alpha_beta: float  # Sensitivity coefficient for adaptive alpha strategy
    use_additional_epoch: bool
    use_cumulative_usage: bool
    usage_decay_factor: float
    use_data_replication: bool
    balancing_strategy: str
    balancing_target: str
    mix2sfl_smashmix_enabled: bool
    mix2sfl_smashmix_ns_ratio: float
    mix2sfl_smashmix_lambda_dist: str
    mix2sfl_smashmix_beta_alpha: float
    mix2sfl_gradmix_enabled: bool
    mix2sfl_gradmix_phi: float
    mix2sfl_gradmix_reduce: str
    mix2sfl_gradmix_cprime_selection: str

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
    # G Measurement
    enable_g_measurement: bool  # Enable gradient dissimilarity (G) measurement
    diagnostic_rounds: str  # Comma-separated list of rounds to run G measurement
    use_variance_g: bool
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

    # ==================
    #  DO NOT REMOVE THIS ARGUMENT!!!!!
    #  DATASET DISTRIBUTION
    #  THIS ARGUMENT IS NOT PROVIDED BY USER, SERVER WILL GENERATE IT
    # ==================
    mask_ids: list[int]
    """
    Mask IDs for model partitioning.
    """


@dataclass
class Config:
    client_id: int
    server_uri: str
    device: str


def validate_device(value):
    if value in ["cpu", "mps", "cuda"]:
        return value
    if re.match(r"^cuda:\d+$", value):
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid device: '{value}'. Must be 'cpu', 'mps', or 'cuda:{int}'."
    )


def parse_args(custom_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-cid",
        "--client-id",
        help="Defines client ID",
        action="store",
        type=int,
        dest="client_id",
        required=True,
    )

    parser.add_argument(
        "-su",
        "--server-uri",
        help="Defines server URI",
        action="store",
        type=str,
        dest="server_uri",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Defines device to use",
        action="store",
        type=validate_device,
        dest="device",
        required=True,
    )

    try:
        args = parser.parse_args(args=custom_args)
    except SystemExit:
        args, _ = parser.parse_known_args(args=custom_args)

    print(args)
    return Config(**vars(args))
