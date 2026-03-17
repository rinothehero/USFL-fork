"""
Self-contained configuration. No external framework dependencies.

Two input modes:
    python -m sfl_sim --config experiment.json          # JSON config
    python -m sfl_sim --config experiment.json -gr 10   # JSON + CLI overrides
    python -m sfl_sim -d cifar10 -m resnet18 -M sfl ... # CLI flags only
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class Config:
    """All experiment parameters in one place."""

    # --- Core ---
    method: str = "sfl"
    dataset: str = "cifar10"
    model: str = "resnet18"
    device: str = "cpu"
    seed: int = 42

    # --- Training ---
    global_round: int = 100
    num_clients: int = 100
    num_clients_per_round: int = 10
    local_epochs: int = 5
    batch_size: int = 50

    # --- Optimizer ---
    optimizer: str = "sgd"
    learning_rate: float = 0.001
    server_learning_rate: Optional[float] = None
    scale_server_lr: bool = False
    momentum: float = 0.9
    weight_decay: float = 0.0
    criterion: str = "ce"

    # --- Model splitting ---
    split_strategy: str = "layer_name"
    split_layer: str = "layer2"
    split_ratio: float = 0.5

    # --- Data distribution ---
    distribution: str = "shard_dirichlet"
    dirichlet_alpha: float = 0.3
    labels_per_client: int = 2
    min_require_size: int = 10

    # --- Selection & Aggregation ---
    selector: str = "uniform"
    aggregator: str = "fedavg"

    # --- Gradient ---
    clip_grad: bool = False
    clip_grad_max_norm: float = 1.0
    scale_client_grad: bool = False  # Multiply activation grad by num active clients

    # --- Server training mode ---
    # "per_client": SFL default (one server step per client)
    # "concatenated": USFL default (detach + re-forward, safe for all hooks)
    # "concatenated_fused": optimized (register_hook, no re-forward, faster)
    server_training_mode: str = ""  # empty = auto (hook decides)

    # --- Data exhaustion policy ---
    # What to do when a client runs out of data mid-round:
    # "cycling": restart from beginning (may overfit small clients in Non-IID)
    # "skip": skip exhausted clients in remaining iterations
    #          Note: may cause client drift in later iterations,
    #          aggregation weights still use total dataset_size (not actual trained batches)
    # "break": stop all clients when any one is exhausted (wastes large client data)
    # "dbs": Dynamic Batch Scheduler — adjust batch sizes so all exhaust simultaneously
    #        (recommended for USFL; requires use_dynamic_batch_scheduler=true)
    #        Note: in SFL (per_client mode), "dbs" falls back to cycling
    #        because DBS scheduling only runs in USFL hook's pre_round
    exhaustion_policy: str = "dbs"

    # --- USFL-specific (Phase 2) ---
    balancing_strategy: str = "trimming"
    balancing_target: str = "mean"
    gradient_shuffle: bool = False
    gradient_shuffle_strategy: str = "random"
    gradient_average_weight: float = 0.5
    adaptive_alpha_beta: float = 0.1
    use_dynamic_batch_scheduler: bool = False
    use_cumulative_usage: bool = False
    use_fresh_scoring: bool = False
    usage_decay_factor: float = 0.9
    freshness_decay_rate: float = 0.1

    # --- GAS-specific ---
    g_measure_frequency: int = 10  # Measure G-scores every N rounds
    v_test_batches: int = 10       # Number of test batches for V-value estimation
    warmup_rounds: int = 5         # Use uniform selection for first N rounds
    oracle_max_batches: Optional[int] = None  # Limit oracle computation (None = use all)

    # --- Result output ---
    result_dir: str = "results"


# JSON key → Config field name (only for keys that differ)
_JSON_TO_CONFIG = {
    "rounds": "global_round",
    "alpha": "dirichlet_alpha",
    "clients_per_round": "num_clients_per_round",
}


def _str_to_bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def _load_json_config(path: str) -> dict:
    """Load JSON config, map keys to Config field names, skip comments."""
    with open(path) as f:
        raw = json.load(f)

    config_field_names = {f.name for f in fields(Config)}
    mapped = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        field_name = _JSON_TO_CONFIG.get(k, k)
        if field_name in config_field_names:
            mapped[field_name] = v
    return mapped


def parse_args(argv: list[str] | None = None) -> Config:
    """
    Parse config from JSON file and/or CLI flags.

    Priority: CLI flags > JSON config > defaults
    """
    # Pre-parse: extract --config before main parsing
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", dest="config_file", default=None)
    pre_ns, remaining = pre.parse_known_args(argv)

    # Load JSON defaults if provided
    json_defaults = {}
    if pre_ns.config_file:
        json_defaults = _load_json_config(pre_ns.config_file)

    # Main parser
    p = argparse.ArgumentParser(description="SFL Simulation Framework")
    p.add_argument("--config", dest="_config_file", default=None)

    # Core
    p.add_argument("-d", dest="dataset", default="cifar10")
    p.add_argument("-m", dest="model", default="resnet18")
    p.add_argument("-M", dest="method", default="sfl")
    p.add_argument("-de", dest="device", default="cpu")
    p.add_argument("-sd", dest="seed", type=int, default=42)

    # Training
    p.add_argument("-gr", dest="global_round", type=int, default=100)
    p.add_argument("-nc", dest="num_clients", type=int, default=100)
    p.add_argument("-ncpr", dest="num_clients_per_round", type=int, default=10)
    p.add_argument("-le", dest="local_epochs", type=int, default=5)
    p.add_argument("-bs", dest="batch_size", type=int, default=50)

    # Optimizer
    p.add_argument("-o", dest="optimizer", default="sgd")
    p.add_argument("-lr", dest="learning_rate", type=float, default=0.001)
    p.add_argument("-slr", dest="server_learning_rate", type=float, default=None)
    p.add_argument("--scale-server-lr", dest="scale_server_lr",
                    type=_str_to_bool, default=False)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    p.add_argument("-c", dest="criterion", default="ce")

    # Server training mode
    p.add_argument("--server-training-mode", dest="server_training_mode", default="")

    # Data exhaustion policy
    p.add_argument("--exhaustion-policy", dest="exhaustion_policy", default="dbs")

    # Model splitting
    p.add_argument("-ss", dest="split_strategy", default="layer_name")
    p.add_argument("-sl", dest="split_layer", default="layer2")
    p.add_argument("-sr", dest="split_ratio", type=float, default=0.5)

    # Data distribution
    p.add_argument("-distr", dest="distribution", default="shard_dirichlet")
    p.add_argument("-diri-alpha", dest="dirichlet_alpha", type=float, default=0.3)
    p.add_argument("-lpc", dest="labels_per_client", type=int, default=2)
    p.add_argument("-mrs", dest="min_require_size", type=int, default=10)

    # Selection & Aggregation
    p.add_argument("-s", dest="selector", default="uniform")
    p.add_argument("-aggr", dest="aggregator", default="fedavg")

    # Gradient
    p.add_argument("--clip-grad", dest="clip_grad", type=_str_to_bool, default=False)
    p.add_argument("--clip-grad-max-norm", dest="clip_grad_max_norm",
                    type=float, default=1.0)
    p.add_argument("--scale-client-grad", dest="scale_client_grad",
                    type=_str_to_bool, default=False)

    # USFL-specific
    p.add_argument("-bstrat", dest="balancing_strategy", default="trimming")
    p.add_argument("-btarget", dest="balancing_target", default="mean")
    p.add_argument("-gs", dest="gradient_shuffle", action="store_true")
    p.add_argument("-gss", dest="gradient_shuffle_strategy", default="random")
    p.add_argument("-gaw", dest="gradient_average_weight", type=float, default=0.5)
    p.add_argument("-aab", dest="adaptive_alpha_beta", type=float, default=0.1)
    p.add_argument("-udbs", dest="use_dynamic_batch_scheduler", action="store_true")
    p.add_argument("-ucu", dest="use_cumulative_usage", action="store_true")
    p.add_argument("-ufs", dest="use_fresh_scoring", action="store_true")
    p.add_argument("-udf", dest="usage_decay_factor", type=float, default=0.9)
    p.add_argument("-fdr", dest="freshness_decay_rate", type=float, default=0.1)

    # GAS-specific
    p.add_argument("--g-measure-frequency", dest="g_measure_frequency",
                    type=int, default=10)
    p.add_argument("--v-test-batches", dest="v_test_batches",
                    type=int, default=10)
    p.add_argument("--warmup-rounds", dest="warmup_rounds",
                    type=int, default=5)
    p.add_argument("--oracle-max-batches", dest="oracle_max_batches",
                    type=int, default=None)

    # Result output
    p.add_argument("--result-dir", dest="result_dir", default="results")

    # Accepted but ignored (for CLI compatibility with old framework)
    p.add_argument("-sma", dest="_server_model_aggregation",
                    type=_str_to_bool, default=False)
    p.add_argument("-nnf", dest="_no_negative_filter", action="store_true")

    # Apply JSON config as defaults (CLI flags will override)
    if json_defaults:
        p.set_defaults(**json_defaults)

    ns = p.parse_args(argv)

    # Build Config from parsed args, filtering out ignored flags
    config_field_names = {f.name for f in fields(Config)}
    kwargs = {k: v for k, v in vars(ns).items() if k in config_field_names}
    return Config(**kwargs)
