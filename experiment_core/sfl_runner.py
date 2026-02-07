"""
SFL simulation runner — bridges batch_runner's JSON spec with simulation.py's
internal run_simulation() function.

Usage:
    python -m experiment_core.sfl_runner --spec spec.json --repo-root .

This script:
1. Reads a single-experiment spec JSON
2. Converts it to the short-flag list that simulation.py's argparser expects
3. Calls run_simulation() directly (same process, asyncio)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


# simulation.py config key → short flag (value flags)
_VALUE_FLAGS = {
    "dataset": "-d",
    "model": "-m",
    "method": "-M",
    "local_epochs": "-le",
    "global_epochs": "-gr",
    "device": "-de",
    "total_clients": "-nc",
    "clients_per_round": "-ncpr",
    "distributer": "-distr",
    "labels_per_client": "-lpc",
    "split_strategy": "-ss",
    "split_layer": "-sl",
    "split_ratio": "-sr",
    "learning_rate": "-lr",
    "server_learning_rate": "-slr",
    "momentum": "-mt",
    "batch_size": "-bs",
    "dirichlet_alpha": "-diri-alpha",
    "min_require_size": "-mrs",
    "port": "-p",
    "server_model_aggregation": "-sma",
    "optimizer": "-o",
    "criterion": "-c",
    "seed": "-sd",
    "selector": "-s",
    "aggregator": "-aggr",
    "gradient_shuffle_strategy": "-gss",
    "gradient_shuffle_target": "-gst",
    "gradient_average_weight": "-gaw",
    "adaptive_alpha_beta": "-aab",
    "usage_decay_factor": "-udf",
    "freshness_decay_rate": "-fdr",
    "balancing_strategy": "-bstrat",
    "balancing_target": "-btarget",
    "g_measure_frequency": "--g-measure-frequency",
    "g_measurement_mode": "--g-measurement-mode",
    "g_measurement_k": "--g-measurement-k",
    "drift_sample_interval": "--drift-sample-interval",
    "result_output_dir": "--result-output-dir",
    "delete_fraction_of_data": "-df",
    # Mix2SFL
    "mix2sfl_smashmix_enabled": "--mix2sfl-smashmix-enabled",
    "mix2sfl_smashmix_ns_ratio": "--mix2sfl-smashmix-ns-ratio",
    "mix2sfl_smashmix_lambda_dist": "--mix2sfl-smashmix-lambda-dist",
    "mix2sfl_smashmix_beta_alpha": "--mix2sfl-smashmix-beta-alpha",
    "mix2sfl_gradmix_enabled": "--mix2sfl-gradmix-enabled",
    "mix2sfl_gradmix_phi": "--mix2sfl-gradmix-phi",
    "mix2sfl_gradmix_reduce": "--mix2sfl-gradmix-reduce",
    "mix2sfl_gradmix_cprime_selection": "--mix2sfl-gradmix-cprime-selection",
}

# store_true flags: append only when value is "true"
_STORE_TRUE_FLAGS = {
    "gradient_shuffle": "-gs",
    "use_dynamic_batch_scheduler": "-udbs",
    "use_additional_epoch": "-uae",
    "use_cumulative_usage": "-ucu",
    "use_fresh_scoring": "-ufs",
    "use_data_replication": "-udr",
    "use_variance_g": "--use-variance-g",
    "enable_g_measurement": "--enable-g-measurement",
    "enable_drift_measurement": "--enable-drift-measurement",
    "enable_inround_evaluation": "--enable-inround-evaluation",
    "networking_fairness": "-nf",
    "enable_concatenation": "-ec",
    "enable_logit_adjustment": "-ela",
}

# store_false flags: append when value is "false"
# Needed for args whose argparse default is True (e.g., networking_fairness).
_STORE_FALSE_FLAGS = {
    "networking_fairness": "-nnf",
    "enable_concatenation": "-nec",
    "enable_logit_adjustment": "-nela",
    "gradient_shuffle": "-ngs",
}


def spec_to_workload(spec: Dict[str, Any]) -> Dict[str, str]:
    """Convert a single-experiment spec to simulation.py workload dict."""
    common = spec["common"]
    dist = common["distribution"]
    overrides = spec.get("framework_overrides", {})

    method = str(spec.get("method", "sfl"))
    selector_default = "usfl" if method == "usfl" else "uniform"
    batch_size = (
        int(common["server_batch_size"]) if method == "usfl" else int(common["client_batch_size"])
    )

    workload: Dict[str, str] = {
        "host": "0.0.0.0",
        "port": "3000",
        "dataset": str(common["dataset"]),
        "model": str(common["model"]),
        "method": method,
        "criterion": "ce",
        "optimizer": "sgd",
        "learning_rate": str(common["learning_rate"]),
        "momentum": str(common["momentum"]),
        "local_epochs": str(common["local_epochs"]),
        "global_epochs": str(common["rounds"]),
        "batch_size": str(batch_size),
        "device": str(common["device"]),
        "selector": str(overrides.get("selector", selector_default)),
        "aggregator": str(overrides.get("aggregator", "fedavg")),
        "distributer": str(dist.get("mode", "shard_dirichlet")),
        "total_clients": str(common["total_clients"]),
        "clients_per_round": str(common["clients_per_round"]),
        "split_strategy": "layer_name",
        "split_layer": str(common["split_layer"]),
        "seed": str(common["seed"]),
        "server_model_aggregation": "false",
        "networking_fairness": "false",
        "delete_fraction_of_data": "false",
    }

    if dist.get("dirichlet_alpha") is not None:
        workload["dirichlet_alpha"] = str(dist["dirichlet_alpha"])
    if dist.get("labels_per_client") is not None:
        workload["labels_per_client"] = str(dist["labels_per_client"])
    if dist.get("min_require_size") is not None:
        workload["min_require_size"] = str(dist["min_require_size"])

    # G measurement
    if common.get("enable_g_measurement"):
        workload["enable_g_measurement"] = "true"
    if common.get("use_variance_g"):
        workload["use_variance_g"] = "true"
    if common.get("g_measurement_mode"):
        workload["g_measurement_mode"] = str(common["g_measurement_mode"])
    if common.get("g_measurement_k"):
        workload["g_measurement_k"] = str(common["g_measurement_k"])
    if common.get("g_measure_frequency"):
        workload["g_measure_frequency"] = str(common["g_measure_frequency"])

    # Drift
    if common.get("drift", {}).get("enabled"):
        workload["enable_drift_measurement"] = "true"
        workload["drift_sample_interval"] = str(
            common.get("drift", {}).get("sample_interval", 1)
        )

    # Result output directory (from batch_runner)
    result_output_dir = spec.get("execution", {}).get("result_output_dir", "")
    if result_output_dir:
        workload["result_output_dir"] = result_output_dir

    # Per-method sfl_args overrides
    sfl_args = overrides.get("sfl_args", {})
    for k, v in sfl_args.items():
        if v is None:
            continue  # skip null values from JSON config
        config_key = k.replace("-", "_")
        if isinstance(v, bool):
            if v:
                workload[config_key] = "true"
        else:
            workload[config_key] = str(v)

    return workload


def workload_to_server_args(workload: Dict[str, str]) -> List[str]:
    """Convert workload dict to short-flag list for s_parse_args()."""
    args: List[str] = []

    for key, value in workload.items():
        if key in _STORE_TRUE_FLAGS:
            if str(value).lower() == "true":
                args.append(_STORE_TRUE_FLAGS[key])
            elif key in _STORE_FALSE_FLAGS:
                # Explicitly disable (needed when argparse default is True)
                args.append(_STORE_FALSE_FLAGS[key])
            continue

        if key in _VALUE_FLAGS:
            args.extend([_VALUE_FLAGS[key], str(value)])

    return args


def workload_to_client_args(
    workload: Dict[str, str],
    gpu_id: int | None = None,
) -> List[List[str]]:
    """Build client arg lists (one per client), same as simulation.py."""
    import torch

    total_clients = int(workload.get("total_clients", "100"))
    device = workload.get("device", "cuda")
    port = workload.get("port", "3000")

    available_gpus: List[int] = []
    if device == "cuda":
        if gpu_id is not None:
            # CUDA_VISIBLE_DEVICES already set, so device 0 is our GPU
            available_gpus = [0]
        else:
            available_gpus = list(range(torch.cuda.device_count()))

    client_commands: List[List[str]] = []
    for i in range(total_clients):
        client_dev = (
            f"cuda:{available_gpus[i % len(available_gpus)]}"
            if device == "cuda" and available_gpus
            else device
        )
        client_commands.append([
            "-cid", str(i),
            "-d", client_dev,
            "-su", f"localhost:{port}",
        ])

    return client_commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFL simulation from spec JSON")
    parser.add_argument("--spec", required=True, help="Path to single-experiment spec JSON")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    repo_root = Path(args.repo_root).resolve()
    sfl_dir = repo_root / "sfl_framework-fork-feature-training-tracker"

    # Add SFL framework paths (same as simulation.py lines 79-82)
    sys.path.insert(0, str(sfl_dir / "client"))
    sys.path.insert(0, str(sfl_dir / "server"))
    sys.path.insert(0, str(sfl_dir))

    os.chdir(sfl_dir)

    # Import simulation's run function (after path setup)
    from simulation import run_simulation

    workload = spec_to_workload(spec)
    server_args = workload_to_server_args(workload)
    client_args = workload_to_client_args(workload)

    print(f"[sfl_runner] Method: {workload.get('method', 'sfl')}")
    print(f"[sfl_runner] Server args: {' '.join(server_args)}")
    print(f"[sfl_runner] Clients: {len(client_args)}")
    print()

    asyncio.run(run_simulation(server_args, client_args))


if __name__ == "__main__":
    main()
