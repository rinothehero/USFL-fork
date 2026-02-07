"""
Simulation Runner with Workload Management

QUICK START (edit this file, then run it):
1) Set BASE_CONFIG (dataset/model/optimizer).
2) Add entries in USFL_OPTIONS (name is historical; any method can go here).
3) Run: python simulation.py

MINIMAL EXAMPLE (mix2sfl):
    USFL_OPTIONS = {
        "MIX2": {
            "method": "mix2sfl",
            "selector": "uniform",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18",
            "batch_size": "64",
            "labels_per_client": "2",
            "dirichlet_alpha": "0.3",
            "split_layer": "layer1.1.bn2",
        },
    }

FEATURES:
1. Skip/Resume: Set START_INDEX and END_INDEX to run specific workload ranges
2. Auto-timeout: Automatically skip stuck workloads after WORKLOAD_TIMEOUT seconds
3. Filtering: Skip specific parameter combinations using SKIP_FILTERS
4. Error handling: Continue to next workload on errors

EXAMPLES:

  1. Run all 900 workloads:
      START_INDEX = 0
      END_INDEX = None
      WORKLOAD_TIMEOUT = 3600
      SKIP_FILTERS = []

  2. Resume from workload 101:
      START_INDEX = 100
      END_INDEX = None

  3. Run workloads 101-200:
      START_INDEX = 100
      END_INDEX = 200

  4. Test with first 10 workloads:
      START_INDEX = 0
      END_INDEX = 10
      WORKLOAD_TIMEOUT = 300

  5. Skip problematic combinations (extreme non-IID):
      SKIP_FILTERS = [
          {"dirichlet_alpha": 0.1, "labels_per_client": 2},
      ]

  6. Skip multiple conditions:
      SKIP_FILTERS = [
          {"dirichlet_alpha": 0.1, "labels_per_client": 2},  # Skip extreme non-IID
          {"batch_size": 256},                               # Skip large batches
          {"usfl_option": "B"},                              # Skip USFL disabled
      ]

  7. Skip only USFL-B with specific alpha:
      SKIP_FILTERS = [
          {"dirichlet_alpha": 0.1, "usfl_option": "B"},
      ]

  8. Skip gradient shuffle with random strategy:
      SKIP_FILTERS = [
          {"gradient_shuffle": "true", "gradient_shuffle_strategy": "random"},
      ]
"""

import asyncio
import os
import sys
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "client"))
sys.path.insert(0, os.path.join(current_dir, "server"))
sys.path.insert(0, os.path.join(current_dir))

import torch

from client.client_args import parse_args as c_parse_args
from client.modules.ws.inmemory_connection import (
    InMemoryConnection as InMemoryClientConnection,
)
from client.simulation_client import SimulationClient
from server.modules.global_dict.global_dict import GlobalDict as GlobalDictServer
from server.modules.ws.inmemory_connection import (
    InMemoryConnection as InMemoryServerConnection,
)
from server.server_args import parse_args as s_parse_args
from server.simulation_server import SimulationServer


async def run_simulation(server_args, client_args_list):
    server_args = s_parse_args(server_args)
    client_args_list = [c_parse_args(args) for args in client_args_list]

    server_global_dict = GlobalDictServer(server_args)
    inmemory_server_connection = InMemoryServerConnection(
        server_args, server_global_dict
    )

    inmemory_client_connections = [
        InMemoryClientConnection(client_args, inmemory_server_connection)
        for client_args in client_args_list
    ]

    simulation_server = SimulationServer(server_args, inmemory_server_connection)
    simulation_clients = [
        SimulationClient(client_args, inmemory_client_connections[i])
        for i, client_args in enumerate(client_args_list)
    ]

    await asyncio.gather(
        simulation_server.run(),
        *[client.run() for client in simulation_clients],
    )


if __name__ == "__main__":
    import itertools

    # Parse command line arguments for GPU selection
    # Usage: python simulation.py gpu=0,1,2
    GPU_DEVICES = None  # None means use all available GPUs
    for arg in sys.argv[1:]:
        if arg.startswith("gpu="):
            gpu_str = arg.split("=")[1]
            GPU_DEVICES = [int(g.strip()) for g in gpu_str.split(",")]
            print(f"[GPU] Using specified GPUs: {GPU_DEVICES}")
            break

    if GPU_DEVICES is None:
        print(f"[GPU] Using all available GPUs (use 'gpu=0,1' to specify)")

    # Fixed base configuration
    BASE_CONFIG = {
        "host": "0.0.0.0",
        "port": "3000",
        "networking_fairness": "false",
        "seed": "42",
        "criterion": "ce",
        "optimizer": "sgd",
        "learning_rate": "0.001",
        "momentum": "0.0",
        "local_epochs": "5",
        "global_epochs": "300",
        "device": "cuda",
        "min_require_size": "10",
        "total_clients": "100",
        "clients_per_round": "10",
        "delete_fraction_of_data": "false",
        "model_path": "./.models",
        "dataset_path": "./.datasets",
        "server_model_aggregation": "false",
        "split_strategy": "layer_name",
        "use_additional_epoch": "false",
        "g_measure_frequency": "10",
        "enable_g_measurement": "false",
        "use_cumulative_usage": "false",
        "usage_decay_factor": "0.95",
        "use_fresh_scoring": "false",
        "freshness_decay_rate": "0.5",
        # Drift Measurement (SCAFFOLD-style)
        "enable_drift_measurement": "true",
        "drift_sample_interval": "1",
        # In-round evaluation (training accuracy during training)
        "enable_inround_evaluation": "false",
    }

    # USFL feature groups
    USFL_OPTIONS = {
        "sfl_iid": {
            "distributer": "uniform",
            "method": "sfl",
            "selector": "uniform",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18_flex",
            "batch_size": "50",
            #"labels_per_client": "2",
            #"dirichlet_alpha": "0.3",
            "gradient_shuffle": "false",
            "gradient_shuffle_strategy": "random",
            "use_dynamic_batch_scheduler": "false",
            "use_variance_g": "true",
            "balancing_strategy": "target",
            "balancing_target": "mean",
            "split_layer": "layer2",
            # G Measurement options:
            # - "single": First batch only (default)
            # - "k_batch": First K batches (set g_measurement_k)
            # - "accumulated": All batches in round
            #"g_measurement_mode": "single",
            #"g_measurement_k": "5",  # Only used when g_measurement_mode is "k_batch"
        },
        "sfl_noniid": {
            "distributer": "shard_dirichlet",
            "method": "sfl",
            "selector": "uniform",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18_flex",
            "batch_size": "50",
            "labels_per_client": "2",
            "dirichlet_alpha": "0.3",
            "gradient_shuffle": "false",
            "gradient_shuffle_strategy": "random",
            "use_dynamic_batch_scheduler": "false",
            "use_variance_g": "true",
            #"balancing_strategy": "target",
            #"balancing_target": "mean",
            "split_layer": "layer2",
            # G Measurement options:
            # - "single": First batch only (default)
            # - "k_batch": First K batches (set g_measurement_k)
            # - "accumulated": All batches in round
            #"g_measurement_mode": "single",
            #"g_measurement_k": "5",  # Only used when g_measurement_mode is "k_batch"
        },
        "usfl": {
            "distributer": "shard_dirichlet",
            "method": "usfl",
            "selector": "usfl",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18_flex",
            "batch_size": "500",
            "labels_per_client": "2",
            "dirichlet_alpha": "0.3",
            "gradient_shuffle": "true",
            "gradient_shuffle_strategy": "random",
            "use_dynamic_batch_scheduler": "true",
            #"use_variance_g": "true",
            "balancing_strategy": "target",
            "balancing_target": "mean",
            "split_layer": "layer2",
            # G Measurement options:
            # - "single": First batch only (default)
            # - "k_batch": First K batches (set g_measurement_k)
            # - "accumulated": All batches in round
            #"g_measurement_mode": "single",
            #"g_measurement_k": "5",  # Only used when g_measurement_mode is "k_batch"
        },
        "sfl_full_10outof10": {
            "distributer": "shard_dirichlet",
            "method": "sfl",
            "selector": "uniform",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18_flex",
            "batch_size": "50",
            "labels_per_client": "2",
            "dirichlet_alpha": "0.3",
            "gradient_shuffle": "false",
            "gradient_shuffle_strategy": "random",
            "use_dynamic_batch_scheduler": "false",
            "use_variance_g": "true",
            "total_clients": "10",
            #"balancing_strategy": "target",
            #"balancing_target": "mean",
            "split_layer": "layer2",
            # G Measurement options:
            # - "single": First batch only (default)
            # - "k_batch": First K batches (set g_measurement_k)
            # - "accumulated": All batches in round
            #"g_measurement_mode": "single",
            #"g_measurement_k": "5",  # Only used when g_measurement_mode is "k_batch"
        },
        "fedcbs": {
            "method": "sfl",
            "selector": "fedcbs",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18",
            "batch_size": "50",
            "labels_per_client": "2",
            "dirichlet_alpha": "0.3",
            "gradient_shuffle": "false",
            "gradient_shuffle_strategy": "random",
            "use_dynamic_batch_scheduler": "false",
            "use_variance_g": "true",
            "balancing_strategy": "target",
            "balancing_target": "mean",
            "split_layer": "layer1.1.bn2",
            # G Measurement options:
            # - "single": First batch only (default)
            # - "k_batch": First K batches (set g_measurement_k)
            # - "accumulated": All batches in round
            #"g_measurement_mode": "single",
            #"g_measurement_k": "5",  # Only used when g_measurement_mode is "k_batch"
        },
        "mix2sfl": {
            "method": "mix2sfl",
            "selector": "uniform",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18",
            "batch_size": "50",
            "labels_per_client": "2",
            "dirichlet_alpha": "0.3",
            "gradient_shuffle": "false",
            "gradient_shuffle_strategy": "random",
            "use_dynamic_batch_scheduler": "false",
            "use_variance_g": "true",
            "balancing_strategy": "target",
            "balancing_target": "mean",
            "split_layer": "layer1.1.bn2",
            # G Measurement options:
            # - "single": First batch only (default)
            # - "k_batch": First K batches (set g_measurement_k)
            # - "accumulated": All batches in round
            #"g_measurement_mode": "single",
            #"g_measurement_k": "5",  # Only used when g_measurement_mode is "k_batch"
        },
        "scaffold": {
            "method": "scaffold_sfl",
            "selector": "uniform",
            "aggregator": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18",
            "batch_size": "50",
            "labels_per_client": "2",
            "dirichlet_alpha": "0.3",
            "gradient_shuffle": "false",
            "gradient_shuffle_strategy": "random",
            "use_dynamic_batch_scheduler": "false",
            "use_variance_g": "true",
            "balancing_strategy": "target",
            "balancing_target": "mean",
            "split_layer": "layer1.1.bn2",
            # G Measurement options:
            # - "single": First batch only (default)
            # - "k_batch": First K batches (set g_measurement_k)
            # - "accumulated": All batches in round
            #"g_measurement_mode": "single",
            #"g_measurement_k": "5",  # Only used when g_measurement_mode is "k_batch"
        },
    }

    # Use only USFL_OPTIONS (no parameter sweeps)
    combinations = list(USFL_OPTIONS.keys())

    total_workloads = len(combinations)

    print(f"\n{'=' * 70}")
    print(f"USFL Experiment Runner")
    print(f"{'=' * 70}")
    print(f"Total workloads: {total_workloads}")
    print(f"  - USFL options: {len(USFL_OPTIONS)}")
    print(f"{'=' * 70}\n")

    # ===== WORKLOAD RANGE SELECTION =====
    # Uncomment to run specific range:
    # combinations = combinations[:10]  # First 10 only
    # combinations = combinations[100:110]  # Workloads 101-110
    # combinations = combinations[500:]  # Skip first 500, run 501-900

    # Or set start/end manually:
    START_INDEX = 3  # Start from workload N (0-indexed)
    END_INDEX = 4  # End at workload N (None = run to end)

    original_total = len(combinations)
    if START_INDEX > 0 or END_INDEX is not None:
        combinations = combinations[START_INDEX:END_INDEX]
        end_display = END_INDEX if END_INDEX else original_total
        print(
            f"\n📍 Running workloads {START_INDEX + 1} to {end_display} (of {original_total} total)\n"
        )

    # ===== TIMEOUT SETTINGS =====
    # Set timeout per workload (in seconds)
    # If a workload runs longer than this, it will be killed and skipped
    WORKLOAD_TIMEOUT = 100000  # 1 hour per workload (adjust as needed)
    # Set to None to disable timeout

    failed_workloads = []
    timeout_workloads = []

    for local_idx, usfl_option in enumerate(combinations, 1):
        # Calculate global index (accounting for START_INDEX offset)
        idx = START_INDEX + local_idx

        # Build workload configuration
        # Merge BASE_CONFIG with USFL option
        workload = {**BASE_CONFIG, **USFL_OPTIONS[usfl_option]}

        # Extract dataset/model from USFL option (if present) or use BASE_CONFIG
        dataset_name = workload.get("dataset", "unknown")
        model_name = workload.get("model", "unknown")
        split_layer = workload.get("split_layer", "unknown")

        # Create descriptive name
        workload_name = f"{model_name}-{dataset_name}-{split_layer}-usfl{usfl_option}"

        print(f"\n{'=' * 70}")
        print(
            f"Workload {idx}/{original_total} (local: {local_idx}/{len(combinations)})"
        )
        print(f"Name: {workload_name}")
        print(
            f"Config: model={model_name}, dataset={dataset_name}, split={split_layer}, usfl={usfl_option}"
        )
        print(f"{'=' * 70}\n")
        DATASET = workload.get("dataset", None)
        MODEL = workload.get("model", None)
        METHOD = workload.get("method", None)
        LOCAL_EPOCHS = workload.get("local_epochs", None)
        GLOBAL_EPOCHS = workload.get("global_epochs", None)
        DEVICE = workload.get("device", None)
        DISTRIBUTER = workload.get("distributer", None)
        SPLIT_STRATEGY = workload.get("split_strategy", None)
        SPLIT_LAYER = workload.get("split_layer", None)
        SPLIT_RATIO = workload.get("split_ratio", None)
        LABELS_PER_CLIENT = workload.get("labels_per_client", None)
        TOTAL_CLIENTS = workload.get("total_clients", None)
        CLIENTS_PER_ROUND = workload.get("clients_per_round", None)
        LEARNING_RATE = workload.get("learning_rate", None)
        SERVER_LEARNING_RATE = workload.get("server_learning_rate", None)
        MOMENTUM = workload.get("momentum", None)
        BATCH_SIZE = workload.get("batch_size", None)
        DIRICHLET_ALPHA = workload.get("dirichlet_alpha", None)
        PORT = workload.get("port", "1000")
        SERVER_MODEL_AGGREGATION = workload.get("server_model_aggregation", None)
        OPTIMIZER = workload.get("optimizer", None)
        DELETE_FRACTION_OF_DATA = workload.get("delete_fraction_of_data", None)
        SELECTOR = workload.get("selector", None)
        AGGREGATOR = workload.get("aggregator", None)
        GRADIENT_SHUFFLE = workload.get("gradient_shuffle", None)
        GRADIENT_SHUFFLE_STRATEGY = workload.get("gradient_shuffle_strategy", None)
        GRADIENT_SHUFFLE_TARGET = workload.get("gradient_shuffle_target", None)
        GRADIENT_AVERAGE_WEIGHT = workload.get("gradient_average_weight", None)
        ADAPTIVE_ALPHA_BETA = workload.get("adaptive_alpha_beta", None)
        USE_ADDITIONAL_EPOCH = workload.get("use_additional_epoch", None)
        USE_CUMULATIVE_USAGE = workload.get("use_cumulative_usage", None)
        USAGE_DECAY_FACTOR = workload.get("usage_decay_factor", None)
        NETWORKING_FAIRNESS = workload.get("networking_fairness", None)
        USE_DYNAMIC_BATCH_SCHEDULER = workload.get("use_dynamic_batch_scheduler", None)
        USE_VARIANCE_G = workload.get("use_variance_g", None)
        USE_FRESH_SCORING = workload.get("use_fresh_scoring", None)
        FRESHNESS_DECAY_RATE = workload.get("freshness_decay_rate", None)
        USE_DATA_REPLICATION = workload.get("use_data_replication", None)
        BALANCING_STRATEGY = workload.get("balancing_strategy", None)
        BALANCING_TARGET = workload.get("balancing_target", None)
        ENABLE_G_MEASUREMENT = workload.get("enable_g_measurement", None)
        G_MEASURE_FREQUENCY = workload.get("g_measure_frequency", None)
        ENABLE_CONCATENATION = workload.get("enable_concatenation", None)
        ENABLE_LOGIT_ADJUSTMENT = workload.get("enable_logit_adjustment", None)
        MIN_REQUIRE_SIZE = workload.get("min_require_size", None)
        G_MEASUREMENT_MODE = workload.get("g_measurement_mode", None)
        G_MEASUREMENT_K = workload.get("g_measurement_k", None)
        ENABLE_DRIFT_MEASUREMENT = workload.get("enable_drift_measurement", None)
        DRIFT_SAMPLE_INTERVAL = workload.get("drift_sample_interval", None)
        ENABLE_INROUND_EVALUATION = workload.get("enable_inround_evaluation", None)

        server_command = []

        if DATASET:
            server_command.extend(["-d", DATASET])
        if MODEL:
            server_command.extend(["-m", MODEL])
        if METHOD:
            server_command.extend(["-M", METHOD])
        if LOCAL_EPOCHS:
            server_command.extend(["-le", LOCAL_EPOCHS])
        if GLOBAL_EPOCHS:
            server_command.extend(["-gr", GLOBAL_EPOCHS])
        if DEVICE:
            # Use first specified GPU for server, or just "cuda" if using all GPUs
            if DEVICE == "cuda" and GPU_DEVICES is not None:
                server_command.extend(["-de", f"cuda:{GPU_DEVICES[0]}"])
            else:
                server_command.extend(["-de", DEVICE])
        if TOTAL_CLIENTS:
            server_command.extend(["-nc", TOTAL_CLIENTS])
        if CLIENTS_PER_ROUND:
            server_command.extend(["-ncpr", CLIENTS_PER_ROUND])
        if DISTRIBUTER:
            server_command.extend(["-distr", DISTRIBUTER])
        if LABELS_PER_CLIENT:
            server_command.extend(["-lpc", LABELS_PER_CLIENT])
        if SPLIT_STRATEGY:
            server_command.extend(["-ss", SPLIT_STRATEGY])
        if SPLIT_LAYER:
            server_command.extend(["-sl", SPLIT_LAYER])
        if SPLIT_RATIO:
            server_command.extend(["-sr", SPLIT_RATIO])
        if LEARNING_RATE:
            server_command.extend(["-lr", LEARNING_RATE])
        if SERVER_LEARNING_RATE:
            server_command.extend(["-slr", SERVER_LEARNING_RATE])
        if MOMENTUM:
            server_command.extend(["-mt", MOMENTUM])
        if BATCH_SIZE:
            server_command.extend(["-bs", BATCH_SIZE])
        if DIRICHLET_ALPHA:
            server_command.extend(["-diri-alpha", DIRICHLET_ALPHA])
        if MIN_REQUIRE_SIZE:
            server_command.extend(["-mrs", MIN_REQUIRE_SIZE])
        if PORT:
            server_command.extend(["-p", PORT])
        if SERVER_MODEL_AGGREGATION:
            server_command.extend(["-sma", SERVER_MODEL_AGGREGATION])
        if OPTIMIZER:
            server_command.extend(["-o", OPTIMIZER])
        if DELETE_FRACTION_OF_DATA:
            server_command.extend(["-df", DELETE_FRACTION_OF_DATA])
        if SELECTOR:
            server_command.extend(["-s", SELECTOR])
        if AGGREGATOR:
            server_command.extend(["-aggr", AGGREGATOR])
        if GRADIENT_SHUFFLE and GRADIENT_SHUFFLE.lower() == "true":
            server_command.extend(["-gs"])
        if GRADIENT_SHUFFLE_STRATEGY:
            server_command.extend(["-gss", GRADIENT_SHUFFLE_STRATEGY])
        if GRADIENT_SHUFFLE_TARGET:
            server_command.extend(["-gst", GRADIENT_SHUFFLE_TARGET])
        if GRADIENT_AVERAGE_WEIGHT:
            server_command.extend(["-gaw", GRADIENT_AVERAGE_WEIGHT])
        if ADAPTIVE_ALPHA_BETA:
            server_command.extend(["-aab", ADAPTIVE_ALPHA_BETA])
        if USE_ADDITIONAL_EPOCH and USE_ADDITIONAL_EPOCH.lower() == "true":
            server_command.extend(["-uae"])
        if USE_CUMULATIVE_USAGE and USE_CUMULATIVE_USAGE.lower() == "true":
            server_command.extend(["-ucu"])
        if USAGE_DECAY_FACTOR:
            server_command.extend(["-udf", USAGE_DECAY_FACTOR])
        if (
            USE_DYNAMIC_BATCH_SCHEDULER
            and USE_DYNAMIC_BATCH_SCHEDULER.lower() == "true"
        ):
            server_command.extend(["-udbs"])
        if USE_VARIANCE_G and USE_VARIANCE_G.lower() == "true":
            server_command.extend(["--use-variance-g"])
        if USE_FRESH_SCORING and USE_FRESH_SCORING.lower() == "true":
            server_command.extend(["-ufs"])
        if FRESHNESS_DECAY_RATE:
            server_command.extend(["-fdr", FRESHNESS_DECAY_RATE])
        if USE_DATA_REPLICATION and USE_DATA_REPLICATION.lower() == "true":
            server_command.extend(["-udr"])
        if BALANCING_STRATEGY:
            server_command.extend(["-bstrat", BALANCING_STRATEGY])
        if BALANCING_TARGET:
            server_command.extend(["-btarget", BALANCING_TARGET])
        if ENABLE_G_MEASUREMENT and ENABLE_G_MEASUREMENT.lower() == "true":
            server_command.extend(["--enable-g-measurement"])
        if G_MEASURE_FREQUENCY:
            server_command.extend(["--g-measure-frequency", G_MEASURE_FREQUENCY])
        if ENABLE_CONCATENATION and ENABLE_CONCATENATION.lower() == "true":
            server_command.extend(["-ec"])
        if ENABLE_LOGIT_ADJUSTMENT and ENABLE_LOGIT_ADJUSTMENT.lower() == "true":
            server_command.extend(["-ela"])
        if G_MEASUREMENT_MODE:
            server_command.extend(["--g-measurement-mode", G_MEASUREMENT_MODE])
        if G_MEASUREMENT_K:
            server_command.extend(["--g-measurement-k", str(G_MEASUREMENT_K)])
        if ENABLE_DRIFT_MEASUREMENT and ENABLE_DRIFT_MEASUREMENT.lower() == "true":
            server_command.extend(["--enable-drift-measurement"])
        if DRIFT_SAMPLE_INTERVAL:
            server_command.extend(["--drift-sample-interval", str(DRIFT_SAMPLE_INTERVAL)])
        if ENABLE_INROUND_EVALUATION and ENABLE_INROUND_EVALUATION.lower() == "true":
            server_command.extend(["--enable-inround-evaluation"])
        if NETWORKING_FAIRNESS:
            if NETWORKING_FAIRNESS.lower() == "true":
                server_command.extend(["-nf"])
            elif NETWORKING_FAIRNESS.lower() == "false":
                server_command.extend(["-nnf"])

        client_commands = []
        available_gpus = []
        if DEVICE == "cuda":
            if GPU_DEVICES is not None:
                # Use specified GPUs
                available_gpus = GPU_DEVICES
            else:
                # Use all available GPUs
                available_gpus = list(range(torch.cuda.device_count()))

        if METHOD != "cl":
            client_commands = [
                [
                    "-cid",
                    str(i),
                    "-d",
                    f"cuda:{available_gpus[i % len(available_gpus)]}" if DEVICE == "cuda" and available_gpus else DEVICE,
                    "-su",
                    f"localhost:{PORT}",
                ]
                for i in range(int(TOTAL_CLIENTS))
            ]

        try:
            if WORKLOAD_TIMEOUT:
                # Run with timeout
                async def run_with_timeout():
                    return await asyncio.wait_for(
                        run_simulation(server_command, client_commands),
                        timeout=WORKLOAD_TIMEOUT,
                    )

                asyncio.run(run_with_timeout())
            else:
                # Run without timeout
                asyncio.run(run_simulation(server_command, client_commands))

            print(f"\n✓ Workload {idx} completed successfully")

        except asyncio.TimeoutError:
            print(f"\n⏱️  TIMEOUT in workload {idx} after {WORKLOAD_TIMEOUT} seconds")
            print(f"   Skipping to next workload...")
            timeout_workloads.append(
                (idx, workload_name, f"Timeout after {WORKLOAD_TIMEOUT}s")
            )
            continue

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user at workload {idx}")
            print(f"   To resume from this point, set START_INDEX = {idx}")
            raise

        except Exception as e:
            print(f"\n❌ ERROR in workload {idx}: {e}")
            import traceback

            traceback.print_exc()
            failed_workloads.append((idx, workload_name, str(e)))
            continue

    # Summary
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total workloads attempted: {len(combinations)}")
    print(
        f"Completed: {len(combinations) - len(failed_workloads) - len(timeout_workloads)}"
    )
    print(f"Failed: {len(failed_workloads)}")
    print(f"Timeout: {len(timeout_workloads)}")

    if timeout_workloads:
        print(f"\nTimeout workloads:")
        for idx, name, error in timeout_workloads:
            print(f"  {idx}. {name}")
            print(f"     {error}")

    if failed_workloads:
        print(f"\nFailed workloads:")
        for idx, name, error in failed_workloads:
            print(f"  {idx}. {name}")
            print(f"     Error: {error[:100]}...")

    print(f"{'=' * 70}\n")
