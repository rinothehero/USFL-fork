"""
Batch spec generator for experiment deployment.

Reads experiment_configs/common.json + per-method configs and produces
a batch_spec.json consumable by batch_runner.py.

Usage:
    # CLI
    python -m experiment_core.generate_spec \
        --methods usfl gas \
        --gpu-map '{"usfl": 0, "gas": 1}' \
        --output batch_spec.json

    # Library
    from experiment_core.generate_spec import generate_batch_spec
    spec = generate_batch_spec(config_dir, methods, gpu_map)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Framework/method routing tables ──────────────────────────────────

FRAMEWORK_MAP = {
    "sfl": "sfl", "sfl_iid": "sfl", "usfl": "sfl", "scaffold": "sfl", "mix2sfl": "sfl",
    "gas": "gas", "multisfl": "multisfl",
}

METHOD_NAME_MAP = {
    "sfl": "sfl", "sfl_iid": "sfl", "usfl": "usfl", "scaffold": "scaffold_sfl", "mix2sfl": "mix2sfl",
    "gas": "gas", "multisfl": "multisfl",
}

# SFL adapter: these go as top-level framework_overrides (not sfl_args)
SFL_TOP_KEYS = {"selector", "aggregator"}

# SFL adapter: store_true flags (just need the flag, no value)
STORE_TRUE_FLAGS = {
    "gradient-shuffle", "use-dynamic-batch-scheduler",
    "use-additional-epoch", "use-cumulative-usage",
    "use-fresh-scoring", "use-data-replication",
    "scale-server-lr", "scale-client-grad",
}

# GAS: per-method config key → env var (only GAS-specific; shared params handled by adapter)
GAS_ENV_MAP = {
    "use_resnet_image_style": "GAS_USE_RESNET_IMAGE_STYLE",
    "use_sfl_oracle": "GAS_USE_SFL_ORACLE",
    "generate": "GAS_GENERATE",
    "sample_frequency": "GAS_SAMPLE_FREQUENCY",
    "v_test": "GAS_V_TEST",
    "v_test_frequency": "GAS_V_TEST_FREQUENCY",
    "accu_test_frequency": "GAS_ACCU_TEST_FREQUENCY",
}


# ── Core functions ───────────────────────────────────────────────────

def build_common_spec(common_path: Path) -> Dict[str, Any]:
    """Read common.json and build the unified common spec dict."""
    with open(common_path) as f:
        cr = json.load(f)

    return {
        "dataset": cr.get("dataset", "cifar10"),
        "model": cr.get("model", "resnet18_flex"),
        "seed": cr.get("seed", 42),
        "rounds": cr.get("rounds", 100),
        "total_clients": cr.get("total_clients", 100),
        "clients_per_round": cr.get("clients_per_round", 10),
        "local_epochs": cr.get("local_epochs", 5),
        "client_batch_size": cr.get("batch_size", 50),
        "server_batch_size": cr.get("server_batch_size", 500),
        "learning_rate": cr.get("learning_rate", 0.001),
        "momentum": cr.get("momentum", 0.0),
        "device": "cuda",
        "split_layer": cr.get("split_layer", "layer1.1.bn2"),
        "distribution": {
            "mode": "shard_dirichlet",
            "dirichlet_alpha": cr.get("alpha", 0.3),
            "labels_per_client": cr.get("labels_per_client", 2),
            "min_require_size": cr.get("min_require_size", 10),
        },
        "enable_g_measurement": cr.get("enable_g_measurement", False),
        "g_measurement_mode": cr.get("g_measurement_mode", "single"),
        "g_measurement_k": cr.get("g_measurement_k", 5),
        "use_variance_g": cr.get("use_variance_g", False),
        "g_measure_frequency": cr.get("g_measure_frequency", 10),
        "g_oracle_mode": cr.get("g_oracle_mode", "global"),
        "drift": {
            "enabled": cr.get("enable_drift_measurement", False),
            "sample_interval": cr.get("drift_sample_interval", 1),
        },
        "weight_decay": cr.get("weight_decay", 0.0),
        "clip_grad": cr.get("clip_grad", False),
        "clip_grad_max_norm": cr.get("clip_grad_max_norm", 10.0),
        "use_sfl_transform": cr.get("use_sfl_transform", False),
        "use_torchvision_init": cr.get("use_torchvision_init", False),
        "use_full_epochs": cr.get("use_full_epochs", False),
        "client_schedule_path": cr.get("client_schedule_path", ""),
        "probe_source": cr.get("probe_source", "test"),
        "probe_indices_path": cr.get("probe_indices_path", ""),
        "probe_num_samples": cr.get("probe_num_samples", 0),
        "probe_batch_size": cr.get("probe_batch_size", 0),
        "probe_max_batches": cr.get("probe_max_batches", 1),
        "probe_seed": cr.get("probe_seed", cr.get("seed", 42)),
        "probe_class_balanced": cr.get("probe_class_balanced", False),
        "probe_class_balanced_batches": cr.get(
            "probe_class_balanced_batches", False
        ),
        "save_mu_c": cr.get("save_mu_c", False),
        "reference_mu_c_path": cr.get("reference_mu_c_path", ""),
    }


def build_overrides(method: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert per-method config dict into framework-specific overrides."""
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_") and v is not None}
    fw = FRAMEWORK_MAP[method]

    if fw == "sfl":
        overrides: Dict[str, Any] = {}
        sfl_args: Dict[str, Any] = {}
        for k, v in cfg.items():
            if k in SFL_TOP_KEYS:
                overrides[k] = v
            else:
                cli_key = k.replace("_", "-")
                if isinstance(v, bool) and cli_key not in STORE_TRUE_FLAGS:
                    v = str(v).lower()
                sfl_args[cli_key] = v
        if sfl_args:
            overrides["sfl_args"] = sfl_args
        return overrides

    elif fw == "gas":
        overrides = {}
        gas_env: Dict[str, str] = {}
        for k, v in cfg.items():
            env_key = GAS_ENV_MAP.get(k)
            if env_key:
                gas_env[env_key] = str(v).lower() if isinstance(v, bool) else str(v)
        if gas_env:
            overrides["gas_env"] = gas_env
        return overrides

    else:  # multisfl
        return cfg


def generate_batch_spec(
    config_dir: Path,
    methods: List[str],
    gpu_map: Dict[str, Optional[int | str]],
    output_dir: Optional[str] = None,
    method_configs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate a batch_spec dict from config files.

    Args:
        config_dir: Directory containing common.json and per-method configs
        methods: List of method names (sfl, usfl, scaffold, gas, multisfl)
        gpu_map: Method name → GPU index (None = CPU)
        output_dir: Results output directory (auto-generated if None)
        method_configs_dir: Override directory for per-method configs
    """
    common_path = config_dir / "common.json"
    common = build_common_spec(common_path)

    if output_dir is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ds = common["dataset"]
        alpha = common["distribution"]["dirichlet_alpha"]
        rounds = common["rounds"]
        output_dir = f"results/{ds}_a{alpha}_r{rounds}_{ts}"

    mcdir = method_configs_dir or config_dir

    experiments = []
    for method in methods:
        cfg_path = mcdir / f"{method}.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                method_cfg = json.load(f)
        else:
            print(f"Warning: {cfg_path} not found, using empty config")
            method_cfg = {}

        exp = {
            "name": method,
            "framework": FRAMEWORK_MAP[method],
            "method": METHOD_NAME_MAP[method],
            "gpu": gpu_map.get(method),
            "overrides": build_overrides(method, method_cfg),
        }
        experiments.append(exp)

    return {
        "output_dir": output_dir,
        "common": common,
        "experiments": experiments,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate batch_spec.json from experiment configs"
    )
    parser.add_argument(
        "--config-dir", default="experiment_configs",
        help="Directory containing common.json and per-method configs",
    )
    parser.add_argument(
        "--methods", nargs="+", required=True,
        choices=list(FRAMEWORK_MAP.keys()),
        help="Methods to include in the batch spec",
    )
    parser.add_argument(
        "--gpu-map", type=json.loads, default="{}",
        help='Method→GPU mapping as JSON (e.g. \'{"usfl": 0, "gas": 1}\')',
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for batch_spec.json",
    )
    parser.add_argument(
        "--output-dir",
        help="Results output directory (auto-generated if omitted)",
    )
    parser.add_argument(
        "--method-configs-dir",
        help="Override directory for per-method configs (for temp-modified configs)",
    )
    parser.add_argument(
        "--copy-configs-to",
        help="Copy per-method configs to this directory for reference",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    mcdir = Path(args.method_configs_dir) if args.method_configs_dir else None

    spec = generate_batch_spec(
        config_dir=config_dir,
        methods=args.methods,
        gpu_map=args.gpu_map,
        output_dir=args.output_dir,
        method_configs_dir=mcdir,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Generated: {output_path}")

    # Copy per-method configs for reference
    if args.copy_configs_to:
        dst_dir = Path(args.copy_configs_to)
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_dir = mcdir or config_dir
        for method in args.methods:
            src = src_dir / f"{method}.json"
            if src.exists():
                shutil.copy2(src, dst_dir / f"{method}_config.json")


if __name__ == "__main__":
    main()
