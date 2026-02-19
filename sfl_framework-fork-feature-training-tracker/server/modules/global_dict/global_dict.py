import json
import os
import time
from collections import deque
from typing import TYPE_CHECKING

from .model_queue.model_queue import ModelQueue

if TYPE_CHECKING:
    from server_args import Config


class GlobalDict:
    def __init__(
        self,
        config: "Config",
    ):
        self.config = config
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.round = 0

        self.global_dict = {
            "config": {k: v for k, v in config.__dict__.items() if k != "mask_ids"},
            "metric": {
                round_index: [] for round_index in range(0, config.global_round + 1)
            },
            "activation_queue": deque(),
            "model_queue": ModelQueue(),
            "waiting_clients": {},
            "cumulative_usage": {},
        }

    def get(self, key: str):
        if key not in self.global_dict:
            raise KeyError(f"Key '{key}' not found in global_dict")
        return self.global_dict[key]

    def set(self, key: str, value):
        if key in ["config", "metric"]:
            raise KeyError(f"Key '{key}' is reserved by system.")

        self.global_dict[key] = value

    def set_round(self, round_number):
        self.round = round_number

    def add_event(self, event_name: str, params: dict = {}):
        round_metric = self.global_dict["metric"][self.round]

        round_metric.append(
            {"timestamp": time.time(), "event": event_name, "params": params}
        )

    def set_waiting_clients(self, client_id: int, waiting: bool):
        self.global_dict["waiting_clients"][client_id] = waiting

    def get_all_waiting_clients(self):
        return [
            client_id
            for client_id, waiting in self.global_dict["waiting_clients"].items()
            if waiting
        ]

    def get_waiting_clients_count(self):
        return sum(self.global_dict["waiting_clients"].values())

    def remove_waiting_client(self, client_id: int):
        del self.global_dict["waiting_clients"][client_id]

    def save_metric(self):
        # Build comprehensive filename with all relevant configs
        parts = [
            f"result-{self.config.method}",
            f"{self.config.model}",
            f"{self.config.dataset}",
            f"dist-{self.config.distributer}",
        ]

        # Add dirichlet_alpha if using dirichlet
        if self.config.distributer in [
            "dirichlet",
            "label_dirichlet",
            "shard_dirichlet",
        ]:
            parts.append(f"alpha-{self.config.dirichlet_alpha}")

        # Add selector and aggregator
        parts.append(f"sel-{self.config.selector}")
        parts.append(f"agg-{self.config.aggregator}")

        # Add split info for SFL methods
        if self.config.method in ["sfl", "sfl-u", "usfl", "scala", "mix2sfl"]:
            split_ratio_str = "-".join(str(r) for r in self.config.split_ratio)
            parts.append(f"split-{split_ratio_str}")

        # Add batch size
        parts.append(f"bs-{self.config.batch_size}")

        # Add USFL-specific features
        if self.config.method == "usfl":
            # Gradient Shuffle
            if getattr(self.config, "gradient_shuffle", False):
                parts.append("gradshuf")
                gradient_shuffle_strategy = getattr(
                    self.config, "gradient_shuffle_strategy", "default"
                )
                parts.append(f"gradshuf-{gradient_shuffle_strategy}")
            # Dynamic batch scheduler
            if getattr(self.config, "use_dynamic_batch_scheduler", False):
                parts.append("dbs")
            if getattr(self.config, "use_additional_epoch", False):
                parts.append("ae")
            # Cumulative usage
            if getattr(self.config, "use_cumulative_usage", False):
                parts.append("ucu")
                udf = getattr(self.config, "usage_decay_factor", 0.99)
                parts.append(f"udf-{udf}")

            # Fresh scoring
            if getattr(self.config, "use_fresh_scoring", False):
                parts.append("fresh")
                fdr = getattr(self.config, "freshness_decay_rate", 0.5)
                parts.append(f"fdr-{fdr}")

        # Add timestamp at the end
        parts.append(self.timestamp)

        filename = "-".join(parts) + ".json"

        # Use result_output_dir if configured, otherwise save to CWD
        result_dir = getattr(self.config, "result_output_dir", "")
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            filename = os.path.join(result_dir, filename)

        with open(filename, "w") as f:
            g_measurements = []
            drift_measurements = []
            for round_idx, events in self.global_dict["metric"].items():
                for event in events:
                    if event.get("event") == "G_MEASUREMENT":
                        g_measurements.append(
                            {
                                "round": round_idx,
                                "timestamp": event.get("timestamp"),
                                "params": event.get("params"),
                            }
                        )
                    elif event.get("event") == "DRIFT_MEASUREMENT":
                        drift_measurements.append(
                            {
                                "round": round_idx,
                                "timestamp": event.get("timestamp"),
                                "params": event.get("params"),
                            }
                        )

            # Build drift_history in same format as GAS/MultiSFL for consistency
            drift_history = None
            if drift_measurements:
                drift_history = {
                    # Client metrics (full names)
                    "G_drift_client": [m["params"].get("G_drift_client", 0.0) for m in drift_measurements],
                    "G_drift_client_stepweighted": [m["params"].get("G_drift_client_stepweighted") for m in drift_measurements],
                    "G_end_client": [m["params"].get("G_end_client", 0.0) for m in drift_measurements],
                    "G_end_client_weighted": [m["params"].get("G_end_client_weighted") for m in drift_measurements],
                    "G_drift_norm_client": [m["params"].get("G_drift_norm_client", 0.0) for m in drift_measurements],
                    "delta_client_norm_sq": [m["params"].get("delta_client_norm_sq", 0.0) for m in drift_measurements],
                    "D_dir_client_weighted": [m["params"].get("D_dir_client_weighted") for m in drift_measurements],
                    "D_rel_client_weighted": [m["params"].get("D_rel_client_weighted") for m in drift_measurements],
                    # Server metrics
                    "G_drift_server": [m["params"].get("G_drift_server", 0.0) for m in drift_measurements],
                    "G_end_server": [m["params"].get("G_end_server", 0.0) for m in drift_measurements],
                    "G_drift_norm_server": [m["params"].get("G_drift_norm_server", 0.0) for m in drift_measurements],
                    "delta_server_norm_sq": [m["params"].get("delta_server_norm_sq", 0.0) for m in drift_measurements],
                    # Combined
                    "G_drift_total": [m["params"].get("G_drift_total", 0.0) for m in drift_measurements],
                    "G_end_total": [m["params"].get("G_end_total", 0.0) for m in drift_measurements],
                    # Legacy compatibility (client-only aliases)
                    "G_drift": [m["params"].get("G_drift", 0.0) for m in drift_measurements],
                    "G_end": [m["params"].get("G_end", 0.0) for m in drift_measurements],
                    "G_drift_norm": [m["params"].get("G_drift_norm", 0.0) for m in drift_measurements],
                    "delta_global_norm_sq": [m["params"].get("delta_global_norm_sq", 0.0) for m in drift_measurements],
                    # Update alignment
                    "A_cos": [m["params"].get("A_cos") for m in drift_measurements],
                    "M_norm": [m["params"].get("M_norm") for m in drift_measurements],
                    "n_valid_alignment": [m["params"].get("n_valid_alignment") for m in drift_measurements],
                    # Experiment A
                    "expA_A_c_ratio": [
                        (m["params"].get("experiment_a") or {}).get("A_c_ratio")
                        for m in drift_measurements
                    ],
                    "expA_A_c_rel": [
                        (m["params"].get("experiment_a") or {}).get("A_c_rel")
                        for m in drift_measurements
                    ],
                    "expA_B_c": [
                        (m["params"].get("experiment_a") or {}).get("B_c")
                        for m in drift_measurements
                    ],
                    "expA_B_c_vs_sfl_iid_mu": [
                        (m["params"].get("experiment_a") or {}).get(
                            "B_c_vs_sfl_iid_mu"
                        )
                        for m in drift_measurements
                    ],
                    "expA_cos_c_vs_sfl_iid_mu": [
                        (m["params"].get("experiment_a") or {}).get(
                            "cos_c_vs_sfl_iid_mu"
                        )
                        for m in drift_measurements
                    ],
                    "expA_sfl_iid_mu_round_available": [
                        (m["params"].get("experiment_a") or {}).get(
                            "sfl_iid_mu_round_available"
                        )
                        for m in drift_measurements
                    ],
                    "expA_C_c": [
                        (m["params"].get("experiment_a") or {}).get("C_c")
                        for m in drift_measurements
                    ],
                    "expA_C_c_per_client_probe": [
                        (m["params"].get("experiment_a") or {}).get(
                            "C_c_per_client_probe"
                        )
                        for m in drift_measurements
                    ],
                    "expA_B_s": [
                        (m["params"].get("experiment_a") or {}).get("B_s")
                        for m in drift_measurements
                    ],
                    "expA_m2_c": [
                        (m["params"].get("experiment_a") or {}).get("m2_c")
                        for m in drift_measurements
                    ],
                    "expA_u2_c": [
                        (m["params"].get("experiment_a") or {}).get("u2_c")
                        for m in drift_measurements
                    ],
                    "expA_var_c": [
                        (m["params"].get("experiment_a") or {}).get("var_c")
                        for m in drift_measurements
                    ],
                    "expA_server_mag_per_step": [
                        (m["params"].get("experiment_a") or {}).get(
                            "server_mag_per_step"
                        )
                        for m in drift_measurements
                    ],
                    "expA_server_mag_per_step_sq": [
                        (m["params"].get("experiment_a") or {}).get(
                            "server_mag_per_step_sq"
                        )
                        for m in drift_measurements
                    ],
                    "experiment_a": [
                        (m["params"].get("experiment_a") or {})
                        for m in drift_measurements
                    ],
                    # Per-round details
                    "per_round": [m["params"] for m in drift_measurements],
                }

            result = {
                "config": self.global_dict["config"],
                "metric": self.global_dict["metric"],
                "g_measurements": g_measurements,
                "drift_measurements": drift_measurements,
                "drift_history": drift_history,
            }

            json.dump(result, f, indent=4)
