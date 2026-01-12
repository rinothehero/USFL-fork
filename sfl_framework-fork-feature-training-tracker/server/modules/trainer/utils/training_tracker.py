"""
Training Data Tracker

Provides iteration-level tracking of:
- Per-client data usage (batch size, label distribution)
- Server-side total data per iteration
- Aggregation weights with justification
"""

import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


class TrainingTracker:
    """Tracks training data usage at iteration level and aggregation weights."""

    _instance: Optional[logging.Logger] = None
    _log_filename: Optional[str] = None
    _current_round: int = 0
    _iteration_count: int = 0

    @classmethod
    def initialize(cls, config, log_dir: str = "logs"):
        """
        Initialize the tracker with a unique filename.

        Args:
            config: Server configuration object
            log_dir: Directory to store log files
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cls._log_filename = f"tracking-{timestamp}.log"
        cls._current_round = 0
        cls._iteration_count = 0

        # Create logs directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_logger(cls, log_dir: str = "logs") -> logging.Logger:
        """Get or create the tracking logger instance."""
        if cls._instance is not None:
            return cls._instance

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_filename = cls._log_filename if cls._log_filename else f"tracking-{time.strftime('%Y%m%d-%H%M%S')}.log"
        log_path = os.path.join(log_dir, log_filename)

        logger = logging.getLogger("training_tracker")
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            # Simple format without timestamp for cleaner output
            formatter = logging.Formatter(fmt="%(message)s")
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        cls._instance = logger
        return logger

    @classmethod
    def start_round(cls, round_number: int):
        """Start tracking a new round."""
        cls._current_round = round_number
        cls._iteration_count = 0
        logger = cls.get_logger()
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"ROUND {round_number} - ITERATION TRACKING")
        logger.info("=" * 80)

    @classmethod
    def log_iteration_data(
        cls,
        round_number: int,
        iteration: int,
        client_data: Dict[int, Dict],
    ):
        """
        Log per-client data for each iteration.

        Args:
            round_number: Current round number
            iteration: Current iteration number
            client_data: {
                client_id: {
                    "batch_size": int,
                    "label_distribution": {label: count, ...}
                }
            }
        """
        logger = cls.get_logger()
        logger.info("")
        logger.info(f"--- Round {round_number}, Iteration {iteration} ---")
        logger.info("[CLIENT DATA]")

        for client_id, data in sorted(client_data.items(), key=lambda x: x[0]):
            batch_size = data.get("batch_size", 0)
            label_dist = data.get("label_distribution", {})
            # Sort labels numerically
            sorted_dist = {k: label_dist[k] for k in sorted(label_dist.keys(), key=lambda x: int(x))}
            logger.info(f"  Client {client_id}: {batch_size} samples | {sorted_dist}")

    @classmethod
    def log_server_iteration(
        cls,
        round_number: int,
        iteration: int,
        total_samples: int,
        label_distribution: Dict[str, int],
    ):
        """
        Log server-side total data per iteration.

        Args:
            round_number: Current round number
            iteration: Current iteration number
            total_samples: Total number of samples in this iteration
            label_distribution: {label: count, ...}
        """
        logger = cls.get_logger()
        # Sort labels numerically
        sorted_dist = {k: label_distribution[k] for k in sorted(label_distribution.keys(), key=lambda x: int(x))}
        logger.info(f"[SERVER TOTAL]")
        logger.info(f"  Total: {total_samples} samples | {sorted_dist}")

    @classmethod
    def log_aggregation_weights(
        cls,
        round_number: int,
        client_ids: List[int],
        client_weights: List[float],
        label_distributions: List[Dict[str, int]],
        total_per_label: Dict[str, int],
        augmented_label_distributions: List[Dict[str, int]] = None,
        augmented_total_per_label: Dict[str, int] = None,
    ):
        """
        Log aggregation weights with justification.

        Args:
            round_number: Current round number
            client_ids: List of client IDs
            client_weights: Normalized weights for each client
            label_distributions: Each client's ORIGINAL label distribution
            total_per_label: Global total per label (ORIGINAL)
            augmented_label_distributions: Each client's ACTUAL USED label distribution (optional)
            augmented_total_per_label: Global total per label for ACTUAL USED data (optional)
        """
        logger = cls.get_logger()
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"ROUND {round_number} - AGGREGATION")
        logger.info("=" * 80)

        logger.info("[WEIGHT CALCULATION]")
        logger.info(f"  Global label totals (Original): {dict(sorted(total_per_label.items(), key=lambda x: int(x[0])))}")
        if augmented_total_per_label:
            logger.info(f"  Global label totals (Actual):   {dict(sorted(augmented_total_per_label.items(), key=lambda x: int(x[0])))}")
        logger.info("")

        logger.info("[PER-CLIENT CONTRIBUTION]")
        for i, (cid, weight, label_dist) in enumerate(zip(client_ids, client_weights, label_distributions)):
            sorted_dist = {k: label_dist.get(k, 0) for k in sorted(label_dist.keys(), key=lambda x: int(x))}
            total_samples = sum(label_dist.values())
            logger.info(f"  Client {cid}:")
            logger.info(f"    - Original data: {total_samples} samples | {sorted_dist}")
            
            # Log augmented (actual used) data if available
            if augmented_label_distributions and i < len(augmented_label_distributions):
                aug_dist = augmented_label_distributions[i]
                if aug_dist:
                    sorted_aug_dist = {k: aug_dist.get(k, 0) for k in sorted(aug_dist.keys(), key=lambda x: int(x))}
                    aug_total = sum(aug_dist.values())
                    logger.info(f"    - Actual used:   {aug_total} samples | {sorted_aug_dist}")
            
            logger.info(f"    - Weight: {weight:.4f} ({weight*100:.2f}%)")

        logger.info("")
        logger.info("[FINAL RATIO]")
        ratio_parts = [f"{cid}:{w:.4f}" for cid, w in zip(client_ids, client_weights)]
        logger.info(f"  {' | '.join(ratio_parts)}")

        # Verify weights sum to 1
        weight_sum = sum(client_weights)
        logger.info(f"  (Sum: {weight_sum:.6f})")

    @classmethod
    def increment_iteration(cls) -> int:
        """Increment and return the current iteration count."""
        cls._iteration_count += 1
        return cls._iteration_count
