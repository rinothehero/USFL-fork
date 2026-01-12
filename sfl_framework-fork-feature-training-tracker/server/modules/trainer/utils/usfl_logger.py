"""
USFL Feature Logger

Provides a dedicated logging system for USFL features that separates:
- Terminal: Essential training progress only
- Log File: Detailed feature operation logs
"""

import logging
import os
from pathlib import Path
from typing import Optional


class USFLLogger:
    """Logger for USFL feature details (freshness scoring, batch scheduling, etc.)"""

    _instance: Optional[logging.Logger] = None
    _initialized: bool = False
    _log_filename: Optional[str] = None

    @classmethod
    def initialize(cls, config, log_dir: str = "logs"):
        """
        Initialize the logger with a unique filename based on config and timestamp.

        Args:
            config: Server configuration object
            log_dir: Directory to store log files (default: "logs")
        """
        import time

        # Build unique log filename with timestamp and key configs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        parts = [
            f"usfl-{config.method}",
            f"{config.dataset}",
            f"bs-{config.batch_size}",
        ]

        # Add USFL-specific features
        if config.method == "usfl":
            if getattr(config, "use_dynamic_batch_scheduler", False):
                parts.append("dbs")
            if getattr(config, "use_fresh_scoring", False):
                parts.append("fresh")

        parts.append(timestamp)
        cls._log_filename = "-".join(parts) + ".log"

    @classmethod
    def get_logger(cls, log_dir: str = "logs") -> logging.Logger:
        """
        Get or create the USFL feature logger instance.

        Args:
            log_dir: Directory to store log files (default: "logs")

        Returns:
            Logger instance configured for file output
        """
        if cls._instance is not None:
            return cls._instance

        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Use unique filename if initialized, otherwise fallback to default
        log_filename = cls._log_filename if cls._log_filename else "usfl_features.log"
        log_path = os.path.join(log_dir, log_filename)

        # Create logger
        logger = logging.getLogger("usfl_features")
        logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if not logger.handlers:
            # File handler for detailed logs (each execution gets its own file)
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            # Formatter with timestamp and level
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        cls._instance = logger
        cls._initialized = True

        return logger

    @classmethod
    def log_round_separator(cls, round_number: int):
        """Log a visual separator for new round"""
        logger = cls.get_logger()
        logger.info("=" * 80)
        logger.info(f"ROUND {round_number} - USFL FEATURES")
        logger.info("=" * 80)

    @classmethod
    def log_freshness_selection(
        cls,
        round_number: int,
        selection_step: int,
        total_selections: int,
        candidate_scores: list,
        primary_candidates: list,
        freshness_scores: dict,
        selected_client: int,
    ):
        """Log detailed freshness scoring selection process"""
        logger = cls.get_logger()

        logger.info("")
        logger.info(f"--- Freshness Selection: Client {selection_step}/{total_selections} (Round {round_number}) ---")

        logger.info("1. Candidate temp_min scores:")
        for cs in candidate_scores:
            logger.info(f"   - Candidate {cs['id']}: temp_min = {cs['temp_min']}")

        logger.info(f"2. Primary candidates (top scores): {primary_candidates}")

        logger.info("3. Freshness scores for primary candidates:")
        for cand_id, score in freshness_scores.items():
            logger.info(f"   - Candidate {cand_id}: freshness_score = {score:.2f}")

        logger.info(f"4. Selected: Client {selected_client} (Highest Freshness: {freshness_scores.get(selected_client, 0):.2f})")

    @classmethod
    def log_batch_schedule(
        cls,
        round_number: int,
        client_data_usage: dict,
        total_iterations: int,
        schedule_by_client: dict,
    ):
        """Log dynamic batch scheduling details"""
        logger = cls.get_logger()

        logger.info("")
        logger.info(f"--- Dynamic Batch Scheduler (Round {round_number}) ---")

        logger.info("1. Planned data usage per client:")
        for client_id, usage in client_data_usage.items():
            logger.info(f"   - Client {client_id}: {usage} samples")

        logger.info(f"2. Total iterations: k = {total_iterations}")

        logger.info("3. Batch schedule per client per iteration:")
        for client_id, schedule in schedule_by_client.items():
            logger.info(f"   - Client {client_id}: {schedule}")

    @classmethod
    def log_cumulative_usage_init(cls):
        """Log cumulative usage initialization"""
        logger = cls.get_logger()
        logger.info("Initializing cumulative usage tracking (exponential bins)")

    @classmethod
    def log_cumulative_usage_update(cls, round_number: int, effective_epochs: int):
        """Log cumulative usage update"""
        logger = cls.get_logger()
        logger.info(f"Updating cumulative usage for round {round_number} (effective_epochs={effective_epochs})")

    @classmethod
    def log_warning(cls, message: str):
        """Log a warning message"""
        logger = cls.get_logger()
        logger.warning(message)

    @classmethod
    def log_debug(cls, message: str):
        """Log a debug message"""
        logger = cls.get_logger()
        logger.debug(message)
