"""
Unified configuration: wraps the existing SFL Config dataclass and adds
method-specific fields for GAS and MultiSFL.

Does NOT duplicate the 200+ SFL Config fields — delegates to them.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _setup_sfl_path() -> Path:
    """Add SFL framework directories to sys.path (idempotent)."""
    repo_root = Path(__file__).resolve().parent.parent
    sfl_dir = repo_root / "sfl_framework-fork-feature-training-tracker"
    # Order matters! Insert client first, then server, then sfl_dir.
    # sys.path.insert(0, ...) pushes previous entries back, so final order is:
    #   sfl_dir → server → client → ...
    # This matches sfl_runner.py and ensures server's modules/ takes priority
    # over client's modules/ (client has a different get_dataset that expects mask_ids).
    for p in [str(sfl_dir / "client"), str(sfl_dir / "server"), str(sfl_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return sfl_dir


# Ensure SFL framework is importable
SFL_DIR = _setup_sfl_path()


@dataclass
class UnifiedConfig:
    """Wraps SFL Config and adds GAS + MultiSFL fields."""

    sfl_config: Any = None  # server_args.Config instance
    method: str = "sfl"

    # --- GAS-specific ---
    gas_generate_features: bool = False
    gas_sample_frequency: int = 5
    gas_logit_adjustment: bool = True
    gas_use_variance_g: bool = False
    gas_v_test: bool = False
    gas_diagonal_covariance: bool = True

    # --- MultiSFL-specific ---
    multisfl_branches: int = 3
    multisfl_alpha_master_pull: float = 0.1
    multisfl_p0: float = 0.3
    multisfl_p_min: float = 0.05
    multisfl_p_max: float = 0.8
    multisfl_p_update: str = "abs_ratio"
    multisfl_gamma: float = 0.5
    multisfl_replay_min_total: int = 10
    multisfl_max_assistant_trials: int = 5
    multisfl_local_steps: int = 5

    # --- Convenience property delegation ---
    @property
    def dataset(self) -> str:
        return self.sfl_config.dataset

    @property
    def model_name(self) -> str:
        return self.sfl_config.model

    @property
    def device(self) -> str:
        return self.sfl_config.device

    @property
    def global_round(self) -> int:
        return self.sfl_config.global_round

    @property
    def num_clients(self) -> int:
        return self.sfl_config.num_clients

    @property
    def num_clients_per_round(self) -> int:
        return self.sfl_config.num_clients_per_round

    @property
    def batch_size(self) -> int:
        return self.sfl_config.batch_size

    @property
    def local_epochs(self) -> int:
        return self.sfl_config.local_epochs

    @property
    def learning_rate(self) -> float:
        return self.sfl_config.learning_rate

    @property
    def server_learning_rate(self) -> float:
        return self.sfl_config.server_learning_rate

    @property
    def momentum(self) -> float:
        return self.sfl_config.momentum

    @property
    def optimizer_name(self) -> str:
        return self.sfl_config.optimizer

    @property
    def criterion_name(self) -> str:
        return self.sfl_config.criterion

    @property
    def seed(self) -> int:
        return self.sfl_config.seed

    @property
    def weight_decay(self) -> float:
        return getattr(self.sfl_config, "weight_decay", 0.0)

    @property
    def clip_grad(self) -> bool:
        return getattr(self.sfl_config, "clip_grad", False)

    @property
    def clip_grad_max_norm(self) -> float:
        return getattr(self.sfl_config, "clip_grad_max_norm", 1.0)

    @property
    def scale_server_lr(self) -> bool:
        return getattr(self.sfl_config, "scale_server_lr", False)

    @property
    def server_model_aggregation(self) -> bool:
        return self.sfl_config.server_model_aggregation


def parse_unified_config(
    method: str,
    sfl_args: List[str],
    extra_args: Optional[Dict[str, Any]] = None,
) -> UnifiedConfig:
    """Parse unified config from SFL-style CLI args + method-specific extras."""
    from server_args import parse_args as sfl_parse_args

    sfl_config = sfl_parse_args(sfl_args)
    config = UnifiedConfig(sfl_config=sfl_config, method=method)

    if extra_args:
        for k, v in extra_args.items():
            if hasattr(config, k) and not isinstance(
                getattr(type(config), k, None), property
            ):
                setattr(config, k, v)

    return config
