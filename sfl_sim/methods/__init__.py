"""Method hook registry."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseMethodHook


def get_method_hook(method: str, config, trainer) -> "BaseMethodHook":
    """Create the appropriate method hook."""
    if method == "sfl":
        from .sfl import SFLHook
        return SFLHook(config, trainer)
    # Phase 2+: usfl, scaffold, mix2sfl, gas, multisfl
    raise ValueError(f"Unknown method: {method}")
