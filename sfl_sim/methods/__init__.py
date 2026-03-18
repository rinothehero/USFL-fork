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
    if method == "usfl":
        from .usfl import USFLHook
        return USFLHook(config, trainer)
    if method == "gas":
        from .gas import GASHook
        return GASHook(config, trainer)
    if method == "scaffold":
        from .scaffold import SCAFFOLDHook
        return SCAFFOLDHook(config, trainer)
    if method == "mix2sfl":
        from .mix2sfl import Mix2SFLHook
        return Mix2SFLHook(config, trainer)
    raise ValueError(f"Unknown method: {method}")
