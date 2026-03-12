"""Method hook registry."""
from .base import BaseMethodHook


def get_method_hook(method: str, config, resources: dict) -> BaseMethodHook:
    """Factory: create the appropriate method hook."""
    # Lazy imports to avoid circular dependencies
    if method == "sfl":
        from .sfl import SFLHook
        return SFLHook(config, resources)
    elif method == "usfl":
        from .usfl import USFLHook
        return USFLHook(config, resources)
    elif method == "scaffold_sfl":
        from .scaffold import ScaffoldHook
        return ScaffoldHook(config, resources)
    elif method == "mix2sfl":
        from .mix2sfl import Mix2SFLHook
        return Mix2SFLHook(config, resources)
    elif method == "gas":
        from .gas import GASHook
        return GASHook(config, resources)
    elif method == "multisfl":
        from .multisfl import MultiSFLHook
        return MultiSFLHook(config, resources)
    else:
        raise ValueError(f"Unknown method: {method}. Available: sfl, usfl, scaffold_sfl, mix2sfl, gas, multisfl")
