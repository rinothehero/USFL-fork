from .fedcbs_selector import FedCBSSelector
from .missing_class_selector import MissingClassSelector
from .uniform_selector import UniformSelector
from .usfl_selector import USFLSelector


def get_selector(config):
    if config.selector == "uniform":
        return UniformSelector()
    elif config.selector == "usfl":
        return USFLSelector(config)
    elif config.selector == "missing_class":
        return MissingClassSelector(config)
    elif config.selector == "fedcbs":
        return FedCBSSelector(config)
    else:
        raise ValueError(f"Selector {config.selector} not found")
