from typing import TYPE_CHECKING

from .fedavg_aggregator import FedAvgAggregator
from .fitfl_aggregator import FitFLAggregator
from .usfl_aggregator import USFLAggregator

if TYPE_CHECKING:
    from server_args import Config


def get_aggregator(config: "Config"):
    if config.aggregator == "fedavg":
        return FedAvgAggregator()
    elif config.aggregator == "fitfl":
        return FitFLAggregator()
    elif config.aggregator == "usfl":
        return USFLAggregator()
    else:
        raise ValueError(f"Aggregator {config.aggregator} not found")
