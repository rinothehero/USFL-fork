from typing import TYPE_CHECKING

from .ratio_strategy import RatioStrategy
from .name_strategy import NameStrategy

if TYPE_CHECKING:
    from server_args import Config


def get_strategy(config: "Config"):
    if config.split_strategy in ["ratio_param", "ratio_layer"]:
        return RatioStrategy(config)
    elif config.split_strategy == "layer_name":
        return NameStrategy(config)
    else:
        return None

