from typing import TYPE_CHECKING

from .distilbert_splitter import DistilBertSplitter
from .mobilenet_splitter import MobileNetSplitter
from .resnet_splitter import ResnetSplitter
from .strategy.strategy import get_strategy
from .vgg_spliter import VGGSplitter

if TYPE_CHECKING:
    from server_args import Config


def get_splitter(config: "Config"):
    strategy = get_strategy(config)

    if config.model == "distilbert":
        return DistilBertSplitter(config, strategy)
    elif config.model in ["vgg11", "tiny_vgg11", "alexnet", "alexnet_scala", "lenet"]:
        return VGGSplitter(config, strategy)
    elif config.model == "resnet18":
        return ResnetSplitter(config, strategy)
    elif config.model == "mobilenet":
        return MobileNetSplitter(config, strategy)
    else:
        raise ValueError(f"Unknown model name: {config.model}")
