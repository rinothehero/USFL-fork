from typing import TYPE_CHECKING

from .deit_splitter import DeiTSplitter
from .distilbert_splitter import DistilBertSplitter
from .flexible_resnet_splitter import FlexibleResnetSplitter
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
    elif config.model in ["resnet18", "resnet18_cifar"]:
        return ResnetSplitter(config, strategy)
    elif config.model == "resnet18_flex":
        # FlexibleResnetSplitter: returns pre-built client/server models
        # Supports layer boundary splits (layer2, layer3, layer4)
        return FlexibleResnetSplitter(config)
    elif config.model == "deit_s":
        return DeiTSplitter(config)
    elif config.model == "mobilenet":
        return MobileNetSplitter(config, strategy)
    else:
        raise ValueError(f"Unknown model name: {config.model}")
