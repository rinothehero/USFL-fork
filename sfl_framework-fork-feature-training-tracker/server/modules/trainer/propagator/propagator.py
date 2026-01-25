from typing import TYPE_CHECKING

from .distilbert_propagator import DistilbertPropagator
from .flexible_resnet_propagator import FlexibleResnetPropagator
from .mobilenet_propagator import MobileNetPropagator
from .resnet_propagator import ResnetPropagator
from .vgg_propagator import VGGPropagator

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


def get_propagator(config: "Config", model: "Module"):
    if config.model in ["resnet18", "resnet18_cifar"]:
        return ResnetPropagator(model, config)
    elif config.model == "resnet18_flex":
        # FlexibleResnetPropagator: handles both Tensor and Tuple inputs
        # Supports layer boundary splits (layer2, layer3, layer4)
        return FlexibleResnetPropagator(model, config)
    elif config.model in ["vgg11", "tiny_vgg11", "lenet", "alexnet"]:
        return VGGPropagator(model, config)
    elif config.model == "mobilenet":
        return MobileNetPropagator(model, config)
    elif config.model == "distilbert":
        return DistilbertPropagator(model)
    else:
        raise ValueError(f"Unknown model {config.model}")
