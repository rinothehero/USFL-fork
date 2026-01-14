from typing import TYPE_CHECKING

from .distilbert_propagator import DistilbertPropagator
from .resnet_propagator import ResnetPropagator
from .vgg_propagator import VGGPropagator

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from torch.nn import Module


def get_propagator(server_config: "ServerConfig", config: "Config", model: "Module"):
    if server_config.model in ["resnet18", "resnet18_cifar"]:
        return ResnetPropagator(model, config)
    elif server_config.model in [
        "vgg11",
        "tiny_vgg11",
        "mobilenet",
        "lenet",
        "alexnet",
    ]:
        return VGGPropagator(model, config)
    elif server_config.model == "distilbert":
        return DistilbertPropagator(model, config)
    else:
        raise ValueError(f"Unknown model {server_config.model}")
