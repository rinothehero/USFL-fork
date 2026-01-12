from typing import TYPE_CHECKING

from .alexnet import AlexNet
from .distilbert import DistilBert
from .lenet import LeNet
from .mobilenet import MobileNet
from .resnet import ResNet
from .vgg11 import VGG11

if TYPE_CHECKING:
    from server_args import Config


def get_model(config: "Config", num_classes: int):
    if config.model == "vgg11" or config.model == "tiny_vgg11":
        return VGG11(config, num_classes)
    elif config.model == "distilbert":
        return DistilBert(config, num_classes)
    elif config.model == "resnet18":
        return ResNet(config, num_classes)
    elif config.model == "alexnet" or config.model == "alexnet_scala":
        return AlexNet(config, num_classes)
    elif config.model == "mobilenet":
        return MobileNet(config, num_classes)
    elif config.model == "lenet":
        return LeNet(config, num_classes)
    else:
        raise ValueError(f"Model {config.model} not found")
