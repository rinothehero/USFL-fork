from typing import TYPE_CHECKING

from .cifar import CIFAR
from .glue import GLUEDataset
from .mnist import MNIST

if TYPE_CHECKING:
    from server_args import Config


def get_dataset(config: "Config"):
    if config.dataset == "cifar10" or config.dataset == "cifar100":
        return CIFAR(config)
    elif config.dataset == "mnist":
        return MNIST(config)
    elif config.dataset in [
        "cola",
        "sst2",
        "mrpc",
        "sts-b",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "wnli",
        "ax",
    ]:
        return GLUEDataset(config)
    elif config.dataset in ["mnist", "fmnist"]:
        return MNIST(config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")
