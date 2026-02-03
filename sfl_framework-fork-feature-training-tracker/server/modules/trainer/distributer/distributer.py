from typing import TYPE_CHECKING

from .dirichlet_distributer import DirichletDistributer
from .label_dirichlet_distributer import LabelDirichletDistributer
from .label_distributer import LabelDistributer
from .shard_dirichlet_distributer import ShardDirichletDistributer
from .uniform_distributer import UniformDistributer

if TYPE_CHECKING:
    from server_args import Config


def get_distributer(config: "Config"):
    if config.distributer == "uniform" or config.distributer == "iid":
        return UniformDistributer(config)
    elif config.distributer == "dirichlet":
        return DirichletDistributer(config)
    elif config.distributer == "label":
        return LabelDistributer(config)
    elif config.distributer == "label_dirichlet":
        return LabelDirichletDistributer(config)
    elif config.distributer == "shard_dirichlet":
        return ShardDirichletDistributer(config)
    else:
        raise ValueError(f"Unknown distributer {config.distributer}")

