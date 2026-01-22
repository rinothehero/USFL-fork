from typing import TYPE_CHECKING

from .fedprox_model_trainer import FedProxModelTrainer
from .fitfl_model_trainer import FitFLModelTrainer
from .fl_model_trainer import FLModelTrainer
from .scala_model_trainer import ScalaModelTrainer
from .sfl_model_trainer import SFLModelTrainer
from .sflprox_model_trainer import SFLProxModelTrainer
from .usfl_model_trainer import USFLModelTrainer

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI
    from torch.nn import Module


def get_model_trainer(
    config: "Config",
    server_config: "ServerConfig",
    dataset: "BaseDataset",
    model: "Module",
    training_params: dict,
    api: "CommonAPI",
):
    model = model.to(config.device)

    if server_config.method in ["fl", "prunefl", "fedsparsify", "nestfl", "fedcbs"]:
        return FLModelTrainer(config, server_config, dataset, model, training_params)
    elif server_config.method == "fedprox":
        return FedProxModelTrainer(
            config, server_config, dataset, model, training_params
        )
    elif server_config.method == "sfl":
        return SFLModelTrainer(
            config, server_config, dataset, model, training_params, api
        )
    elif server_config.method == "mix2sfl":
        return USFLModelTrainer(
            config, server_config, dataset, model, training_params, api
        )
    elif server_config.method == "sflprox":
        return SFLProxModelTrainer(
            config, server_config, dataset, model, training_params, api
        )
    elif server_config.method == "scala":
        return ScalaModelTrainer(
            config, server_config, dataset, model, training_params, api
        )
    elif server_config.method == "usfl":
        return USFLModelTrainer(
            config, server_config, dataset, model, training_params, api
        )
    elif server_config.method == "fitfl":
        return FitFLModelTrainer(config, server_config, dataset, model, training_params)
    else:
        raise ValueError(f"Invalid method (get_model_trainer): {server_config.method}")
