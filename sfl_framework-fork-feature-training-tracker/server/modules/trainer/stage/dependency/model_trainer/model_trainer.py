from typing import TYPE_CHECKING

from .cl_model_trainer import CLModelTrainer

if TYPE_CHECKING:
    from modules.dataset.base_dataset import BaseDataset
    from server_args import Config
    from torch.nn import Module


def get_model_trainer(
    config: "Config",
    dataset: "BaseDataset",
    model: "Module",
    training_params: dict,
):
    model = model.to(config.device)

    if config.method in ["cl"]:
        return CLModelTrainer(config, dataset, model, training_params)
    else:
        raise ValueError(f"Invalid method (get_model_trainer): {config.method}")
