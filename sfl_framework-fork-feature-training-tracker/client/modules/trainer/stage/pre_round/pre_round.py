from typing import TYPE_CHECKING

import psutil

from utils.log_utils import vprint

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI


class PreRound:
    def __init__(
        self,
        config: "Config",
        server_config: "ServerConfig",
    ):
        self.config = config
        self.server_config = server_config

    async def notify_wait_for_training(self, api: "CommonAPI"):
        await api.notify_wait_for_training()
        vprint("Notified wait for training", 2)

    async def notify_client_information(
        self, api: "CommonAPI", filter: list = None, dataset: "BaseDataset" = None
    ):
        cpu_dict = {}
        memory_dict = {}
        dataset_dict = {}

        if "cpu" in filter:
            cpu_usage = psutil.cpu_percent()
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            cpu_dict = {
                "usage": cpu_usage,
                "core": 1 if cpu_count == 4 else 10,
                "frequency": cpu_freq._asdict(),
            }

        if "memory" in filter:
            memory = psutil.virtual_memory()
            memory_dict = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
            }

        if "dataset" in filter:
            dataset_size = len(self.server_config.mask_ids)

            if dataset is not None:
                label_distribution = dataset.trainset.get_label_distribution()

            dataset_dict = {
                "size": dataset_size,
                "label_distribution": (
                    label_distribution if dataset is not None else {}
                ),
            }

        await api.notify_client_information(
            {"cpu": cpu_dict, "memory": memory_dict, "dataset": dataset_dict}
        )

        vprint(f"Sent client information: {cpu_dict}, {memory_dict}, {dataset_dict}", 2)

    async def wait_for_start_round(self, api: "CommonAPI"):
        model, training_params = await api.wait_for_start_round()
        if model is not None:
            # print(f"Received start round signal: {training_params}")
            model.to(self.config.device)
        return model, training_params
