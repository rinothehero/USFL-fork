import time
from typing import TYPE_CHECKING

from tqdm import tqdm

from .base_model_trainer import BaseModelTrainer
from .propagator.propagator import get_propagator

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI
    from torch.nn import Module


class ScalaModelTrainer(BaseModelTrainer):
    def __init__(
        self,
        config: "Config",
        server_config: "ServerConfig",
        dataset: "BaseDataset",
        model: "Module",
        training_params: dict,
        api: "CommonAPI",
    ):
        self.config = config
        self.server_config = server_config
        self.dataset = dataset
        self.model = model
        self.training_params = training_params
        self.api = api

        self.criterion = self.get_criterion(server_config)
        self.trainloader = self.dataset.get_trainloader()

        self.optimizer = None
        self.propagator = get_propagator(server_config, config, model)

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)
        iterations = params["iterations"]

        for epoch in range(self.training_params["local_epochs"]):
            total_labels = 0

            for batch_idx, batch in enumerate(
                tqdm(self.trainloader, desc="Training Batches", total=iterations)
            ):
                if batch_idx >= iterations:
                    break

                inputs, labels = batch
                total_labels += len(labels)
                inputs, labels = inputs.to(self.config.device), labels.to(
                    self.config.device
                )

                optimizer.zero_grad()

                outputs = self.propagator.forward(inputs)
                await self.api.submit_activations(
                    {
                        "outputs": outputs,
                        "labels": labels,
                        "model_index": self.training_params["model_index"],
                        "client_id": self.config.client_id,
                    },
                    signiture=self.training_params["signiture"],
                )
                gradients, model_index = await self.api.wait_for_gradients()

                self.propagator.backward(gradients)
                optimizer.step()

    async def _train_glue_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)
        iterations = params["iterations"]

        for epoch in range(self.training_params["local_epochs"]):
            for batch_idx, batch in enumerate(
                tqdm(self.trainloader, desc="Training Batches", total=iterations)
            ):
                if batch_idx >= iterations:
                    break

                inputs = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["label"].to(self.config.device)

                optimizer.zero_grad()

                outputs = self.propagator.forward(
                    inputs, {"attention_mask": attention_mask}
                )
                await self.api.submit_activations(
                    {
                        "outputs": outputs,
                        "labels": labels,
                        "model_index": self.training_params["model_index"],
                        "attention_mask": attention_mask,
                        "client_id": self.config.client_id,
                    },
                    signiture=self.training_params["signiture"],
                )
                gradients, model_index = await self.api.wait_for_gradients()

                self.propagator.backward(gradients)
                optimizer.step()

    async def train(self, params: dict):
        if self.server_config.dataset in [
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
        ]:
            await self._train_glue_dataset(params)
        else:
            await self._train_default_dataset(params)
