import asyncio
import copy
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from utils.log_utils import vprint, TQDM_DISABLED

from .base_model_trainer import BaseModelTrainer

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from torch.nn import Module


class FedProxModelTrainer(BaseModelTrainer):
    def __init__(
        self,
        config: "Config",
        server_config: "ServerConfig",
        dataset: "BaseDataset",
        model: "Module",
        training_params: dict,
    ):
        self.config = config
        self.server_config = server_config
        self.dataset = dataset
        self.model = model
        self.raw_model = copy.deepcopy(model)
        self.training_params = training_params
        self.criterion = self.get_criterion(server_config)
        self.trainloader = self.dataset.get_trainloader()

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)

        for epoch in range(self.training_params["local_epochs"]):
            running_loss = 0.0
            running_corrects = 0
            total = 0
            total_labels = 0

            for batch in tqdm(self.trainloader, desc="Training Batches", disable=TQDM_DISABLED):
                inputs, labels = batch
                total_labels += len(labels)
                inputs, labels = inputs.to(self.config.device), labels.to(
                    self.config.device
                )

                outputs = self.model(inputs)

                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.raw_model.parameters()):
                    proximal_term += (w - w_t).norm(2) ** 2

                loss = (
                    self.criterion(outputs, labels)
                    + (self.server_config.prox_mu / 2) * proximal_term
                )

                await asyncio.to_thread(optimizer.zero_grad)
                await asyncio.to_thread(loss.backward)
                await asyncio.to_thread(optimizer.step)

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
            epoch_loss = running_loss / float(total)
            epoch_acc = running_corrects / float(total)

            vprint(
                f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}, Total labels = {total_labels}", 2
            )

    async def _train_glue_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)

        for epoch in range(self.training_params["local_epochs"]):
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for batch_idx, batch in enumerate(self.trainloader):
                inputs = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["label"].to(self.config.device)

                outputs = self.model(input_ids=inputs, attention_mask=attention_mask)

                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.raw_model.parameters()):
                    proximal_term += (w - w_t).norm(2) ** 2

                loss = (
                    self.criterion(outputs.logits, labels)
                    + (self.server_config.prox_mu / 2) * proximal_term
                )

                await asyncio.to_thread(optimizer.zero_grad)
                await asyncio.to_thread(loss.backward)
                await asyncio.to_thread(optimizer.step)

                running_loss += loss.item()
                _, preds = torch.max(outputs.logits, 1)
                running_corrects += torch.sum(preds == labels)
                total += labels.size(0)

            epoch_loss = running_loss / float(total)
            epoch_acc = running_corrects / float(total)

            vprint(
                f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}", 2
            )

    async def train(self, params: dict = None):
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
