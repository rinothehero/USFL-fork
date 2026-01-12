import asyncio
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from .base_model_trainer import BaseModelTrainer

if TYPE_CHECKING:
    from modules.dataset.base_dataset import BaseDataset
    from server_args import Config, ServerConfig
    from torch.nn import Module


class CLModelTrainer(BaseModelTrainer):
    def __init__(
        self,
        config: "Config",
        dataset: "BaseDataset",
        model: "Module",
        training_params: dict,
    ):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.training_params = training_params
        self.criterion = self.get_criterion(config)
        self.trainloader = self.dataset.get_trainloader()

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.config)

        for epoch in range(self.training_params["local_epochs"]):
            running_loss = 0.0
            running_corrects = 0
            total = 0
            total_labels = 0

            for batch in tqdm(self.trainloader, desc="Training Batches"):
                inputs, labels = batch
                total_labels += len(labels)
                inputs, labels = inputs.to(self.config.device), labels.to(
                    self.config.device
                )

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                await asyncio.to_thread(optimizer.zero_grad)
                await asyncio.to_thread(loss.backward)
                await asyncio.to_thread(optimizer.step)

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
            epoch_loss = running_loss / float(total)
            epoch_acc = running_corrects / float(total)

            print(
                f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}, Total labels = {total_labels}"
            )

    async def _train_glue_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.config)

        for epoch in range(self.training_params["local_epochs"]):
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for batch_idx, batch in enumerate(self.trainloader):
                inputs = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["label"].to(self.config.device)

                outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)

                await asyncio.to_thread(optimizer.zero_grad)
                await asyncio.to_thread(loss.backward)
                await asyncio.to_thread(optimizer.step)

                running_loss += loss.item()
                _, preds = torch.max(outputs.logits, 1)
                running_corrects += torch.sum(preds == labels)
                total += labels.size(0)

            epoch_loss = running_loss / float(total)
            epoch_acc = running_corrects / float(total)

            print(
                f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}"
            )

    async def train(self, params: dict = None):
        if self.config.dataset in [
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
