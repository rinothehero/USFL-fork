import torch

from typing import TYPE_CHECKING

from tqdm import tqdm

from .base_model_trainer import BaseModelTrainer
from .propagator.propagator import get_propagator

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI
    from torch.nn import Module


class USFLModelTrainer(BaseModelTrainer):
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

        self.optimizer = self.get_optimizer(server_config)
        self.propagator = get_propagator(server_config, config, model)

        # G Measurement: Accumulated client gradients for this round
        self.enable_g_measurement = getattr(
            server_config, "enable_g_measurement", False
        )
        self.g_measurement_mode = getattr(server_config, "g_measurement_mode", "single")
        self.g_measurement_k = getattr(server_config, "g_measurement_k", 5)
        self.accumulated_gradients: list = []  # List of gradient dicts
        self.gradient_weights: list = []  # Weights for each gradient (batch size)
        self.measurement_gradient = None  # For 1-step measurement protocol
        self.measurement_gradient_weight = None
        self._accumulated_grad_sum: dict = {}
        self._accumulated_grad_samples: int = 0
        self._client_batch_count: int = 0  # For k_batch mode

    def _reset_g_accumulation(self):
        self.accumulated_gradients = []
        self.gradient_weights = []
        self._accumulated_grad_sum = {}
        self._accumulated_grad_samples = 0
        self._client_batch_count = 0

    def _accumulate_client_grad(self, batch_size: int):
        if batch_size <= 0:
            return
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().cpu()
            if name not in self._accumulated_grad_sum:
                self._accumulated_grad_sum[name] = grad * batch_size
            else:
                self._accumulated_grad_sum[name] += grad * batch_size
        self._accumulated_grad_samples += batch_size

    def _finalize_g_accumulation(self):
        if self.g_measurement_mode not in ("accumulated", "k_batch"):
            return
        if self._accumulated_grad_samples <= 0:
            return
        avg_grad = {
            name: grad / float(self._accumulated_grad_samples)
            for name, grad in self._accumulated_grad_sum.items()
        }
        self.accumulated_gradients = [avg_grad]
        self.gradient_weights = [self._accumulated_grad_samples]
        if self.g_measurement_mode == "k_batch":
            print(f"[Client] K-batch finalized: {self._client_batch_count} batches, {self._accumulated_grad_samples} samples")

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.optimizer
        iterations = params["iterations"]
        schedule_dict = params.get("batch_schedule")
        my_schedule = (
            schedule_dict.get(str(self.config.client_id)) if schedule_dict else None
        )

        for epoch in range(self.training_params["local_epochs"]):
            print(f"Epoch: {epoch}, client_id: {self.config.client_id}")
            total_labels = 0
            completed_iterations = 0
            dataloader_iterator = iter(self.trainloader)

            progress_bar = tqdm(
                total=iterations, desc="Default Dataset Training", leave=True
            )

            if my_schedule:
                # Dynamic batch scheduler: Manually construct batches
                for batch_idx, current_batch_size in enumerate(my_schedule):
                    if batch_idx >= iterations:
                        break

                    if current_batch_size == 0:
                        # Send empty activation to avoid server deadlock
                        await self.api.submit_activations(
                            {
                                "outputs": torch.tensor([]).to(self.config.device),
                                "labels": torch.tensor([]).to(self.config.device),
                                "model_index": self.training_params["model_index"],
                                "client_id": self.config.client_id,
                            },
                            signiture=self.training_params["signiture"],
                        )
                        # Still need to wait for gradients (even if empty) to stay synchronized
                        gradients, model_index = await self.api.wait_for_gradients()

                        completed_iterations += 1
                        progress_bar.update(1)
                        continue

                    batch_data = []
                    batch_target = []
                    for _ in range(current_batch_size):
                        try:
                            data, target = next(dataloader_iterator)
                            batch_data.append(data)
                            batch_target.append(target)
                        except StopIteration:
                            # 데이터가 모두 소진되었으면 루프 종료
                            break

                    if not batch_data:
                        break

                    inputs = torch.cat(batch_data).to(self.config.device)
                    labels = torch.cat(batch_target).to(self.config.device)

                    total_labels += len(labels)

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

                    # === MEASUREMENT MODE: 1-step only, no optimizer ===
                    is_measurement = self.training_params.get("measurement_only", False)

                    if is_measurement:
                        # Capture gradient for G measurement (optimizer 없음)
                        self.measurement_gradient = {
                            name: param.grad.clone().detach().cpu()
                            for name, param in self.model.named_parameters()
                            if param.grad is not None
                        }
                        self.measurement_gradient_weight = len(labels)
                        print(
                            f"[Client {self.config.client_id}] Captured measurement gradient: {len(self.measurement_gradient)} params"
                        )
                        return  # 1-step만 하고 종료

                    # === NORMAL MODE ===
                    if self.enable_g_measurement:
                        if self.g_measurement_mode == "accumulated":
                            self._accumulate_client_grad(len(labels))
                        elif self.g_measurement_mode == "k_batch":
                            # K-batch mode: collect first K batches
                            if self._client_batch_count < self.g_measurement_k:
                                self._accumulate_client_grad(len(labels))
                                self._client_batch_count += 1
                        elif len(self.accumulated_gradients) == 0:
                            client_grad = {
                                name: param.grad.clone().detach().cpu()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            self.accumulated_gradients.append(client_grad)
                            self.gradient_weights.append(len(labels))

                    optimizer.step()

                    completed_iterations += 1
                    progress_bar.update(1)

                    self.dataset.reshuffle_dataset()

                    if completed_iterations >= iterations:
                        break
            else:
                # Original min_iterations scheduler
                while completed_iterations < iterations:
                    try:
                        batch = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(self.trainloader)
                        batch = next(dataloader_iterator)

                    inputs, labels = batch
                    total_labels += len(labels)
                    inputs, labels = (
                        inputs.to(self.config.device),
                        labels.to(self.config.device),
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

                    # === MEASUREMENT MODE: 1-step only, no optimizer ===
                    is_measurement = self.training_params.get("measurement_only", False)

                    if is_measurement:
                        # Capture gradient for G measurement (optimizer 없음)
                        self.measurement_gradient = {
                            name: param.grad.clone().detach().cpu()
                            for name, param in self.model.named_parameters()
                            if param.grad is not None
                        }
                        self.measurement_gradient_weight = len(labels)
                        print(
                            f"[Client {self.config.client_id}] Captured measurement gradient: {len(self.measurement_gradient)} params"
                        )
                        return  # 1-step만 하고 종료

                    # === NORMAL MODE ===
                    if self.enable_g_measurement:
                        if self.g_measurement_mode == "accumulated":
                            self._accumulate_client_grad(len(labels))
                        elif self.g_measurement_mode == "k_batch":
                            # K-batch mode: collect first K batches
                            if self._client_batch_count < self.g_measurement_k:
                                self._accumulate_client_grad(len(labels))
                                self._client_batch_count += 1
                        elif len(self.accumulated_gradients) == 0:
                            client_grad = {
                                name: param.grad.clone().detach().cpu()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            self.accumulated_gradients.append(client_grad)
                            self.gradient_weights.append(len(labels))

                    optimizer.step()

                    completed_iterations += 1
                    progress_bar.update(1)

                    self.dataset.reshuffle_dataset()

                    if completed_iterations >= iterations:
                        break

            progress_bar.close()

    async def _train_glue_dataset(self, params: dict):
        self.model.train()
        optimizer = self.optimizer
        iterations = params["iterations"]
        batch_schedule = params.get("batch_schedule")

        for epoch in range(self.training_params["local_epochs"]):
            dataloader_iterator = iter(self.trainloader)
            completed_iterations = 0
            progress_bar = tqdm(
                total=iterations, desc="GLUE Dataset Training", leave=True
            )

            if batch_schedule:
                # Dynamic batch scheduler: Manually construct batches
                for batch_idx, current_batch_size in enumerate(batch_schedule):
                    if batch_idx >= iterations:
                        break

                    if current_batch_size == 0:
                        # Send empty activation to avoid server deadlock
                        await self.api.submit_activations(
                            {
                                "outputs": torch.tensor([]).to(self.config.device),
                                "labels": torch.tensor([]).to(self.config.device),
                                "model_index": self.training_params["model_index"],
                                "client_id": self.config.client_id,
                            },
                            signiture=self.training_params["signiture"],
                        )
                        # Still need to wait for gradients (even if empty) to stay synchronized
                        gradients, model_index = await self.api.wait_for_gradients()

                        completed_iterations += 1
                        progress_bar.update(1)
                        continue

                    batch_inputs = []
                    batch_attention_mask = []
                    batch_labels = []
                    for _ in range(current_batch_size):
                        try:
                            batch = next(dataloader_iterator)
                            batch_inputs.append(batch["input_ids"])
                            batch_attention_mask.append(batch["attention_mask"])
                            batch_labels.append(batch["label"])
                        except StopIteration:
                            break

                    if not batch_inputs:
                        break

                    inputs = torch.cat(batch_inputs).to(self.config.device)
                    attention_mask = torch.cat(batch_attention_mask).to(
                        self.config.device
                    )
                    labels = torch.cat(batch_labels).to(self.config.device)

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

                    # G Measurement for GLUE dataset (dynamic batch scheduler)
                    if self.enable_g_measurement:
                        if self.g_measurement_mode == "accumulated":
                            self._accumulate_client_grad(len(labels))
                        elif self.g_measurement_mode == "k_batch":
                            if self._client_batch_count < self.g_measurement_k:
                                self._accumulate_client_grad(len(labels))
                                self._client_batch_count += 1
                        elif len(self.accumulated_gradients) == 0:
                            # Single mode: only first batch
                            client_grad = {
                                name: param.grad.clone().detach().cpu()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            self.accumulated_gradients.append(client_grad)
                            self.gradient_weights.append(len(labels))

                    optimizer.step()

                    completed_iterations += 1
                    progress_bar.update(1)

                    self.dataset.reshuffle_dataset()
                    if completed_iterations >= iterations:
                        break
            else:
                # Original min_iterations scheduler
                while completed_iterations < iterations:
                    try:
                        batch = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(self.trainloader)
                        batch = next(dataloader_iterator)

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

                    # G Measurement for GLUE dataset (original min_iterations)
                    if self.enable_g_measurement:
                        if self.g_measurement_mode == "accumulated":
                            self._accumulate_client_grad(len(labels))
                        elif self.g_measurement_mode == "k_batch":
                            if self._client_batch_count < self.g_measurement_k:
                                self._accumulate_client_grad(len(labels))
                                self._client_batch_count += 1
                        elif len(self.accumulated_gradients) == 0:
                            # Single mode: only first batch
                            client_grad = {
                                name: param.grad.clone().detach().cpu()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            self.accumulated_gradients.append(client_grad)
                            self.gradient_weights.append(len(labels))

                    optimizer.step()

                    completed_iterations += 1
                    progress_bar.update(1)

                    self.dataset.reshuffle_dataset()
                    if completed_iterations >= iterations:
                        break

            progress_bar.close()

    async def train(self, params: dict):
        self._reset_g_accumulation()
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
        self._finalize_g_accumulation()
