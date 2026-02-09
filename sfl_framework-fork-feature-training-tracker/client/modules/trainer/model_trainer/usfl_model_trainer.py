import torch

from typing import TYPE_CHECKING

from tqdm import tqdm

from utils.log_utils import vprint

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

        # Drift Measurement (SCAFFOLD-style)
        self.enable_drift_measurement = getattr(
            server_config, "enable_drift_measurement", False
        )
        self.drift_sample_interval = getattr(server_config, "drift_sample_interval", 1)
        self._round_start_params: dict = {}
        self._drift_trajectory_sum: float = 0.0
        self._batch_step_count: int = 0
        self._endpoint_drift: float = 0.0

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
            vprint(f"[Client] K-batch finalized: {self._client_batch_count} batches, {self._accumulated_grad_samples} samples", 2)

    # Drift Measurement Methods (SCAFFOLD-style)
    def _reset_drift_measurement(self):
        self._round_start_params = {}
        self._drift_trajectory_sum = 0.0
        self._batch_step_count = 0
        self._endpoint_drift = 0.0

    def _snapshot_round_start(self):
        self._round_start_params = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }
        self._drift_trajectory_sum = 0.0
        self._batch_step_count = 0

    def _compute_drift_to_start(self) -> float:
        drift_sq = 0.0
        for name, param in self.model.named_parameters():
            if name in self._round_start_params:
                diff = param.data - self._round_start_params[name]
                drift_sq += (diff ** 2).sum().item()
        return drift_sq

    def _accumulate_drift(self):
        self._batch_step_count += 1
        if self._batch_step_count % self.drift_sample_interval == 0:
            drift = self._compute_drift_to_start()
            self._drift_trajectory_sum += drift
            self._endpoint_drift = drift

    def _finalize_drift_measurement(self):
        self._endpoint_drift = self._compute_drift_to_start()
        if self.drift_sample_interval > 1:
            sampled_steps = self._batch_step_count // self.drift_sample_interval
            if sampled_steps > 0:
                self._drift_trajectory_sum = (
                    self._drift_trajectory_sum * self._batch_step_count / sampled_steps
                )

    def get_drift_metrics(self) -> dict:
        return {
            "drift_trajectory_sum": self._drift_trajectory_sum,
            "drift_batch_steps": self._batch_step_count,
            "drift_endpoint": self._endpoint_drift,
        }

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.optimizer
        iterations = params["iterations"]
        schedule_dict = params.get("batch_schedule")
        my_schedule = (
            schedule_dict.get(str(self.config.client_id)) if schedule_dict else None
        )

        # Drift Measurement: Snapshot round start params
        if self.enable_drift_measurement:
            self._snapshot_round_start()

        for epoch in range(self.training_params["local_epochs"]):
            vprint(f"Epoch: {epoch}, client_id: {self.config.client_id}", 2)
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
                        vprint(
                            f"[Client {self.config.client_id}] Captured measurement gradient: {len(self.measurement_gradient)} params", 2
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

                    # Drift Measurement: Accumulate after optimizer step
                    if self.enable_drift_measurement:
                        self._accumulate_drift()

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
                        vprint(
                            f"[Client {self.config.client_id}] Captured measurement gradient: {len(self.measurement_gradient)} params", 2
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

                    # Drift Measurement: Accumulate after optimizer step
                    if self.enable_drift_measurement:
                        self._accumulate_drift()

                    completed_iterations += 1
                    progress_bar.update(1)

                    self.dataset.reshuffle_dataset()

                    if completed_iterations >= iterations:
                        break

            progress_bar.close()

        # Drift Measurement: Finalize at round end
        if self.enable_drift_measurement:
            self._finalize_drift_measurement()

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
        self._reset_drift_measurement()
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
