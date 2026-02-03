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


class Mix2SFLModelTrainer(BaseModelTrainer):
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

        self.used_samples = 0

        self.enable_g_measurement = getattr(
            server_config, "enable_g_measurement", False
        )
        self.g_measurement_mode = getattr(server_config, "g_measurement_mode", "single")
        self.g_measurement_k = getattr(server_config, "g_measurement_k", 5)
        self.accumulated_gradients: list = []
        self.gradient_weights: list = []
        self.measurement_gradient = None
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

    # Drift Measurement Methods
    def _reset_drift_measurement(self):
        """Reset drift measurement state for new round."""
        self._round_start_params = {}
        self._drift_trajectory_sum = 0.0
        self._batch_step_count = 0
        self._endpoint_drift = 0.0

    def _snapshot_round_start(self):
        """Save model parameters at round start (x_c^{t,0})."""
        if not self.enable_drift_measurement:
            return
        self._round_start_params = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
        }

    def _compute_drift_to_start(self) -> float:
        """Compute ||x_c^{t,b} - x_c^{t,0}||^2."""
        if not self._round_start_params:
            return 0.0
        drift_sq = 0.0
        for name, param in self.model.named_parameters():
            if name in self._round_start_params:
                diff = param.detach().cpu() - self._round_start_params[name]
                drift_sq += (diff ** 2).sum().item()
        return drift_sq

    def _accumulate_drift(self):
        """Accumulate drift after optimizer step."""
        if not self.enable_drift_measurement:
            return
        self._batch_step_count += 1
        if self._batch_step_count % self.drift_sample_interval == 0:
            drift_sq = self._compute_drift_to_start()
            self._drift_trajectory_sum += drift_sq

    def _finalize_drift_measurement(self):
        """Finalize drift measurement at round end."""
        if not self.enable_drift_measurement:
            return
        # Always compute endpoint drift at the very end
        self._endpoint_drift = self._compute_drift_to_start()
        # If sampling was used, extrapolate trajectory sum
        if self.drift_sample_interval > 1:
            sampled_steps = self._batch_step_count // self.drift_sample_interval
            if sampled_steps > 0:
                self._drift_trajectory_sum = (
                    self._drift_trajectory_sum * self._batch_step_count / sampled_steps
                )

    def get_drift_metrics(self) -> dict:
        """Return drift metrics for server."""
        return {
            "drift_trajectory_sum": self._drift_trajectory_sum,
            "drift_batch_steps": self._batch_step_count,
            "drift_endpoint": self._endpoint_drift,
        }

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
        if self.g_measurement_mode == "k_batch":
            print(f"[Client] K-batch finalized: {self._client_batch_count} batches, {self._accumulated_grad_samples} samples")
        self.accumulated_gradients = [avg_grad]
        self.gradient_weights = [self._accumulated_grad_samples]

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)
        iterations = int(params.get("iterations", len(self.trainloader)))

        # Snapshot parameters at round start for drift measurement
        self._snapshot_round_start()

        for epoch in range(self.training_params["local_epochs"]):
            dataloader_iterator = iter(self.trainloader)
            local_iterations = len(self.trainloader)

            for step in tqdm(range(iterations), desc="Training Batches"):
                if step < local_iterations:
                    inputs, labels = next(dataloader_iterator)
                    self.used_samples += len(labels)
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

                    is_measurement = self.training_params.get("measurement_only", False)
                    if is_measurement:
                        self.measurement_gradient = {
                            name: param.grad.clone().detach().cpu()
                            for name, param in self.model.named_parameters()
                            if param.grad is not None
                        }
                        self.measurement_gradient_weight = len(labels)
                        print(
                            f"[Client {self.config.client_id}] Captured measurement gradient: {len(self.measurement_gradient)} params"
                        )
                        return

                    if self.enable_g_measurement:
                        if self.g_measurement_mode == "accumulated":
                            self._accumulate_client_grad(len(labels))
                        elif self.g_measurement_mode == "k_batch":
                            # K-batch mode: collect first K batches
                            if self._client_batch_count < self.g_measurement_k:
                                self._accumulate_client_grad(len(labels))
                                self._client_batch_count += 1
                        elif not self.accumulated_gradients:
                            client_grad = {
                                name: param.grad.clone().detach().cpu()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            self.accumulated_gradients.append(client_grad)
                            self.gradient_weights.append(len(labels))

                    optimizer.step()

                    # Accumulate drift after parameter update
                    self._accumulate_drift()
                else:
                    await self.api.submit_activations(
                        {
                            "outputs": torch.tensor([]).to(self.config.device),
                            "labels": torch.tensor([]).to(self.config.device),
                            "model_index": self.training_params["model_index"],
                            "client_id": self.config.client_id,
                        },
                        signiture=self.training_params["signiture"],
                    )
                    await self.api.wait_for_gradients()

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
            raise ValueError("mix2sfl supports vision datasets only")
        await self._train_default_dataset(params)
        self._finalize_g_accumulation()
        self._finalize_drift_measurement()
