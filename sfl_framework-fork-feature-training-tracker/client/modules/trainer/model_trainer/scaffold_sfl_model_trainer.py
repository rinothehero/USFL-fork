from typing import TYPE_CHECKING
import torch
from tqdm import tqdm

from utils.log_utils import vprint

from .base_model_trainer import BaseModelTrainer
from .propagator.propagator import get_propagator

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI
    from torch.nn import Module


class ScaffoldSFLModelTrainer(BaseModelTrainer):
    """SCAFFOLD-SFL: SCAFFOLD algorithm integrated into SplitFed v2

    SCAFFOLD (Option II): Control variates applied to client-side model only.
    Server-side model remains unchanged (standard SplitFed v2).
    """
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

        # G Measurement (inherited from SFL)
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
        self._client_batch_count: int = 0

        # SCAFFOLD: Control variates (client-side only in SplitFed v2)
        self.c_i = {}  # Per-client control variate (persistent across rounds)
        self.c = {}  # Global control variate (received from server each round)
        self.theta_0 = {}  # Initial parameters (snapshot at round start)
        self.local_step_count = 0  # Count of optimizer.step() calls
        self.delta_c = {}  # Control variate update to send to server

        # Drift Measurement (SCAFFOLD-style)
        self.enable_drift_measurement = getattr(
            server_config, "enable_drift_measurement", False
        )
        self.drift_sample_interval = getattr(server_config, "drift_sample_interval", 1)
        self._round_start_params: dict = {}  # For drift measurement (separate from theta_0)
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

    # Drift Measurement Methods
    def _reset_drift_measurement(self):
        """Reset drift measurement state for new round."""
        self._round_start_params = {}
        self._drift_trajectory_sum = 0.0
        self._batch_step_count = 0
        self._endpoint_drift = 0.0

    def _snapshot_round_start_for_drift(self):
        """Save model parameters at round start for drift measurement (x_c^{t,0})."""
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

    def _snapshot_params(self):
        """Snapshot initial parameters θ0 at round start (stored on CPU)"""
        self.theta_0 = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.theta_0[name] = param.data.clone().detach().cpu()

    def _initialize_control_variates(self):
        """Initialize c_i to zeros if not already set (stored on CPU)"""
        if not self.c_i:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.c_i[name] = torch.zeros_like(param.data).cpu()

    def _apply_scaffold_correction(self):
        """Apply SCAFFOLD gradient correction: grad <- grad - c_i + c"""
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.c_i and name in self.c:
                param.grad.data = param.grad.data - self.c_i[name].to(param.device) + self.c[name].to(param.device)

    def _compute_delta_c(self):
        """Compute control variate update: c_i_new = c_i - c - delta_theta/(K*lr)"""
        lr = self.server_config.learning_rate
        K = self.local_step_count

        if K == 0:
            vprint("[SCAFFOLD] Warning: No local steps taken, skipping delta_c computation", 0)
            return

        c_i_new = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.theta_0:
                continue

            # delta_theta = theta_K - theta_0 (both on CPU)
            delta_theta = param.data.cpu() - self.theta_0[name]

            # c_i_new = c_i - c - delta_theta / (K * lr) (all on CPU)
            c_i_new[name] = (
                self.c_i[name]
                - self.c.get(name, torch.zeros_like(self.c_i[name]))
                - delta_theta / (K * lr)
            )

        # delta_c = c_i_new - c_i
        self.delta_c = {}
        for name in c_i_new:
            self.delta_c[name] = c_i_new[name] - self.c_i[name]

        # Update c_i
        self.c_i = c_i_new

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)

        # SCAFFOLD: Initialize control variates and snapshot params
        self._initialize_control_variates()
        self._snapshot_params()
        self.local_step_count = 0

        # Drift measurement: Snapshot parameters at round start
        self._snapshot_round_start_for_drift()

        for epoch in range(self.training_params["local_epochs"]):
            total_labels = 0

            for batch in tqdm(self.trainloader, desc="Training Batches"):
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
                    self.measurement_gradient = {
                        name: param.grad.clone().detach().cpu()
                        for name, param in self.model.named_parameters()
                        if param.grad is not None
                    }
                    self.measurement_gradient_weight = len(labels)
                    vprint(
                        f"[Client {self.config.client_id}] Captured measurement gradient: {len(self.measurement_gradient)} params", 2
                    )
                    return  # 1-step only

                # === NORMAL MODE ===
                if self.enable_g_measurement:
                    if self.g_measurement_mode == "accumulated":
                        self._accumulate_client_grad(len(labels))
                    elif self.g_measurement_mode == "k_batch":
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

                # SCAFFOLD: Apply gradient correction before optimizer.step()
                self._apply_scaffold_correction()

                optimizer.step()
                self.local_step_count += 1

                # Accumulate drift after parameter update
                self._accumulate_drift()

    async def _train_glue_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)

        # SCAFFOLD: Initialize control variates and snapshot params
        self._initialize_control_variates()
        self._snapshot_params()
        self.local_step_count = 0

        # Drift measurement: Snapshot parameters at round start
        self._snapshot_round_start_for_drift()

        for epoch in range(self.training_params["local_epochs"]):
            for batch in tqdm(self.trainloader, desc="Training Batches"):
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

                # G Measurement for GLUE dataset
                if self.enable_g_measurement:
                    if self.g_measurement_mode == "accumulated":
                        self._accumulate_client_grad(len(labels))
                    elif self.g_measurement_mode == "k_batch":
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

                # SCAFFOLD: Apply gradient correction before optimizer.step()
                self._apply_scaffold_correction()

                optimizer.step()
                self.local_step_count += 1

                # Accumulate drift after parameter update
                self._accumulate_drift()

    async def train(self, params: dict = None):
        self._reset_g_accumulation()
        self._reset_drift_measurement()

        # SCAFFOLD: Receive global control variate from server
        if "control_variate" in self.training_params:
            import pickle
            self.c = pickle.loads(bytes.fromhex(self.training_params["control_variate"]))
            vprint(f"[SCAFFOLD Client {self.config.client_id}] Received global control variate c", 2)
        else:
            # Initialize c to zeros if not provided (on CPU)
            self.c = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.c[name] = torch.zeros_like(param.data).cpu()

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
        self._finalize_drift_measurement()

        # SCAFFOLD: Compute control variate update
        self._compute_delta_c()
        vprint(f"[SCAFFOLD Client {self.config.client_id}] Computed delta_c after {self.local_step_count} steps", 2)
