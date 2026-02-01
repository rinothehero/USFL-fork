from typing import TYPE_CHECKING
import torch
from tqdm import tqdm

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

    def _snapshot_params(self):
        """Snapshot initial parameters θ0 at round start"""
        self.theta_0 = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.theta_0[name] = param.data.clone().detach()

    def _initialize_control_variates(self):
        """Initialize c_i to zeros if not already set"""
        if not self.c_i:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.c_i[name] = torch.zeros_like(param.data)

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
            print("[SCAFFOLD] Warning: No local steps taken, skipping delta_c computation")
            return

        c_i_new = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.theta_0:
                continue

            # delta_theta = theta_K - theta_0
            delta_theta = param.data - self.theta_0[name]

            # c_i_new = c_i - c - delta_theta / (K * lr)
            c_i_new[name] = (
                self.c_i[name]
                - self.c.get(name, torch.zeros_like(param.data))
                - delta_theta.cpu() / (K * lr)
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
                    print(
                        f"[Client {self.config.client_id}] Captured measurement gradient: {len(self.measurement_gradient)} params"
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

    async def _train_glue_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)

        # SCAFFOLD: Initialize control variates and snapshot params
        self._initialize_control_variates()
        self._snapshot_params()
        self.local_step_count = 0

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

    async def train(self, params: dict = None):
        self._reset_g_accumulation()

        # SCAFFOLD: Receive global control variate from server
        if "control_variate" in self.training_params:
            import pickle
            self.c = pickle.loads(bytes.fromhex(self.training_params["control_variate"]))
            print(f"[SCAFFOLD Client {self.config.client_id}] Received global control variate c")
        else:
            # Initialize c to zeros if not provided
            self.c = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.c[name] = torch.zeros_like(param.data)

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

        # SCAFFOLD: Compute control variate update
        self._compute_delta_c()
        print(f"[SCAFFOLD Client {self.config.client_id}] Computed delta_c after {self.local_step_count} steps")
