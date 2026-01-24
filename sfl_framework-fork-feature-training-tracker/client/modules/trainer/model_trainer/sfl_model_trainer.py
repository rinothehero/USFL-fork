from typing import TYPE_CHECKING

from tqdm import tqdm

from .base_model_trainer import BaseModelTrainer
from .propagator.propagator import get_propagator

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.dataset.base_dataset import BaseDataset
    from modules.trainer.apis.common import CommonAPI
    from torch.nn import Module


class SFLModelTrainer(BaseModelTrainer):
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

        # G Measurement: Accumulated client gradients for this round
        self.enable_g_measurement = getattr(
            server_config, "enable_g_measurement", False
        )
        self.g_measurement_mode = getattr(server_config, "g_measurement_mode", "single")
        self.accumulated_gradients: list = []  # List of gradient dicts
        self.gradient_weights: list = []  # Weights for each gradient (batch size)
        self.measurement_gradient = None  # For 1-step measurement protocol
        self.measurement_gradient_weight = None
        self._accumulated_grad_sum: dict = {}
        self._accumulated_grad_samples: int = 0

    def _reset_g_accumulation(self):
        self.accumulated_gradients = []
        self.gradient_weights = []
        self._accumulated_grad_sum = {}
        self._accumulated_grad_samples = 0

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
        if self.g_measurement_mode != "accumulated":
            return
        if self._accumulated_grad_samples <= 0:
            return
        avg_grad = {
            name: grad / float(self._accumulated_grad_samples)
            for name, grad in self._accumulated_grad_sum.items()
        }
        self.accumulated_gradients = [avg_grad]
        self.gradient_weights = [self._accumulated_grad_samples]

    async def _train_default_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)

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
                    elif len(self.accumulated_gradients) == 0:
                        client_grad = {
                            name: param.grad.clone().detach().cpu()
                            for name, param in self.model.named_parameters()
                            if param.grad is not None
                        }
                        self.accumulated_gradients.append(client_grad)
                        self.gradient_weights.append(len(labels))

                optimizer.step()

    async def _train_glue_dataset(self, params: dict):
        self.model.train()
        optimizer = self.get_optimizer(self.server_config)

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
                optimizer.step()

                if self.enable_g_measurement and self.g_measurement_mode == "accumulated":
                    self._accumulate_client_grad(len(labels))

    async def train(self, params: dict = None):
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
