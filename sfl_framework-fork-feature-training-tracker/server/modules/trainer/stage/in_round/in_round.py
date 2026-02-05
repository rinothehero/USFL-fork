import asyncio
import pickle
import time
from typing import TYPE_CHECKING, Tuple

import orjson
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from ...propagator.propagator import get_propagator

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn.modules import ModuleDict

    from ....global_dict.global_dict import GlobalDict
    from ....ws.connection import Connection


class InRound:
    def __init__(self, config: "Config", global_dict: "GlobalDict"):
        self.config = config
        self.global_dict = global_dict

    def _get_criterion(self, config: "Config"):
        if config.criterion == "ce":
            return nn.CrossEntropyLoss()
        elif config.criterion == "mse":
            return nn.MSELoss()
        elif config.criterion == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown criterion {config.criterion}")

    def _get_optimizer(
        self, torch_model: "ModuleDict", config: "Config", lr_override: float = None
    ):
        lr = lr_override if lr_override is not None else config.learning_rate
        if config.optimizer == "sgd":
            return optim.SGD(
                torch_model.parameters(),
                lr=lr,
                momentum=config.momentum,
            )
        elif config.optimizer == "adam":
            return optim.Adam(torch_model.parameters(), lr=lr)
        elif config.optimizer == "adamw":
            return optim.AdamW(torch_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer {config.optimizer}")

    async def wait_for_model_submission(
        self,
        selected_clients: list[int],
        round_end_time: float,
    ):
        model_queue = self.global_dict.get("model_queue")

        while True:
            if model_queue.get_queue_size() >= len(selected_clients):
                break

            if time.time() > round_end_time:
                break

            await asyncio.sleep(0.0001)

    async def wait_for_concatenated_activations(self, selected_clients: list[int]):
        activation_queue = self.global_dict.get("activation_queue")

        previous_length = -1
        while True:
            current_length = len(activation_queue)
            if current_length == len(selected_clients):
                activations = []

                while len(activation_queue) != 0:
                    activations.append(activation_queue.popleft())

                return activations

            if current_length != previous_length:
                previous_length = current_length

            await asyncio.sleep(0.0001)

    async def wait_for_activations(self):
        activation_queue = self.global_dict.get("activation_queue")

        while True:
            if len(activation_queue) != 0:
                activation = activation_queue.popleft()
                return activation

            await asyncio.sleep(0.0001)

    async def forward(self, torch_model: "ModuleDict", activation: dict):
        torch_model = torch_model.to(self.config.device)
        propagator = get_propagator(self.config, torch_model)

        if isinstance(activation["outputs"], Tuple):
            activation_list = list(activation["outputs"])
            for i in range(len(activation_list)):
                if isinstance(activation_list[i], Tensor):
                    activation_list[i] = activation_list[i].to(self.config.device)
            input = tuple(activation_list)
        else:
            input = activation["outputs"].to(self.config.device)

        attention_mask = (
            activation["attention_mask"].to(self.config.device)
            if "attention_mask" in activation
            and activation["attention_mask"] is not None
            else None
        )

        # BatchNorm requires batch_size > 1 during training
        batch_size = input[0].shape[0] if isinstance(input, tuple) else input.shape[0]
        if batch_size == 1 and torch_model.training:
            torch_model.eval()
            result = propagator.forward(input, {"attention_mask": attention_mask})
            torch_model.train()
            return result

        return propagator.forward(input, {"attention_mask": attention_mask})

    async def backward_from_label_without_parameter_update(
        self, torch_model: "ModuleDict", outputs, activation
    ):
        torch_model = torch_model.to(self.config.device)
        outputs = outputs.to(self.config.device)
        labels = activation["labels"].to(self.config.device)
        inputs = activation["outputs"]

        if isinstance(inputs, tuple):
            if inputs[0].requires_grad:
                inputs[0].retain_grad()
        else:
            if inputs.requires_grad:
                inputs.retain_grad()

        criterion = self._get_criterion(self.config)
        loss = criterion(outputs, labels)
        loss.backward()

        grad = (
            inputs[0].grad.clone().detach()
            if isinstance(inputs, tuple)
            else inputs.grad.clone().detach()
        )

        if isinstance(inputs, tuple):
            for i in range(len(inputs)):
                if isinstance(inputs[i], Tensor):
                    inputs[i].grad.detach_()
        else:
            inputs.grad.detach_()

        return grad

    async def backward_from_label_using_prox(
        self,
        client_proximal_term,
        orinal_model: "ModuleDict",
        torch_model: "ModuleDict",
        outputs,
        activation,
        optimizer: "optim.Optimizer" = None,
        criterion: "nn.Module" = None,
    ):
        torch_model = torch_model.to(self.config.device)
        outputs = outputs.to(self.config.device)
        labels = activation["labels"].to(self.config.device)
        inputs = activation["outputs"]

        if isinstance(inputs, tuple):
            for item in inputs:
                if isinstance(item, Tensor) and item.requires_grad:
                    item.retain_grad()
        else:
            if inputs.requires_grad:
                inputs.retain_grad()

        # Use external optimizer/criterion if provided, otherwise create new (legacy)
        if criterion is None:
            criterion = self._get_criterion(self.config)
        if optimizer is None:
            optimizer = self._get_optimizer(torch_model, self.config)

        optimizer.zero_grad()

        proximal_term = float(client_proximal_term)
        for w, w_t in zip(torch_model.parameters(), orinal_model.parameters()):
            proximal_term += (w - w_t).norm(2) ** 2

        # NEED REFACTOR
        # if self.config.dataset == "sts-b":
        #     outputs = outputs.squeeze()
        #     labels = labels.squeeze()
        # elif self.config.dataset in ["cifar10", "cifar100"]:
        #     pass
        # else:
        #     outputs = outputs.view(-1, outputs.size(-1))
        #     labels = labels.view(-1)

        loss = criterion(outputs, labels) + (self.config.prox_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()

        if isinstance(inputs, tuple):
            grad = tuple(
                item.grad.clone().detach()
                for item in inputs
                if isinstance(item, Tensor)
            )
        else:
            grad = inputs.grad.clone().detach()

        if isinstance(inputs, tuple):
            for i in range(len(inputs)):
                if isinstance(inputs[i], Tensor):
                    inputs[i].grad.detach_()
        else:
            inputs.grad.detach_()

        return grad, loss.item()

    async def backward_from_label(
        self,
        torch_model: "ModuleDict",
        outputs,
        activation,
        collect_server_grad: bool = False,
        skip_optimizer: bool = False,
        optimizer: "optim.Optimizer" = None,
        criterion: "nn.Module" = None,
    ):
        """
        Perform backward pass from labels.

        Args:
            torch_model: Server-side model
            outputs: Model outputs
            activation: Activation dict containing labels and inputs
            collect_server_grad: Whether to collect server gradients for G measurement
            skip_optimizer: Whether to skip optimizer.step()
            optimizer: External optimizer for state persistence (recommended).
                       If None, creates a new optimizer (legacy behavior, not recommended).
            criterion: External criterion. If None, creates from config.
        """
        torch_model = torch_model.to(self.config.device)
        outputs = outputs.to(self.config.device)
        labels = activation["labels"].to(self.config.device)
        inputs = activation["outputs"]

        if isinstance(inputs, tuple):
            for item in inputs:
                if isinstance(item, Tensor) and item.requires_grad:
                    item.retain_grad()
        else:
            if inputs.requires_grad:
                inputs.retain_grad()

        # Use external optimizer/criterion if provided, otherwise create new (legacy)
        if criterion is None:
            criterion = self._get_criterion(self.config)
        if optimizer is None:
            optimizer = self._get_optimizer(torch_model, self.config)

        optimizer.zero_grad()

        # NEED REFACTOR
        # if self.config.dataset == "sts-b":
        #     outputs = outputs.squeeze()
        #     labels = labels.squeeze()
        # elif self.config.dataset in ["cifar10", "cifar100"]:
        #     pass
        # else:
        #     outputs = outputs.view(-1, outputs.size(-1))
        #     labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()

        # G Measurement: Collect server model gradients BEFORE optimizer.step()
        server_model_grad = None
        if collect_server_grad:
            server_model_grad = {
                name: param.grad.clone().detach().cpu()
                for name, param in torch_model.named_parameters()
                if param.grad is not None
            }

        # Measurement mode: skip optimizer step
        if not skip_optimizer:
            optimizer.step()

        if isinstance(inputs, tuple):
            grad = tuple(
                item.grad.clone().detach()
                for item in inputs
                if isinstance(item, Tensor)
            )
        else:
            grad = inputs.grad.clone().detach()

        if isinstance(inputs, tuple):
            for i in range(len(inputs)):
                if isinstance(inputs[i], Tensor):
                    inputs[i].grad.detach_()
        else:
            inputs.grad.detach_()

        return grad, loss.item(), server_model_grad

    async def backward_from_grad(self, torch_model: "ModuleDict", grads):
        pass

    async def send_gradients(
        self, connection: "Connection", grads, client_id: int, model_index: int
    ):
        grads = pickle.dumps(grads).hex()
        await connection.send_bytes(
            orjson.dumps(
                {
                    "event": "gradients",
                    "params": {
                        "gradients": grads,
                        "model_index": model_index,
                    },
                }
            ),
            client_id,
            logging=False,
        )
