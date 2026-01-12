import asyncio
import time
from collections import Counter
from typing import TYPE_CHECKING, Tuple

import torch

from .base_stage_organizer import BaseStageOrganizer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...trainer.aggregator.base_aggregator import BaseAggregator
    from ...trainer.seletor.base_selector import BaseSelector
    from ...trainer.splitter.base_splitter import BaseSplitter
    from ...ws.connection import Connection


class ScalaStageOrganizer(BaseStageOrganizer):
    def __init__(
        self,
        config: "Config",
        connection: "Connection",
        global_dict: "GlobalDict",
        aggregator: "BaseAggregator",
        model: "BaseModel",
        dataset: "BaseDataset",
        selector: "BaseSelector",
        splitter: "BaseSplitter",
    ):
        self.config = config
        self.connection = connection
        self.global_dict = global_dict

        self.aggregator = aggregator
        self.model = model
        self.testloader = dataset.get_testloader()
        self.num_classes = dataset.get_num_classes()
        self.selector = selector
        self.splitter = splitter

        self.pre_round = PreRound(config, global_dict)
        self.in_round = InRound(config, global_dict)
        self.post_round = PostRound(config, global_dict)

        self.round_start_time = 0
        self.round_end_time = 0
        self.selected_clients = []
        self.split_models = []

    def _print_memory_usage(self, step):
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{step} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")

    def _calculate_batch_size(self, dataset_sizes: dict[int, int]):
        total_dataset_size = sum(dataset_sizes.values())
        batch_size = self.config.batch_size

        batch_sizes = {
            str(client_id): int(batch_size * dataset_size / total_dataset_size)
            for client_id, dataset_size in dataset_sizes.items()
        }

        for client_id in batch_sizes:
            if batch_sizes[client_id] <= 0:
                batch_sizes[client_id] = 1
                print(f"Warning: Client {client_id} had zero batch size, set to 1")

        iterations_per_client = {
            str(client_id): (
                dataset_sizes[int(client_id)] // batch_sizes[str(client_id)]
            )
            for client_id in batch_sizes
        }

        return batch_sizes, iterations_per_client

    def _calculate_log_class_priors(self, labels):
        label_counts = Counter(labels.tolist())
        total_labels = sum(label_counts.values())

        class_priors = torch.zeros(self.num_classes)
        for label, count in label_counts.items():
            class_priors[label] = count / total_labels

        class_priors[class_priors == 0] = 1e-9

        log_class_priors = torch.log(class_priors).to(self.config.device)
        return log_class_priors

    def _concatenate_activations(self, concatenated_activations, activation):
        if not concatenated_activations:
            if isinstance(activation["outputs"], tuple):
                concatenated_activations["outputs"] = [
                    torch.tensor([], device=self.config.device)
                    for _ in range(len(activation["outputs"]))
                ]
            else:
                concatenated_activations["outputs"] = torch.tensor(
                    [], device=self.config.device
                )

            for key in ["labels", "attention_mask"]:
                if key in activation:
                    concatenated_activations[key] = torch.tensor(
                        [], device=self.config.device
                    )

        if isinstance(activation["outputs"], tuple):
            outputs_list = list(activation["outputs"])
            if isinstance(concatenated_activations["outputs"], tuple):
                concatenated_outputs = list(concatenated_activations["outputs"])
            else:
                concatenated_outputs = concatenated_activations["outputs"]

            for i in range(len(outputs_list)):
                if isinstance(outputs_list[i], torch.Tensor):
                    outputs_list[i] = outputs_list[i].to(self.config.device)

                    if len(concatenated_outputs[i]) == 0:
                        concatenated_outputs[i] = (
                            outputs_list[i].detach().clone().requires_grad_(True)
                        )
                    else:
                        concatenated_outputs[i] = (
                            torch.cat(
                                [
                                    concatenated_outputs[i].to(self.config.device),
                                    outputs_list[i],
                                ],
                                dim=0,
                            )
                            .detach()
                            .clone()
                            .requires_grad_(True)
                        )

            activation["outputs"] = tuple(outputs_list)
            concatenated_activations["outputs"] = tuple(concatenated_outputs)

        else:
            activation_output = activation["outputs"].to(self.config.device)

            if len(concatenated_activations["outputs"]) == 0:
                concatenated_activations["outputs"] = activation_output
            else:
                concatenated_activations["outputs"] = torch.cat(
                    [concatenated_activations["outputs"], activation_output],
                    dim=0,
                )

        for key in ["labels", "attention_mask"]:
            if key in activation:
                tensor_on_device = activation[key].to(self.config.device)

                if (
                    key not in concatenated_activations
                    or len(concatenated_activations[key]) == 0
                ):
                    concatenated_activations[key] = tensor_on_device
                else:
                    concatenated_activations[key] = torch.cat(
                        [concatenated_activations[key], tensor_on_device],
                        dim=0,
                    )

        return concatenated_activations

    async def _pre_round(self, round_number: int):
        await self.pre_round.wait_for_client_informations()
        await self.pre_round.wait_for_clients()

        client_informations = self.global_dict.get("client_informations")
        self.selected_clients = self.pre_round.select_clients(
            self.selector,
            self.connection,
            {
                "client_informations": client_informations,
                "num_classes": self.num_classes,
                "batch_size": self.config.batch_size,
            },
        )
        dataset_sizes = {
            client_information["client_id"]: client_information["dataset"]["size"]
            for client_information in client_informations.values()
            if client_information["client_id"] in self.selected_clients
        }

        batch_sizes, iterations_per_client = self._calculate_batch_size(dataset_sizes)
        min_iterations = min(iterations_per_client.values())

        split_models = self.splitter.split(
            self.model.get_torch_model(), self.config.__dict__
        )
        self.split_models = split_models

        model_queue = self.global_dict.get("model_queue")
        model_queue.start_insert_mode()
        self.round_start_time, self.round_end_time = (
            self.pre_round.calculate_round_end_time()
        )

        await self.pre_round.send_customized_global_model(
            self.selected_clients,
            [self.split_models[0] for _ in range(len(self.selected_clients))],
            self.connection,
            {
                "round_number": round_number,
                "round_end_time": self.round_end_time,
                "round_start_time": self.round_start_time,
                "signiture": model_queue.get_signiture(),
                "local_epochs": self.config.local_epochs,
                "split_count": len(self.split_models),
                "model_index": 0,
                "batch_sizes": batch_sizes,
                "iterations": min_iterations,
            },
        )

    async def _in_round(self, round_number: int):
        async def __server_side_training():
            enable_concatenation = self.config.enable_concatenation
            enable_logit_adjustment = self.config.enable_logit_adjustment

            if enable_concatenation and enable_logit_adjustment:
                print("Server side training with concatenation and logit adjustment")
                while True:
                    activations = await self.in_round.wait_for_concatenated_activations(
                        self.selected_clients
                    )

                    activation_length_per_client = {
                        activation["client_id"]: len(activation["labels"])
                        for activation in activations
                    }

                    concatenated_activations = {}

                    for activation in activations:
                        concatenated_activations = self._concatenate_activations(
                            concatenated_activations, activation
                        )

                    global_class_priors = self._calculate_log_class_priors(
                        concatenated_activations["labels"],
                    )
                    logits = await self.in_round.forward(
                        self.split_models[1], concatenated_activations
                    )
                    logits += global_class_priors
                    await self.in_round.backward_from_label(
                        self.split_models[1], logits, concatenated_activations
                    )

                    start_index = 0
                    for activation in activations:
                        client_id = activation["client_id"]
                        end_index = (
                            start_index + activation_length_per_client[client_id]
                        )

                        client_class_priors = self._calculate_log_class_priors(
                            activation["labels"]
                        )
                        logits = await self.in_round.forward(
                            self.split_models[1], concatenated_activations
                        )
                        logits += client_class_priors
                        client_grad = await self.in_round.backward_from_label_without_parameter_update(
                            self.split_models[1],
                            logits,
                            concatenated_activations,
                        )
                        client_grad = client_grad[start_index:end_index]

                        await self.in_round.send_gradients(
                            self.connection,
                            client_grad,
                            client_id,
                            0,
                        )

                        start_index = end_index

            elif enable_concatenation and not enable_logit_adjustment:
                print("Server side training with concatenation and no logit adjustment")
                while True:
                    activations = await self.in_round.wait_for_concatenated_activations(
                        self.selected_clients
                    )

                    activation_length_per_client = {
                        activation["client_id"]: len(activation["labels"])
                        for activation in activations
                    }

                    concatenated_activations = {}

                    for activation in activations:
                        concatenated_activations = self._concatenate_activations(
                            concatenated_activations, activation
                        )

                    logits = await self.in_round.forward(
                        self.split_models[1], concatenated_activations
                    )
                    grad, loss = await self.in_round.backward_from_label(
                        self.split_models[1], logits, concatenated_activations
                    )

                    start_index = 0
                    for activation in activations:
                        client_id = activation["client_id"]
                        end_index = (
                            start_index + activation_length_per_client[client_id]
                        )

                        client_grad = grad[start_index:end_index]

                        await self.in_round.send_gradients(
                            self.connection,
                            client_grad,
                            client_id,
                            0,
                        )

                        start_index = end_index

            elif not enable_concatenation and enable_logit_adjustment:
                print("Server side training with no concatenation and logit adjustment")
                while True:
                    activation = await self.in_round.wait_for_activations()

                    class_priors = self._calculate_log_class_priors(
                        activation["labels"],
                    )

                    logits = await self.in_round.forward(
                        self.split_models[1], activation
                    )
                    logits += class_priors

                    grad, loss = await self.in_round.backward_from_label(
                        self.split_models[1], logits, activation
                    )

                    await self.in_round.send_gradients(
                        self.connection,
                        grad,
                        activation["client_id"],
                        activation["model_index"],
                    )
            else:
                raise ValueError("Invalid configuration")

        wait_for_models = asyncio.create_task(
            self.in_round.wait_for_model_submission(
                self.selected_clients, self.round_end_time
            )
        )
        server_side_training = asyncio.create_task(__server_side_training())

        done, pending = await asyncio.wait(
            [wait_for_models, server_side_training], return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        await asyncio.gather(*pending, return_exceptions=True)

    async def _post_round(self, round_number: int):
        model_queue = self.global_dict.get("model_queue")
        model_queue.end_insert_mode()

        client_ids = [model[0] for model in model_queue.queue]
        self.global_dict.add_event(
            "MODEL_AGGREGATION_START", {"client_ids": client_ids}
        )
        updated_torch_model = self.post_round.aggregate_models(
            self.aggregator, model_queue
        )
        self.global_dict.add_event("MODEL_AGGREGATION_END", {"client_ids": client_ids})

        if isinstance(updated_torch_model, torch.nn.ModuleDict):
            updated_torch_model.update(self.split_models[1])

        if updated_torch_model != None:
            updated_torch_model = self.aggregator.model_reshape(updated_torch_model)
            self.post_round.update_global_model(updated_torch_model, self.model)
            print("Updated global model")

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        print(f"[Round {round_number}] Accuracy: {accuracy}")

        if self.config.device == "cuda":
            self._print_memory_usage(f"[Round {round_number} Before]")
            torch.cuda.empty_cache()
            self._print_memory_usage(f"[Round {round_number} After]")
