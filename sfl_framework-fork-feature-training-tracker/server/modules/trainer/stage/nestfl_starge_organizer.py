import copy
import time
from typing import TYPE_CHECKING

import torch

from .base_stage_organizer import BaseStageOrganizer
from .dependency.functions.nestfl_functions import restore_group
from .dependency.torch_pruning.pruner.algorithms.metapruner import MetaPruner
from .dependency.torch_pruning.pruner.importance import MagnitudeImportance
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
    from ...ws.connection import Connection

INF = 99999999999999


class NestFLStageOrganizer(BaseStageOrganizer):
    def __init__(
        self,
        config: "Config",
        connection: "Connection",
        global_dict: "GlobalDict",
        aggregator: "BaseAggregator",
        model: "BaseModel",
        dataset: "BaseDataset",
        selector: "BaseSelector",
    ):
        self.config = config
        self.connection = connection
        self.global_dict = global_dict

        self.aggregator = aggregator
        self.model = model
        self.testloader = dataset.get_testloader()
        self.selector = selector

        self.pre_round = PreRound(config, global_dict)
        self.in_round = InRound(config, global_dict)
        self.post_round = PostRound(config, global_dict)

        self.round_start_time = 0
        self.round_end_time = 0
        self.selected_clients = []
        self.pruning_records = {}
        self.pruner = {}

        self.parameters = self._count_parameters(self.model.get_torch_model())

    def _get_example_inputs(self):
        if self.config.model == "alexnet":
            return torch.randn(1, 1, 28, 28).to(self.config.device)
        else:
            return torch.randn(1, 3, 32, 32).to(self.config.device)

    def _get_ignored_layers(self, torch_model):
        ignored_layers = []

        if self.config.model == "resnet18":
            ignored_layers = [torch_model.conv1, torch_model.fc]
        elif self.config.model == "alexnet":
            ignored_layers = [torch_model.conv1, torch_model.fc3, torch_model.conv5]
        elif self.config.model == "tiny_vgg11":
            ignored_layers = [torch_model.conv1, torch_model.fc3]
        else:
            raise ValueError("Model type not recognized")

        return ignored_layers

    def _prune(self, torch_model, pruning_ratio):
        print(f"Before pruning parameters: {self._count_parameters(torch_model)}")

        pruner = MetaPruner(
            model=torch_model,
            example_inputs=self._get_example_inputs(),
            ignored_layers=self._get_ignored_layers(torch_model),
            importance=MagnitudeImportance(),
            pruning_ratio=pruning_ratio,
            max_pruning_ratio=pruning_ratio,
        )

        pruning_record = []
        if pruning_ratio > 0:
            for group in pruner.step(interactive=True):
                dep, idxs = group[0]
                target_module = dep.target.module
                pruning_fn = dep.handler
                pruning_record.append((target_module, pruning_fn, idxs))
                group.prune()

        print(f"After pruning parameters: {self._count_parameters(torch_model)}")

        return pruning_record, pruner

    async def _pre_round(self, round_number: int):
        await self.pre_round.wait_for_client_informations()
        await self.pre_round.wait_for_clients()

        self.pruning_records = {}
        self.pruner = {}

        self.selected_clients = self.pre_round.select_clients(
            self.selector, self.connection
        )

        client_informations = self.global_dict.get("client_informations")
        v_cpus = {
            client: client_informations[client]["cpu"]["core"]
            for client in client_informations
        }
        max_v_cpu = max(v_cpus.values())

        self.global_dict.add_event(
            "CLIENTS_SELECTED", {"client_ids": self.selected_clients, "v_cpus": v_cpus}
        )

        pruned_models = []
        for client in self.selected_clients:
            pruned_model = copy.deepcopy(self.model.get_torch_model())
            print(v_cpus)
            pruning_ratio = 1 - (v_cpus[int(client)] / max_v_cpu)

            if pruning_ratio in [0.8, 0.9]:
                pruning_ratio = 0.55

            print(
                f"Pruning ratio for client {client}: {pruning_ratio}, max_v_cpu: {max_v_cpu}, v_cpu: {v_cpus[int(client)]}"
            )

            self.global_dict.add_event(
                "MODEL_PRUNING_START",
                {"client_id": client, "pruning_ratio": pruning_ratio},
            )

            pruning_record, pruner = self._prune(pruned_model, pruning_ratio)
            pruned_models.append(pruned_model)

            self.global_dict.add_event(
                "MODEL_PRUNING_END",
                {"client_id": client, "pruning_ratio": pruning_ratio},
            )

            self.pruning_records[client] = pruning_record
            self.pruner[client] = pruner

        self.round_start_time, self.round_end_time = (
            self.pre_round.calculate_round_end_time()
        )
        model_queue = self.global_dict.get("model_queue")
        model_queue.start_insert_mode()

        self.global_dict.add_event(
            "ROUND_START",
            {
                "strat_timestamp": self.round_start_time,
                "end_timestamp": self.round_end_time,
            },
        )

        await self.pre_round.send_customized_global_model(
            self.selected_clients,
            pruned_models,
            self.connection,
            {
                "round_number": round_number,
                "round_end_time": self.round_end_time,
                "round_start_time": self.round_start_time,
                "signiture": model_queue.get_signiture(),
                "local_epochs": self.config.local_epochs,
            },
        )

    async def _in_round(self, round_number: int):
        await self.in_round.wait_for_model_submission(
            self.selected_clients, self.round_end_time
        )

    async def _post_round(self, round_number: int):
        model_queue = self.global_dict.get("model_queue")
        model_queue.end_insert_mode()

        models = []
        client_ids = [model[0] for model in model_queue.queue]
        if len(client_ids) > 0:
            self.global_dict.add_event(
                "MODEL_RESTORING_START",
            )

            for idx, client_id in enumerate(client_ids):
                pruner = self.pruner[client_id]
                pruner.model.load_state_dict(model_queue.queue[idx][1].state_dict())

                pruning_record = self.pruning_records[client_id]
                pruning_record.reverse()

                for target_module, pruning_fn, idxs in pruning_record:
                    group = pruner.DG.get_pruning_group(target_module, pruning_fn, idxs)
                    restore_group(group, self.config)

                models.append(pruner.model)

            self.global_dict.add_event("MODEL_RESTORING_END")

        _, __, dataset_sizes = model_queue.get_all_models()

        self.global_dict.add_event(
            "MODEL_AGGREGATION_START", {"client_ids": client_ids}
        )
        updated_torch_model = self.post_round.aggregate_torch_models(
            self.aggregator, models, dataset_sizes
        )

        self.global_dict.add_event("MODEL_AGGREGATION_END", {"client_ids": client_ids})

        if updated_torch_model != None:
            self.post_round.update_global_model(updated_torch_model, self.model)
            print("Updated global model")

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)

        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        print(f"[Round {round_number}] Accuracy: {accuracy}")
