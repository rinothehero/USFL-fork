import time
from typing import TYPE_CHECKING

import torch

from utils.log_utils import vprint

from .base_stage_organizer import BaseStageOrganizer
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


class PruneFLStageOrganizer(BaseStageOrganizer):
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

    def _prune(self):
        vprint(f"Parameter count before pruning: {self.parameters}", 2)

        pruner = MetaPruner(
            model=self.model.get_torch_model(),
            example_inputs=self._get_example_inputs(),
            ignored_layers=self._get_ignored_layers(self.model.get_torch_model()),
            importance=MagnitudeImportance(),
            pruning_ratio=self.config.initial_pruning_ratio,
            max_pruning_ratio=self.config.initial_pruning_ratio,
        )

        pruner.step()
        self.parameters = self._count_parameters(self.model.get_torch_model())

        vprint(f"Parameter count after pruning: {self.parameters}", 2)

    async def _pre_round(self, round_number: int):
        await self.pre_round.wait_for_clients()

        if round_number != 1:
            self.selected_clients = self.pre_round.select_clients(
                self.selector, self.connection
            )

            self.round_start_time, self.round_end_time = (
                self.pre_round.calculate_round_end_time()
            )

        else:
            self.selected_clients = [self.config.initial_pruning_device_id]
            self.round_start_time = time.time()
            self.round_end_time = self.round_start_time + INF

        self.global_dict.add_event(
            "CLIENTS_SELECTED", {"client_ids": self.selected_clients}
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

        await self.pre_round.send_global_model(
            self.selected_clients,
            self.model,
            self.connection,
            {
                "round_number": round_number,
                "round_end_time": self.round_end_time,
                "round_start_time": self.round_start_time,
                "signiture": model_queue.get_signiture(),
                "local_epochs": (
                    self.config.local_epochs
                    if round_number != 1
                    else self.config.initial_pruning_epoch
                ),
            },
        )

    async def _in_round(self, round_number: int):
        await self.in_round.wait_for_model_submission(
            self.selected_clients, self.round_end_time
        )

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

        if updated_torch_model != None:
            self.post_round.update_global_model(updated_torch_model, self.model)
            vprint("Updated global model", 2)

        if round_number == 1:
            self._prune()
            vprint("Pruning completed for round 1", 2)

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)

        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        vprint(f"[Round {round_number}] Accuracy: {accuracy}", 1)
