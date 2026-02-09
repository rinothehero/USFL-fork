import copy
from typing import TYPE_CHECKING

import torch

from utils.log_utils import vprint

from .base_stage_organizer import BaseStageOrganizer
from .dependency.torch_pruning.pruner.algorithms.emapruner import EmaPruner
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


class FitFLStageOrganizer(BaseStageOrganizer):
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

        self.selected_clients = []
        self.last_pruning_round = 0
        self.round_start_time = 0
        self.round_end_time = 0
        self.pruning_count = 0
        self.pruning_interval = self.config.pruning_interval
        self.ema_threshold = self.config.ema_threshold
        self.pruning_decision_threshold = self.config.pruning_decision_threshold
        self.pruning_min_acc = self.config.min_pruning_accuracy
        self.max_pruning_ratio = self.config.max_pruning_ratio
        self.min_pruning_ratio = self.config.min_pruning_ratio

        self.global_importance = MagnitudeImportance()
        self.global_pruner = EmaPruner(
            self.model.get_torch_model(),
            example_inputs=self._get_example_inputs(),
            ignored_layers=self._get_ignored_layers(self.model.get_torch_model()),
            importance=self.global_importance,
            ema_length=self.pruning_interval,
            ema_threshold=self.ema_threshold,
            global_pruning=False,
        )
        self.global_pruner.ema_step()
        self.parameters = self._count_parameters(self.model.get_torch_model())

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

    def _get_example_inputs(self):
        if self.config.model == "alexnet":
            return torch.randn(1, 1, 28, 28).to(self.config.device)
        else:
            return torch.randn(1, 3, 32, 32).to(self.config.device)

    # The `_evaluate_model` function was created due to legacy code requirements.
    # For general model evaluation, it is recommended to use the `evaluate_global_model` function
    # defined in the PostRound
    def _evaluate_model(self, model):
        accuracy = 0
        total = 0
        correct = 0

        model.eval()

        with torch.no_grad():
            for batch in self.testloader:
                x, y = batch
                x, y = x.to(self.config.device), y.to(self.config.device)
                output = model(x)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = correct / total

        model.train()

        return accuracy

    def _find_pruning_ratio(self):
        start = 0
        end = 1
        center = 0.5

        test_model = copy.deepcopy(self.model.get_torch_model())
        example_inputs = self._get_example_inputs()

        acc = self._evaluate_model(test_model)
        start_acc = acc
        vprint(f"Start accuracy: {start_acc}", 2)

        search_count = 0
        center_acc_diff = 10000
        adjusted_pruning_decision_threshold = self.pruning_decision_threshold

        vprint(
            f"Adjusted pruning decision threshold: {adjusted_pruning_decision_threshold}", 2
        )

        if start_acc - (self.pruning_min_acc + adjusted_pruning_decision_threshold) < 0:
            vprint(f"Model accuracy is too low. Skip pruning", 2)
            return 0

        while search_count < 7:
            search_count += 1
            ignored_layers = self._get_ignored_layers(test_model)

            pruner = MetaPruner(
                model=test_model,
                example_inputs=example_inputs,
                ignored_layers=ignored_layers,
                importance=MagnitudeImportance(),
                pruning_ratio=center,
                max_pruning_ratio=center,
            )

            pruner.step()

            acc = self._evaluate_model(test_model)
            acc_diff = start_acc - acc
            center_acc_diff = acc_diff
            vprint(f"Accuracy difference: {acc_diff}, Pruning ratio: {center}", 2)

            test_model = copy.deepcopy(self.model.get_torch_model())

            if acc_diff < adjusted_pruning_decision_threshold:
                vprint("Pruning ratio is too low", 2)
                start = center
                center = (start + end) / 2
            else:
                vprint("Pruning ratio is too high", 2)
                end = center
                center = (start + end) / 2

        if center < self.min_pruning_ratio:
            vprint("Cannot find proper pruning ratio", 2)
            center = 0

        if center > self.max_pruning_ratio:
            vprint("Cannot find proper pruning ratio", 2)
            center = self.max_pruning_ratio

        if center_acc_diff > adjusted_pruning_decision_threshold:
            vprint("Acc diff is too high. Cannot find proper pruning ratio", 2)
            center = 0

        vprint(f"Pruning ratio: {center}", 2)
        return center

    def _prune(self, round_number):
        if round_number - self.last_pruning_round >= self.pruning_interval:
            self.pruning_count += 1
            global_pruning_ratio = self._find_pruning_ratio()

            if global_pruning_ratio == 0:
                self.pruning_count -= 1
                return False

            vprint(
                f"[Round {round_number}] Global model pruning ratio: {global_pruning_ratio}", 2
            )

            self.global_pruner = EmaPruner(
                model=self.model.get_torch_model(),
                example_inputs=self._get_example_inputs(),
                ignored_layers=self._get_ignored_layers(self.model.get_torch_model()),
                global_pruning=False,
                pruning_ratio=global_pruning_ratio,
                max_pruning_ratio=global_pruning_ratio,
                importance=self.global_importance,
                ema_length=self.pruning_interval,
                _ema=copy.deepcopy(self.global_pruner.ema),
            )

            vprint(f"[Round {round_number}] Global model pruning start", 2)
            vprint(
                f"[Round {round_number}] Global model parameters before: {self.parameters}", 2
            )
            self.global_pruner.step()
            self.parameters = self._count_parameters(self.model.get_torch_model())
            vprint("Pruned global model", 2)
            vprint(
                f"[Round {round_number}] Global model parameters after: {self.parameters}", 2
            )
            self.last_pruning_round = round_number

            return True

        return False

    async def _pre_round(self, round_number: int):
        await self.pre_round.wait_for_clients()

        self.selected_clients = self.pre_round.select_clients(
            self.selector, self.connection
        )
        self.global_dict.add_event(
            "CLIENTS_SELECTED", {"client_ids": self.selected_clients}
        )

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

        await self.pre_round.send_global_model(
            self.selected_clients,
            self.model,
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
            self.global_pruner.model = self.model.get_torch_model()
            self.global_pruner.ignored_layers = self._get_ignored_layers(
                self.model.get_torch_model(),
            )
            self.global_pruner.ema.ignored_layers = self._get_ignored_layers(
                self.model.get_torch_model(),
            )
            self.global_pruner.ema_step()

        self.global_dict.add_event(
            "MODEL_PRUNING_START",
        )
        self._prune(round_number)
        self.global_dict.add_event(
            "MODEL_PRUNING_END",
        )

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)

        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        vprint(f"[Round {round_number}] Accuracy: {accuracy}", 1)
