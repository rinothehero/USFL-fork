import asyncio
import copy
from typing import TYPE_CHECKING

import evaluate
import torch

from utils.log_utils import vprint

from .base_stage_organizer import BaseStageOrganizer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...ws.connection import Connection
    from ..aggregator.base_aggregator import BaseAggregator
    from ..seletor.base_selector import BaseSelector
    from ..splitter.base_splitter import BaseSplitter


class SFLProxStageOrganizer(BaseStageOrganizer):
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

        self.server_models = []

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

        self.global_dict.add_event(
            "CLIENTS_SELECTED", {"client_ids": self.selected_clients}
        )

        split_models = self.splitter.split(
            self.model.get_torch_model(), self.config.__dict__
        )
        self.split_models = split_models

        model_queue = self.global_dict.get("model_queue")
        model_queue.start_insert_mode()
        self.round_start_time, self.round_end_time = (
            self.pre_round.calculate_round_end_time()
        )

        if self.config.server_model_aggregation:
            self.server_models = [
                copy.deepcopy(self.split_models[1])
                for _ in range(len(self.selected_clients))
            ]

        self.global_dict.add_event(
            "ROUND_START",
            {
                "strat_timestamp": self.round_start_time,
                "end_timestamp": self.round_end_time,
            },
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
            },
        )

    async def _in_round(self, round_number: int):
        total = 0
        total_labels = 0
        training_loss = 0.0

        # Only load metrics if in-round evaluation is enabled
        enable_inround_eval = getattr(self.config, "enable_inround_evaluation", False)
        metrics = self._load_metrics() if enable_inround_eval else []
        predictions: list["ndarray"] = []
        references: list["ndarray"] = []
        results: list[dict] = []

        original_model = copy.deepcopy(self.split_models[1])

        async def __server_side_training():
            nonlocal total, training_loss, total_labels
            nonlocal predictions, references
            while True:
                activation = await self.in_round.wait_for_activations()

                server_model = None
                if self.config.server_model_aggregation:
                    server_model = self.server_models[
                        self.selected_clients.index(activation["client_id"])
                    ]
                else:
                    server_model = self.split_models[1]

                client_proximal_term = activation["proximal_term"]
                output = await self.in_round.forward(server_model, activation)
                grad, loss = await self.in_round.backward_from_label_using_prox(
                    client_proximal_term,
                    original_model,
                    server_model,
                    output,
                    activation,
                )

                # Compute metrics (only if in-round evaluation is enabled)
                if enable_inround_eval:
                    predicted = (
                        output
                        if self.config.dataset == "sts-b"
                        else torch.argmax(output, dim=-1)
                    )
                    predictions.extend(predicted.cpu().numpy())
                    references.extend(activation["labels"].cpu().numpy())

                total += 1
                total_labels += len(activation["labels"])
                training_loss += loss

                await self.in_round.send_gradients(
                    self.connection,
                    grad,
                    activation["client_id"],
                    activation["model_index"],
                )

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

        epoch_loss = training_loss / float(total) if float(total) != 0 else 0

        if enable_inround_eval:
            for metric in metrics:
                result = metric.compute(predictions=predictions, references=references)
                results.append(result)
            vprint(
                f"TRAIN ROUND {round_number}: Loss = {epoch_loss:.4f}, Accuracy = {results}, Total labels = {total_labels}", 1
            )
        else:
            vprint(
                f"TRAIN ROUND {round_number}: Loss = {epoch_loss:.4f}, Total labels = {total_labels}", 1
            )

    async def _post_round(self, round_number: int):
        model_queue = self.global_dict.get("model_queue")
        model_queue.end_insert_mode()

        client_ids = [model[0] for model in model_queue.queue]
        self.global_dict.add_event(
            "MODEL_AGGREGATION_START", {"client_ids": client_ids}
        )

        aggregated_model = None
        if self.config.server_model_aggregation:
            ids, client_models, params = model_queue.get_all_models()
            server_models = [self.server_models[ids.index(id)] for id in client_ids]

            aggregated_client_model = self.post_round.aggregate_torch_models(
                self.aggregator, client_models, params
            )
            aggregated_server_model = self.post_round.aggregate_torch_models(
                self.aggregator, server_models, params
            )

            if isinstance(aggregated_client_model, torch.nn.ModuleDict):
                aggregated_model = aggregated_client_model
                aggregated_model.update(aggregated_server_model)
            elif aggregated_client_model is not None and hasattr(self.model, "get_split_models"):
                # FlexibleResNet with server_model_aggregation
                client_model, server_model = self.model.get_split_models()
                client_model.load_state_dict(aggregated_client_model.state_dict())
                server_model.load_state_dict(aggregated_server_model.state_dict())
                if hasattr(self.model, "sync_full_model_from_split"):
                    self.model.sync_full_model_from_split()
                vprint("Updated global model (FlexibleResNet with server aggregation)", 2)
        else:
            aggregated_client_model = self.post_round.aggregate_models(
                self.aggregator, model_queue
            )

            if isinstance(aggregated_client_model, torch.nn.ModuleDict):
                aggregated_model = aggregated_client_model
                aggregated_model.update(self.split_models[1])
            elif aggregated_client_model is not None and hasattr(self.model, "get_split_models"):
                # FlexibleResNet: Update client_model directly and sync to full model
                client_model, _ = self.model.get_split_models()
                client_model.load_state_dict(aggregated_client_model.state_dict())
                if hasattr(self.model, "sync_full_model_from_split"):
                    self.model.sync_full_model_from_split()
                vprint("Updated global model (FlexibleResNet)", 2)
                # Don't set aggregated_model - skip model_reshape which expects ModuleDict

        self.global_dict.add_event("MODEL_AGGREGATION_END", {"client_ids": client_ids})

        if aggregated_model != None:
            aggregated_model = self.aggregator.model_reshape(aggregated_model)
            self.post_round.update_global_model(aggregated_model, self.model)
            vprint("Updated global model", 2)

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        vprint(f"[Round {round_number}] Accuracy: {accuracy}", 1)

    def _load_metrics(self):
        if self.config.dataset in ["mrpc", "qqp"]:
            metrics = (evaluate.load("f1"), evaluate.load("accuracy"))

        elif self.config.dataset == "cola":
            metrics = (evaluate.load("matthews_correlation"),)

        elif self.config.dataset == "sts-b":
            metrics = (evaluate.load("pearson"), evaluate.load("spearmanr"))
        else:
            metrics = (evaluate.load("accuracy"),)

        return metrics
