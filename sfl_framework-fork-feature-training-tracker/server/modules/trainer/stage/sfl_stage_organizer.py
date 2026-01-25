import asyncio
import copy
from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

import evaluate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_stage_organizer import BaseStageOrganizer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

# G Measurement V2 (Oracle-based)
from utils.g_measurement import (
    GMeasurementSystem,
    snapshot_model,
    restore_model,
    get_param_names,
    compute_g_metrics,
)

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...ws.connection import Connection
    from ..aggregator.base_aggregator import BaseAggregator
    from ..seletor.base_selector import BaseSelector
    from ..splitter.base_splitter import BaseSplitter


class SFLStageOrganizer(BaseStageOrganizer):
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

        self._dataset = dataset
        self.g_measurement_system = None
        if getattr(config, "enable_g_measurement", False):
            diagnostic_rounds = getattr(config, "diagnostic_rounds", "1,3,5")
            if isinstance(diagnostic_rounds, str):
                diagnostic_rounds = [int(x) for x in diagnostic_rounds.split(",")]
            measurement_mode = getattr(config, "g_measurement_mode", "single")
            measurement_k = getattr(config, "g_measurement_k", 5)
            self.g_measurement_system = GMeasurementSystem(
                diagnostic_rounds=diagnostic_rounds,
                device=config.device,
                use_variance_g=getattr(config, "use_variance_g", False),
                measurement_mode=measurement_mode,
                measurement_k=measurement_k,
            )

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

        # FlexibleResNet uses pre-built split models (layer boundary support)
        if hasattr(self.model, "get_split_models"):
            client_model, server_model = self.model.get_split_models()
            split_models = [client_model, server_model]
        else:
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

        if (
            self.g_measurement_system is not None
            and self.g_measurement_system.is_diagnostic_round(round_number)
        ):
            import psutil, os

            process = psutil.Process(os.getpid())
            mem_before_oracle = process.memory_info().rss / (1024**3)
            print(
                f"\n[G Measurement] === Round {round_number}: Computing Oracle === (mem={mem_before_oracle:.2f}GB)"
            )

            if self.g_measurement_system.oracle_calculator is None:
                full_trainset = self._dataset.get_trainset()
                oracle_batch_size = self.config.oracle_batch_size
                full_trainloader = DataLoader(
                    dataset=full_trainset,
                    batch_size=oracle_batch_size
                    if oracle_batch_size is not None
                    else self.config.batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                self.g_measurement_system.initialize(full_trainloader)
                print(
                    f"[G Measurement] Oracle calculator initialized with {len(full_trainloader.dataset)} samples"
                )

            self.g_measurement_system.set_param_names(
                self.split_models[0],
                self.split_models[1],
            )

            if hasattr(self.model, "sync_full_model_from_split"):
                self.model.sync_full_model_from_split()

            full_model = self.model.get_torch_model().to(self.config.device)
            self.g_measurement_system.compute_oracle_split_for_round(
                self.split_models[0],
                self.split_models[1],
                full_model,
                split_layer_name=self.config.split_layer,
                config=self.config,
            )

            mem_after_oracle = process.memory_info().rss / (1024**3)
            print(
                f"[G Measurement] Oracle computed (mem={mem_after_oracle:.2f}GB, Δ={mem_after_oracle - mem_before_oracle:+.2f}GB)"
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
        metrics = self._load_metrics()
        predictions: list["ndarray"] = []
        references: list["ndarray"] = []
        results: list[dict] = []

        # G Measurement: Start accumulated/k_batch round
        if (
            self.g_measurement_system is not None
            and self.g_measurement_system.is_diagnostic_round(round_number)
            and self.g_measurement_system.measurement_mode in ("accumulated", "k_batch")
        ):
            self.g_measurement_system.start_accumulated_round()

        async def __server_side_training():
            nonlocal total, training_loss, total_labels
            nonlocal predictions, references
            while True:
                # 1) 클라이언트로부터 activation을 받음
                activation = await self.in_round.wait_for_activations()
                client_id = activation["client_id"]

                raw_act = activation["outputs"]
                if isinstance(raw_act, tuple):
                    act_tensor = raw_act[0]
                else:
                    act_tensor = raw_act

                # 4) 서버-사이드 모델 선택
                server_model = (
                    self.server_models[self.selected_clients.index(client_id)]
                    if self.config.server_model_aggregation
                    else self.split_models[1]
                )

                # 5) 서버-사이드 forward 및 backward
                output = await self.in_round.forward(
                    server_model, {"outputs": act_tensor, **activation}
                )
                is_diagnostic = (
                    self.g_measurement_system is not None
                    and self.g_measurement_system.is_diagnostic_round(round_number)
                )

                grad, loss, server_grad = await self.in_round.backward_from_label(
                    server_model,
                    output,
                    activation,
                    collect_server_grad=is_diagnostic,
                )

                # G Measurement: Collect server gradient
                if is_diagnostic and server_grad:
                    batch_weight = len(activation["labels"])
                    if self.g_measurement_system.measurement_mode in (
                        "accumulated",
                        "k_batch",
                    ):
                        # Accumulated/K-batch mode: collect (k_batch will stop after K batches internally)
                        self.g_measurement_system.accumulate_server_gradient(
                            server_grad, batch_weight
                        )
                    elif not self.g_measurement_system.server_g_tildes:
                        # Single mode: only first batch
                        self.g_measurement_system.store_server_gradient(
                            server_grad, batch_weight
                        )
                        if (
                            self.g_measurement_system.split_g_tilde is None
                            and grad is not None
                        ):
                            if isinstance(grad, tuple):
                                split_grad = tuple(
                                    g.clone().detach().cpu()
                                    for g in grad
                                    if g is not None
                                )
                                split_grad = tuple(
                                    g.mean(dim=0) if g.dim() >= 1 else g
                                    for g in split_grad
                                )
                                self.g_measurement_system.split_g_tilde = split_grad
                            else:
                                split_grad = grad.clone().detach().cpu()
                                if split_grad.dim() >= 1:
                                    self.g_measurement_system.split_g_tilde = (
                                        split_grad.mean(dim=0)
                                    )
                                else:
                                    self.g_measurement_system.split_g_tilde = split_grad
                            print(
                                f"[G Measurement] Split layer gradient collected (client={client_id})"
                            )
                        print(
                            f"[G Measurement] Server gradient collected (client={client_id}, batch_size={batch_weight})"
                        )

                # 6) metric 계산을 위한 로직 (기존과 동일)
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

                # 7) 클라이언트로 gradient 전송
                await self.in_round.send_gradients(
                    self.connection,
                    grad,
                    client_id,
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

        # G Measurement: Finalize accumulated/k_batch round
        if (
            self.g_measurement_system is not None
            and self.g_measurement_system.is_diagnostic_round(round_number)
            and self.g_measurement_system.measurement_mode in ("accumulated", "k_batch")
        ):
            self.g_measurement_system.finalize_accumulated_round()

        for metric in metrics:
            result = metric.compute(predictions=predictions, references=references)
            results.append(result)

        epoch_loss = training_loss / float(total) if float(total) != 0 else 0
        print(
            f"TRAIN ROUND {round_number}: Loss = {epoch_loss:.4f}, Accuracy = {results}, Total labels = {total_labels}"
        )

    async def _post_round(self, round_number: int):
        model_queue = self.global_dict.get("model_queue")
        model_queue.end_insert_mode()

        if self.g_measurement_system is not None:
            import psutil, os, gc, pickle

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024**3)

            if self.g_measurement_system.is_diagnostic_round(round_number):
                client_grads = {}
                client_weights = {}
                for item in model_queue.queue:
                    if len(item) >= 3:
                        client_id, model, num_samples = item[0], item[1], item[2]
                        if (
                            isinstance(num_samples, dict)
                            and "client_gradient" in num_samples
                        ):
                            grad_hex = num_samples["client_gradient"]
                            if isinstance(grad_hex, str):
                                client_grads[client_id] = pickle.loads(
                                    bytes.fromhex(grad_hex)
                                )
                            else:
                                client_grads[client_id] = grad_hex
                            del num_samples["client_gradient"]

                        weight = None
                        if isinstance(num_samples, dict):
                            measurement_weight = num_samples.get(
                                "measurement_gradient_weight"
                            )
                            if measurement_weight is not None:
                                weight = measurement_weight
                            else:
                                augmented_counts = num_samples.get(
                                    "augmented_label_counts", {}
                                )
                                if augmented_counts:
                                    weight = sum(augmented_counts.values())
                                else:
                                    weight = num_samples.get("dataset_size", 0)
                        else:
                            weight = num_samples

                        if weight is not None:
                            client_weights[client_id] = float(weight)

                if client_grads:
                    self.g_measurement_system.client_g_tildes = client_grads
                    if client_weights:
                        sorted_sizes = [
                            int(client_weights[cid])
                            for cid in sorted(client_weights.keys())
                        ]
                        print(
                            "[G Measurement] Collected gradients from "
                            f"{len(client_grads)} clients (batch_sizes={sorted_sizes})"
                        )
                    else:
                        print(
                            f"[G Measurement] Collected gradients from {len(client_grads)} clients"
                        )

                result = self.g_measurement_system.compute_g(
                    round_number, client_weights=client_weights
                )
                if result:
                    self.global_dict.add_event("G_MEASUREMENT", result.to_dict())
                    print(f"[G Measurement] Round {round_number} G computed and logged")

            self.g_measurement_system.clear_round_data()
            gc.collect()

            mem_after = process.memory_info().rss / (1024**3)
            print(
                f"[G Measurement] Memory: {mem_before:.2f}GB → {mem_after:.2f}GB (freed {mem_before - mem_after:.2f}GB)"
            )

        for item in model_queue.queue:
            if len(item) >= 3:
                num_samples = item[2]
                if isinstance(num_samples, dict) and "client_gradient" in num_samples:
                    del num_samples["client_gradient"]

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
        else:
            aggregated_client_model = self.post_round.aggregate_models(
                self.aggregator, model_queue
            )

            if isinstance(aggregated_client_model, torch.nn.ModuleDict):
                aggregated_model = aggregated_client_model
                aggregated_model.update(self.split_models[1])

        self.global_dict.add_event("MODEL_AGGREGATION_END", {"client_ids": client_ids})

        if aggregated_model != None:
            aggregated_model = self.aggregator.model_reshape(aggregated_model)
            self.post_round.update_global_model(aggregated_model, self.model)
            print("Updated global model")

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        print(f"[Round {round_number}] Accuracy: {accuracy}")

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
