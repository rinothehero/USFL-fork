import asyncio
import copy
import json
import pickle
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
from utils.log_utils import vprint

# G Measurement V2 (Oracle-based)
from utils.g_measurement import (
    GMeasurementSystem,
    snapshot_model,
    restore_model,
    get_param_names,
    compute_g_metrics,
)

# Drift Measurement (SCAFFOLD-style)
from utils.drift_measurement import DriftMeasurementTracker

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...ws.connection import Connection
    from ..aggregator.base_aggregator import BaseAggregator
    from ..seletor.base_selector import BaseSelector
    from ..splitter.base_splitter import BaseSplitter


class ScaffoldSFLStageOrganizer(BaseStageOrganizer):
    """SCAFFOLD-SFL Server Stage Organizer

    Manages global control variate c and aggregates delta_c from clients.
    Only client-side model uses SCAFFOLD; server-side model unchanged.
    """
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
            measurement_mode = getattr(config, "g_measurement_mode", "single")
            measurement_k = getattr(config, "g_measurement_k", 5)
            self.g_measurement_system = GMeasurementSystem(
                diagnostic_frequency=getattr(config, "g_measure_frequency", 10),
                device=config.device,
                use_variance_g=getattr(config, "use_variance_g", False),
                measurement_mode=measurement_mode,
                measurement_k=measurement_k,
            )

        # Drift Measurement (SCAFFOLD-style)
        self.drift_tracker = None
        if getattr(config, "enable_drift_measurement", False):
            self.drift_tracker = DriftMeasurementTracker()
            vprint("[Drift] DriftMeasurementTracker initialized (SCAFFOLD-SFL)", 2)

        # Client schedule (fixed P_t for Experiment A)
        self._client_schedule_cache = None

        # SCAFFOLD: Global control variate c (persistent)
        # Per-client control variates c_i are stored client-side
        self.c = {}  # Global control variate (same structure as client model params)

    def _initialize_global_control_variate(self, client_model):
        """Initialize global control variate c to zeros"""
        if not self.c:
            for name, param in client_model.named_parameters():
                if param.requires_grad:
                    self.c[name] = torch.zeros_like(param.data).cpu()
            vprint(f"[SCAFFOLD Server] Initialized global control variate c with {len(self.c)} parameters", 2)

    def _load_client_schedule(self):
        if self._client_schedule_cache is not None:
            return
        path = getattr(self.config, "client_schedule_path", "") or ""
        if not path:
            self._client_schedule_cache = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._client_schedule_cache = json.load(f)
            vprint(f"[Schedule] Loaded fixed client schedule: {path}", 1)
        except Exception as exc:
            self._client_schedule_cache = {}
            vprint(f"[Schedule] Failed to load {path}: {exc}", 1)

    def _get_scheduled_clients(self, round_number: int):
        self._load_client_schedule()
        schedule = self._client_schedule_cache
        if not schedule:
            return None
        selected = None
        if isinstance(schedule, list):
            idx = round_number - 1
            if 0 <= idx < len(schedule):
                selected = schedule[idx]
        elif isinstance(schedule, dict):
            rounds = schedule.get("rounds")
            if isinstance(rounds, list):
                idx = round_number - 1
                if 0 <= idx < len(rounds):
                    selected = rounds[idx]
            if selected is None:
                selected = schedule.get(str(round_number), schedule.get(round_number))
        if not isinstance(selected, list):
            return None
        out = []
        for cid in selected:
            try:
                out.append(int(cid))
            except (TypeError, ValueError):
                continue
        return out if out else None

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
        scheduled_clients = self._get_scheduled_clients(round_number)
        if scheduled_clients is not None:
            self.selected_clients = scheduled_clients
            vprint(
                f"[Schedule] Round {round_number}: using fixed clients {self.selected_clients}",
                1,
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

        # SCAFFOLD: Initialize global control variate if first round
        self._initialize_global_control_variate(self.split_models[0])

        # Drift Measurement: Snapshot client model at round start
        if self.drift_tracker is not None:
            self.drift_tracker.on_round_start(self.split_models[0])

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
            vprint(
                f"\n[G Measurement] === Round {round_number}: Computing Oracle === (mem={mem_before_oracle:.2f}GB)", 1
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
                vprint(
                    f"[G Measurement] Oracle calculator initialized with {len(full_trainloader.dataset)} samples", 2
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
            vprint(
                f"[G Measurement] Oracle computed (mem={mem_after_oracle:.2f}GB, Δ={mem_after_oracle - mem_before_oracle:+.2f}GB)", 1
            )

        # SCAFFOLD: Serialize global control variate c to send to clients
        c_hex = pickle.dumps(self.c).hex()

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
                "control_variate": c_hex,  # SCAFFOLD: Send global control variate
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

            # Create persistent optimizer and criterion once per round
            server_model = self.split_models[1].to(self.config.device)
            server_optimizer = self.in_round._get_optimizer(server_model, self.config)
            server_criterion = self.in_round._get_criterion(self.config)

            while True:
                # Receive activation from client (arrival order)
                activation = await self.in_round.wait_for_activations()
                client_id = activation["client_id"]

                raw_act = activation["outputs"]
                if isinstance(raw_act, tuple):
                    act_tensor = raw_act[0]
                else:
                    act_tensor = raw_act

                # Select server-side model (aggregation mode or single model)
                if self.config.server_model_aggregation:
                    current_server_model = self.server_models[
                        self.selected_clients.index(client_id)
                    ]
                else:
                    current_server_model = server_model

                # Server-side forward
                output = await self.in_round.forward(
                    current_server_model, {"outputs": act_tensor, **activation}
                )

                is_diagnostic = (
                    self.g_measurement_system is not None
                    and self.g_measurement_system.is_diagnostic_round(round_number)
                )

                # Server-side backward with persistent optimizer
                grad, loss, server_grad = await self.in_round.backward_from_label(
                    current_server_model,
                    output,
                    activation,
                    collect_server_grad=is_diagnostic,
                    optimizer=server_optimizer,
                    criterion=server_criterion,
                )

                # G Measurement: Collect server gradient
                if is_diagnostic and server_grad:
                    batch_weight = len(activation["labels"])
                    if self.g_measurement_system.measurement_mode in (
                        "accumulated",
                        "k_batch",
                    ):
                        self.g_measurement_system.accumulate_server_gradient(
                            server_grad, batch_weight
                        )
                    elif not self.g_measurement_system.server_g_tildes:
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
                            vprint(
                                f"[G Measurement] Split layer gradient collected (client={client_id})", 2
                            )
                        vprint(
                            f"[G Measurement] Server gradient collected (client={client_id}, batch_size={batch_weight})", 2
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

                # Send gradient back to client
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

        if self.g_measurement_system is not None:
            import psutil, os, gc

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
                        vprint(
                            "[G Measurement] Collected gradients from "
                            f"{len(client_grads)} clients (batch_sizes={sorted_sizes})", 1
                        )
                    else:
                        vprint(
                            f"[G Measurement] Collected gradients from {len(client_grads)} clients", 1
                        )

                result = self.g_measurement_system.compute_g(
                    round_number, client_weights=client_weights
                )
                if result:
                    self.global_dict.add_event("G_MEASUREMENT", result.to_dict())
                    vprint(f"[G Measurement] Round {round_number} G computed and logged", 1)

            self.g_measurement_system.clear_round_data()
            gc.collect()

            mem_after = process.memory_info().rss / (1024**3)
            vprint(
                f"[G Measurement] Memory: {mem_before:.2f}GB → {mem_after:.2f}GB (freed {mem_before - mem_after:.2f}GB)", 1
            )

        for item in model_queue.queue:
            if len(item) >= 3:
                num_samples = item[2]
                if isinstance(num_samples, dict) and "client_gradient" in num_samples:
                    del num_samples["client_gradient"]

        # Drift Measurement: Collect drift metrics from clients
        if self.drift_tracker is not None:
            for item in model_queue.queue:
                if len(item) >= 3:
                    client_id, model, num_samples = item[0], item[1], item[2]
                    if isinstance(num_samples, dict):
                        drift_trajectory_sum = num_samples.get("drift_trajectory_sum", 0.0)
                        drift_batch_steps = num_samples.get("drift_batch_steps", 0)
                        drift_endpoint = num_samples.get("drift_endpoint", 0.0)
                        if drift_batch_steps > 0:
                            augmented_counts = num_samples.get("augmented_label_counts", {})
                            client_weight = (
                                sum(augmented_counts.values())
                                if augmented_counts
                                else num_samples.get("dataset_size", 0)
                            )
                            self.drift_tracker.collect_client_drift(
                                client_id,
                                drift_trajectory_sum,
                                drift_batch_steps,
                                drift_endpoint,
                                client_weight=client_weight,
                            )
                            # Collect client model state for A_cos alignment
                            if hasattr(model, "state_dict"):
                                self.drift_tracker.collect_client_model(
                                    client_id,
                                    model.state_dict(),
                                    client_weight=client_weight,
                                )
                            elif isinstance(model, dict):
                                self.drift_tracker.collect_client_model(
                                    client_id, model, client_weight=client_weight
                                )

        # SCAFFOLD: Collect delta_c from clients and update global c
        delta_c_list = []
        for item in model_queue.queue:
            if len(item) >= 3:
                client_id, model, num_samples = item[0], item[1], item[2]
                if isinstance(num_samples, dict) and "delta_c" in num_samples:
                    delta_c_hex = num_samples["delta_c"]
                    delta_c = pickle.loads(bytes.fromhex(delta_c_hex))
                    delta_c_list.append(delta_c)
                    del num_samples["delta_c"]  # Clean up
                    vprint(f"[SCAFFOLD Server] Received delta_c from client {client_id}", 2)

        if delta_c_list:
            # SCAFFOLD Algorithm 1: c <- c + (1/N) * Σ delta_c_i
            # Simple sum over participating clients, scaled by 1/N (total clients)
            N = self.config.num_clients

            for name in self.c:
                # Sum delta_c from all participating clients (unweighted)
                delta_c_sum = sum(
                    delta_c[name].cpu()
                    for delta_c in delta_c_list
                    if name in delta_c
                )
                # c <- c + (1/N) * Σ delta_c_i
                self.c[name] = self.c[name] + delta_c_sum / N

            vprint(f"[SCAFFOLD Server] Updated global control variate c from {len(delta_c_list)} clients (N={N})", 2)

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

        # Drift Measurement: Compute G_drift after aggregation
        if self.drift_tracker is not None:
            if hasattr(self.model, "get_split_models"):
                new_client_model, _ = self.model.get_split_models()
            else:
                new_client_model = self.split_models[0]

            drift_result = self.drift_tracker.on_round_end(round_number, new_client_model)
            if drift_result:
                self.global_dict.add_event("DRIFT_MEASUREMENT", drift_result.to_dict())

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        vprint(f"[Round {round_number}/{self.config.global_round}] Accuracy: {accuracy:.4f}", 1)

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
