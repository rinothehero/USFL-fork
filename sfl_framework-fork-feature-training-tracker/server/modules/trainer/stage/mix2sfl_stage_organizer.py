import asyncio
import json
import math
import os
import random
from typing import TYPE_CHECKING, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_stage_organizer import BaseStageOrganizer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound
from utils.log_utils import vprint
from utils.g_measurement import GMeasurementSystem
from utils.drift_measurement import DriftMeasurementTracker
from utils.experiment_a_probe import compute_split_probe_directions, build_probe_loader

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...ws.connection import Connection
    from ...ws.inmemory_connection import InMemoryConnection
    from ..aggregator.base_aggregator import BaseAggregator
    from ..seletor.base_selector import BaseSelector
    from ..splitter.base_splitter import BaseSplitter


class Mix2SFLStageOrganizer(BaseStageOrganizer):
    def __init__(
        self,
        config: "Config",
        connection: Union["Connection", "InMemoryConnection"],
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
        self._client_schedule_cache = None

        self._dataset = dataset
        self.probe_loader = self.testloader
        self.probe_max_batches = max(int(getattr(config, "probe_max_batches", 1)), 1)
        try:
            self.probe_loader, probe_meta = build_probe_loader(
                default_loader=self.testloader,
                train_dataset=self._dataset.get_trainset(),
                test_dataset=self._dataset.get_testset(),
                source=getattr(config, "probe_source", "test"),
                indices_path=getattr(config, "probe_indices_path", ""),
                num_samples=int(getattr(config, "probe_num_samples", 0)),
                batch_size=int(getattr(config, "probe_batch_size", 0)),
                seed=int(getattr(config, "probe_seed", getattr(config, "seed", 0))),
                class_balanced=bool(
                    getattr(config, "probe_class_balanced", False)
                ),
                class_balanced_batches=bool(
                    getattr(config, "probe_class_balanced_batches", False)
                ),
            )
            if self.probe_loader is None:
                self.probe_loader = self.testloader
            vprint(
                f"[Probe] source={probe_meta.get('source', 'test')} selected={probe_meta.get('selected_samples', 0)} "
                f"batch={probe_meta.get('batch_size', getattr(self.config, 'batch_size', 0))} "
                f"max_batches={self.probe_max_batches}",
                1,
            )
        except Exception as exc:
            self.probe_loader = self.testloader
            vprint(f"[Probe] Failed to build dedicated probe loader: {exc}", 1)

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
            if getattr(config, "save_mu_c", False):
                self.drift_tracker.enable_save_mu_c()
                vprint("[Drift] μ_c saving enabled (IID baseline)", 1)
            ref_path = getattr(config, "reference_mu_c_path", "") or ""
            if ref_path:
                self.drift_tracker.load_reference_mu_c(ref_path)
            vprint("[Drift] DriftMeasurementTracker initialized (Mix2SFL)", 2)

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

    def _soft_cross_entropy(self, logits: torch.Tensor, soft_targets: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def _get_smashmix_ns(self, active_count: int) -> int:
        ratio = max(0.0, self.config.mix2sfl_smashmix_ns_ratio)
        ns = int(active_count * ratio)
        if active_count <= 1:
            return 0
        return min(ns, active_count - 1)

    def _sample_lambda(self) -> float:
        dist = getattr(self.config, "mix2sfl_smashmix_lambda_dist", "uniform")
        if dist == "beta":
            alpha = getattr(self.config, "mix2sfl_smashmix_beta_alpha", 1.0)
            return random.betavariate(alpha, alpha)
        return random.random()

    def _detach_outputs(self, outputs):
        if isinstance(outputs, tuple):
            detached = []
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    detached.append(item.detach().clone())
                else:
                    return None
            return tuple(detached)
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().clone()
        return None

    async def _pre_round(self, round_number: int):
        await self.pre_round.wait_for_client_informations()
        await self.pre_round.wait_for_clients()

        client_informations = self.global_dict.get("client_informations")
        scheduled_clients = self._get_scheduled_clients(round_number)
        if scheduled_clients is not None:
            self.selected_clients = scheduled_clients
            vprint(
                f"[Schedule] Round {round_number}: using fixed clients {self.selected_clients}",
                1,
            )
        else:
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

        dataset_sizes = {
            client_information["client_id"]: client_information["dataset"]["size"]
            for client_information in client_informations.values()
            if client_information["client_id"] in self.selected_clients
        }
        iterations_per_client = {
            client_id: dataset_size // self.config.batch_size
            for client_id, dataset_size in dataset_sizes.items()
        }
        if iterations_per_client:
            max_iterations = max(1, max(iterations_per_client.values()))
        else:
            max_iterations = 1

        # FlexibleResNet uses pre-built split models (layer boundary support)
        if hasattr(self.model, "get_split_models"):
            client_model, server_model = self.model.get_split_models()
            split_models = [client_model, server_model]
        else:
            split_models = self.splitter.split(
                self.model.get_torch_model(), self.config.__dict__
            )
        self.split_models = split_models

        # Drift Measurement: Snapshot client/server model at round start
        if self.drift_tracker is not None:
            self.drift_tracker.on_round_start(self.split_models[0], self.split_models[1])
            try:
                c_client, c_server, probe_meta = compute_split_probe_directions(
                    self.split_models[0],
                    self.split_models[1],
                    self.probe_loader,
                    self.config.device,
                    max_batches=self.probe_max_batches,
                )
                self.drift_tracker.set_probe_directions(c_client, c_server, probe_meta)
            except Exception as exc:
                vprint(f"[Drift][ExpA] Probe direction failed: {exc}", 1)

        model_queue = self.global_dict.get("model_queue")
        model_queue.start_insert_mode()
        self.round_start_time, self.round_end_time = (
            self.pre_round.calculate_round_end_time()
        )

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
                "iterations": max_iterations,
                "split_count": len(self.split_models),
                "model_index": 0,
            },
        )

    async def _in_round(self, round_number: int):
        # G Measurement: Start accumulated/k_batch round
        if (
            self.g_measurement_system is not None
            and self.g_measurement_system.is_diagnostic_round(round_number)
            and self.g_measurement_system.measurement_mode in ("accumulated", "k_batch")
        ):
            self.g_measurement_system.start_accumulated_round()

        async def __server_side_training():
            iteration_count = 0
            server_model = self.split_models[1].to(self.config.device)
            optimizer = self.in_round._get_optimizer(server_model, self.config)
            criterion = self.in_round._get_criterion(self.config)

            while True:
                activations = await self.in_round.wait_for_concatenated_activations(
                    self.selected_clients
                )

                activation_length_per_client = {
                    activation["client_id"]: len(activation["labels"])
                    for activation in activations
                }

                non_empty_activations = [
                    act for act in activations if len(act["labels"]) > 0
                ]

                if not non_empty_activations:
                    for act in activations:
                        cid = act["client_id"]
                        empty_grad = torch.tensor([]).to(self.config.device)
                        await self.in_round.send_gradients(
                            self.connection, empty_grad, cid, 0
                        )
                    continue

                concatenated_activations = {}
                for activation in non_empty_activations:
                    concatenated_activations = self._concatenate_activations(
                        concatenated_activations, activation
                    )

                outputs = await self.in_round.forward(
                    server_model, concatenated_activations
                )

                inputs = concatenated_activations["outputs"]
                if isinstance(inputs, tuple):
                    for item in inputs:
                        if isinstance(item, torch.Tensor) and item.requires_grad:
                            item.retain_grad()
                else:
                    if isinstance(inputs, torch.Tensor) and inputs.requires_grad:
                        inputs.retain_grad()

                labels = concatenated_activations["labels"].to(self.config.device)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()

                if isinstance(inputs, tuple):
                    grad = tuple(
                        item.grad.clone().detach()
                        for item in inputs
                        if isinstance(item, torch.Tensor)
                    )
                else:
                    grad = inputs.grad.clone().detach()

                if isinstance(inputs, tuple):
                    for item in inputs:
                        if isinstance(item, torch.Tensor):
                            item.grad.detach_()
                else:
                    inputs.grad.detach_()

                is_diagnostic = (
                    self.g_measurement_system is not None
                    and self.g_measurement_system.is_diagnostic_round(round_number)
                )
                # G Measurement: Collect server gradient
                if is_diagnostic:
                    server_grad = {
                        name: param.grad.clone().detach().cpu()
                        for name, param in server_model.named_parameters()
                        if param.grad is not None
                    }
                    batch_weight = sum(
                        len(act["labels"]) for act in non_empty_activations
                    )
                    if self.g_measurement_system.measurement_mode in (
                        "accumulated",
                        "k_batch",
                    ):
                        # Accumulated/K-batch mode: collect (k_batch will stop after K batches internally)
                        self.g_measurement_system.accumulate_server_gradient(
                            server_grad, batch_weight
                        )
                    elif iteration_count == 0:
                        # Single mode: only first batch
                        self.g_measurement_system.store_server_gradient(
                            server_grad, batch_weight
                        )

                    if self.g_measurement_system.split_g_tilde is None:
                        if isinstance(grad, tuple):
                            split_grad = tuple(
                                g.clone().detach().cpu() for g in grad if g is not None
                            )
                            split_grad = tuple(
                                g.mean(dim=0) if g.dim() >= 1 else g for g in split_grad
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

                if getattr(self.config, "mix2sfl_smashmix_enabled", True):
                    active_clients = [act["client_id"] for act in non_empty_activations]
                    active_count = len(active_clients)
                    ns = self._get_smashmix_ns(active_count)

                    if ns > 0:
                        activation_by_client = {
                            act["client_id"]: act for act in non_empty_activations
                        }

                        for client_id in active_clients:
                            candidates = [
                                cid for cid in active_clients if cid != client_id
                            ]
                            num_partners = min(ns, len(candidates))
                            if num_partners <= 0:
                                continue

                            partners = random.sample(candidates, num_partners)
                            for partner_id in partners:
                                act_i = activation_by_client[client_id]
                                act_j = activation_by_client[partner_id]

                                outputs_i = self._detach_outputs(act_i["outputs"])
                                outputs_j = self._detach_outputs(act_j["outputs"])
                                if outputs_i is None or outputs_j is None:
                                    continue

                                lam = self._sample_lambda()
                                if isinstance(outputs_i, tuple):
                                    if len(outputs_i) != len(outputs_j):
                                        continue
                                    # Check batch size compatibility
                                    if any(
                                        si.shape[0] != sj.shape[0]
                                        for si, sj in zip(outputs_i, outputs_j)
                                    ):
                                        continue
                                    mixed_outputs = tuple(
                                        lam * si + (1.0 - lam) * sj
                                        for si, sj in zip(outputs_i, outputs_j)
                                    )
                                else:
                                    # Check batch size compatibility
                                    if outputs_i.shape[0] != outputs_j.shape[0]:
                                        continue
                                    mixed_outputs = (
                                        lam * outputs_i + (1.0 - lam) * outputs_j
                                    )

                                labels_i = act_i["labels"].to(self.config.device)
                                labels_j = act_j["labels"].to(self.config.device)
                                if labels_i.shape != labels_j.shape:
                                    continue

                                onehot_i = F.one_hot(
                                    labels_i, num_classes=self.num_classes
                                ).float()
                                onehot_j = F.one_hot(
                                    labels_j, num_classes=self.num_classes
                                ).float()

                                mixed_labels = lam * onehot_i + (1.0 - lam) * onehot_j

                                mixed_logits = await self.in_round.forward(
                                    server_model, {"outputs": mixed_outputs}
                                )
                                mixed_loss = self._soft_cross_entropy(
                                    mixed_logits, mixed_labels
                                )
                                mixed_loss.backward()

                optimizer.step()
                if self.drift_tracker is not None:
                    self.drift_tracker.accumulate_server_drift(server_model)

                gradmix_enabled = getattr(self.config, "mix2sfl_gradmix_enabled", True)
                if gradmix_enabled:
                    phi = max(0.0, min(1.0, self.config.mix2sfl_gradmix_phi))
                    group_size = int(math.floor(len(non_empty_activations) * phi))
                    group_size = min(group_size, len(non_empty_activations))

                    if group_size > 0:
                        cprime = set(
                            random.sample(
                                [act["client_id"] for act in non_empty_activations],
                                group_size,
                            )
                        )
                    else:
                        cprime = set()
                else:
                    cprime = set()

                client_grads = {}
                start = 0
                for act in non_empty_activations:
                    cid = act["client_id"]
                    length = activation_length_per_client[cid]
                    end = start + length

                    if isinstance(grad, tuple):
                        g_slice = tuple(g[start:end].clone() for g in grad)
                    else:
                        g_slice = grad[start:end].clone()
                    client_grads[cid] = g_slice
                    start = end

                if cprime:
                    reduce = getattr(self.config, "mix2sfl_gradmix_reduce", "sum")

                    # Filter clients with matching batch sizes
                    matching_cprime = set()
                    ref_shape = None
                    for cid in cprime:
                        if cid not in client_grads:
                            continue
                        g = client_grads[cid]
                        g_shape = g[0].shape[0] if isinstance(g, tuple) else g.shape[0]
                        if ref_shape is None:
                            ref_shape = g_shape
                        if g_shape == ref_shape:
                            matching_cprime.add(cid)

                    grad_list = [client_grads[cid] for cid in matching_cprime]

                    if grad_list and isinstance(grad_list[0], tuple):
                        summed = []
                        for items in zip(*grad_list):
                            total = torch.zeros_like(items[0])
                            for item in items:
                                total += item
                            summed.append(total)
                        broadcast_grad = tuple(summed)
                    elif grad_list:
                        broadcast_grad = torch.zeros_like(grad_list[0])
                        for g_item in grad_list:
                            broadcast_grad += g_item
                    else:
                        broadcast_grad = None

                    if reduce == "mean" and len(grad_list) > 0:
                        if isinstance(broadcast_grad, tuple):
                            broadcast_grad = tuple(
                                g / float(len(grad_list)) for g in broadcast_grad
                            )
                        else:
                            broadcast_grad = broadcast_grad / float(len(grad_list))
                else:
                    broadcast_grad = None
                    matching_cprime = set()

                for act in non_empty_activations:
                    cid = act["client_id"]
                    if cid in matching_cprime and broadcast_grad is not None:
                        await self.in_round.send_gradients(
                            self.connection, broadcast_grad, cid, 0
                        )
                    else:
                        await self.in_round.send_gradients(
                            self.connection, client_grads[cid], cid, 0
                        )

                for act in activations:
                    cid = act["client_id"]
                    if activation_length_per_client[cid] == 0:
                        empty_grad = torch.tensor([]).to(self.config.device)
                        await self.in_round.send_gradients(
                            self.connection, empty_grad, cid, 0
                        )

                iteration_count += 1

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
                            drift_sample_count = num_samples.get("drift_sample_count")
                            if drift_sample_count is not None and drift_sample_count > 0:
                                client_weight = drift_sample_count
                            else:
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
            if updated_torch_model is not None:
                updated_torch_model = self.aggregator.model_reshape(updated_torch_model)
                self.post_round.update_global_model(updated_torch_model, self.model)
        elif updated_torch_model is not None and hasattr(self.model, "get_split_models"):
            # FlexibleResNet: Update client_model directly and sync to full model
            client_model, _ = self.model.get_split_models()
            client_model.load_state_dict(updated_torch_model.state_dict())
            if hasattr(self.model, "sync_full_model_from_split"):
                self.model.sync_full_model_from_split()

        # Drift Measurement: Compute G_drift after aggregation
        if self.drift_tracker is not None:
            if hasattr(self.model, "get_split_models"):
                new_client_model, _ = self.model.get_split_models()
            else:
                new_client_model = self.split_models[0]

            new_server_model = self.split_models[1]
            drift_result = self.drift_tracker.on_round_end(
                round_number, new_client_model, new_server_model
            )
            if drift_result:
                self.global_dict.add_event("DRIFT_MEASUREMENT", drift_result.to_dict())

            # Save μ_c vectors on last round (for IID baseline generation)
            if (
                self.drift_tracker._save_mu_c
                and round_number == self.config.global_round
            ):
                result_dir = getattr(self.config, "result_output_dir", "") or "."
                mu_c_path = os.path.join(result_dir, "sfl_iid_mu_c.pt")
                self.drift_tracker.save_mu_c_vectors(mu_c_path)

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        vprint(f"[Round {round_number}/{self.config.global_round}] Accuracy: {accuracy:.4f}", 1)
