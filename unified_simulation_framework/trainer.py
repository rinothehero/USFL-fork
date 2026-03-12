"""
SimTrainer — Lightweight synchronous SFL simulation trainer.

This is the heart of the unified framework. It implements the core SFL
training loop shared by all 7 methods, with two server training patterns:

  Pattern A (per_client):  SFL, SCAFFOLD, GAS, MultiSFL
    - Process one client at a time
    - Server forward/backward/step per client

  Pattern B (concatenated): USFL, Mix2SFL
    - Collect all clients' activations
    - Single server forward/backward/step on concatenated batch

All method differentiation happens through hooks (BaseMethodHook).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import UnifiedConfig
from .client_ops import (
    ClientResult,
    ClientState,
    RoundContext,
    RoundResult,
    client_backward,
    client_forward,
    get_next_batch,
    snapshot_model,
    restore_model,
)
from .methods import get_method_hook

# SFL framework imports (path setup done in config.py)
from modules.model.model import get_model
from modules.dataset.dataset import get_dataset
from modules.trainer.splitter.splitter import get_splitter
from modules.trainer.seletor.selector import get_selector
from modules.trainer.aggregator.aggregator import get_aggregator
from modules.trainer.distributer.distributer import get_distributer


class SimTrainer:
    """Lightweight simulation-only SFL trainer. No async, no queues, no polling."""

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._init_components()

        # Initialize method hook
        resources = {
            "model": self.model,
            "dataset_obj": self.dataset_obj,
            "trainset": self.trainset,
            "testloader": self.testloader,
            "distributer": self.distributer,
            "selector": self.selector,
            "aggregator": self.aggregator,
            "splitter": self.splitter,
            "client_data_masks": self.client_data_masks,
            "num_classes": self.num_classes,
        }
        self.hook = get_method_hook(config.method, config, resources)

    def _init_components(self):
        """Initialize dataset, model, distributer, selector, aggregator, splitter."""
        cfg = self.config.sfl_config

        # Dataset
        self.dataset_obj = get_dataset(cfg)
        self.dataset_obj.initialize()
        self.trainset = self.dataset_obj.get_trainset()
        self.testloader = self.dataset_obj.get_testloader(cfg.batch_size)
        self.num_classes = self.dataset_obj.get_num_classes()

        # Model
        self.model = get_model(cfg, self.num_classes)
        self.model.get_torch_model().to(self.device)

        # Data distribution
        self.distributer = get_distributer(cfg)
        client_ids = list(range(cfg.num_clients))
        self.client_data_masks = self.distributer.distribute(
            self.trainset, client_ids
        )

        # Selector, aggregator, splitter
        self.selector = get_selector(cfg)
        self.aggregator = get_aggregator(cfg)
        self.splitter = get_splitter(cfg)

    # =========================================================================
    # Main training loops
    # =========================================================================

    def train(self) -> List[RoundResult]:
        """Main entry point. Dispatches to single or multi-branch loop."""
        total_rounds = self.config.global_round
        if self.hook.is_multi_branch:
            return self._train_multi_branch(total_rounds)
        else:
            return self._train_single_branch(total_rounds)

    def _train_single_branch(self, total_rounds: int) -> List[RoundResult]:
        """Training loop for SFL, USFL, SCAFFOLD, Mix2SFL, GAS."""
        results = []
        for round_num in range(1, total_rounds + 1):
            t0 = time.time()
            ctx = self.hook.pre_round(self, round_num)
            client_results = self._run_sfl_round(ctx)
            round_result = self.hook.post_round(self, round_num, ctx, client_results)
            round_result.metrics["round_time"] = time.time() - t0
            results.append(round_result)
        return results

    def _train_multi_branch(self, total_rounds: int) -> List[RoundResult]:
        """Training loop for MultiSFL."""
        results = []
        for round_num in range(1, total_rounds + 1):
            t0 = time.time()
            self.hook.pre_round(self, round_num)
            B = self.config.multisfl_branches
            for branch in range(B):
                ctx = self.hook.pre_branch(self, round_num, branch)
                client_results = self._run_sfl_round(ctx)
                self.hook.post_branch(self, round_num, branch, ctx, client_results)
            round_result = self.hook.post_round_multi(self, round_num)
            round_result.metrics["round_time"] = time.time() - t0
            results.append(round_result)
        return results

    # =========================================================================
    # Core SFL round: dispatches to per_client or concatenated
    # =========================================================================

    def _run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
        """Execute one SFL round. Dispatches based on hook.server_training_mode."""
        if self.hook.server_training_mode == "concatenated":
            return self._run_sfl_round_concatenated(ctx)
        else:
            return self._run_sfl_round_per_client(ctx)

    # =========================================================================
    # Pattern A: Per-client sequential (SFL, SCAFFOLD, GAS, MultiSFL)
    # =========================================================================

    def _run_sfl_round_per_client(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Per-client server training (ERRATA 1, Pattern A).

        For each client:
          1. Client forward → activation
          2. Server forward on THIS client's activation → logits
          3. Server backward + optimizer.step()
          4. Client backward with activation grad

        Server model is updated K times per iteration (once per client).
        All clients start from the same global client model state.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device
        client_order = sorted(client_states.keys())

        # Snapshot base client state (ERRATA 3: reference semantics)
        client_base_state = ctx.extra.get("client_base_state")

        # Per-client iteration tracking
        # Each client iterates through its full dataset for local_epochs
        client_iters = {}
        for cid in client_order:
            state = client_states[cid]
            n_batches = len(state.dataloader)
            client_iters[cid] = self.config.local_epochs * n_batches

        # Process each client sequentially.
        # All clients share the same nn.Module reference, so we must
        # restore base state before each client and snapshot after.
        completed_snapshots = {}

        for cid in client_order:
            state = client_states[cid]

            # Restore client model to base state (all start from same global model)
            if client_base_state is not None:
                restore_model(state.client_model, client_base_state)

            total_iters = client_iters[cid]

            for it in range(total_iters):
                # Reset data iterator at epoch boundaries
                if it > 0 and it % len(state.dataloader) == 0:
                    state.data_iter = iter(state.dataloader)

                # 1. Get batch
                images, labels = get_next_batch(state, device)

                # 2. Client forward
                activation, labels = client_forward(state, (images, labels), device)

                # 3. Server forward
                if activation.size(0) == 1:
                    server_model.eval()
                else:
                    server_model.train()

                server_optimizer.zero_grad()
                logits = server_model(activation)

                # 4. Compute loss (hookable for GAS logit adjustment)
                loss = self.hook.compute_loss(
                    logits, labels, criterion, ctx, client_id=cid
                )

                # 5. Server backward + step
                loss.backward()
                server_optimizer.step()

                # 6. Get activation gradient
                activation_grad = activation.grad.clone().detach()

                # 7. Client backward
                client_backward(state, images, activation_grad, self.config)

            # Snapshot IMMEDIATELY after this client finishes training.
            # If we wait until after all clients, the shared model has only
            # the last client's state.
            completed_snapshots[cid] = snapshot_model(state.client_model)

        # Collect results from snapshots
        client_results = []
        for cid in client_order:
            state = client_states[cid]
            result = ClientResult(
                client_id=cid,
                model_state_dict=completed_snapshots[cid],
                dataset_size=state.dataset_size,
                label_distribution=state.label_distribution,
            )
            client_results.append(result)

        return client_results

    # =========================================================================
    # Pattern B: Concatenated batch (USFL, Mix2SFL)
    # =========================================================================

    def _run_sfl_round_concatenated(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Concatenated server training (ERRATA 1, Pattern B).

        For each iteration:
          1. ALL clients forward → activations (each from own state)
          2. Concatenate activations
          3. hook.process_activations() (SmashMix)
          4. Single server forward on concatenated batch → logits
          5. Server backward + optimizer.step() ONCE
          6. hook.process_gradients() (gradient shuffle)
          7. ALL clients backward with their gradient slice (each updates own state)

        IMPORTANT: Clients share a single nn.Module reference but maintain
        independent states via state_dict snapshots. Each client's backward
        restores its state before re-forward and update.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device
        client_order = sorted(client_states.keys())

        # Per-client state tracking (snapshot-based)
        # In original framework, each client has its own model copy.
        # We simulate this with state_dict snapshots since all clients
        # share the same nn.Module reference.
        client_base_state = ctx.extra.get("client_base_state")
        client_snapshots = {
            cid: {k: v.clone() for k, v in client_base_state.items()}
            for cid in client_order
        }

        iterations = ctx.iterations

        for it in range(iterations):
            # --- Phase 1: All clients forward from their own state ---
            activations = []
            labels_list = []
            images_list = []
            batch_sizes = []

            client_model = next(iter(client_states.values())).client_model

            for cid in client_order:
                state = client_states[cid]
                # Restore this client's state for forward pass
                restore_model(client_model, client_snapshots[cid])

                images, labels = get_next_batch(state, device)
                activation, labels = client_forward(state, (images, labels), device)
                activations.append(activation)
                labels_list.append(labels)
                images_list.append(images)
                batch_sizes.append(activation.size(0))

            # --- Phase 2: Activation processing (SmashMix hook) ---
            activations, labels_list = self.hook.process_activations(
                activations, labels_list, ctx
            )

            # --- Phase 3: Server forward ---
            concat_act = torch.cat(activations, dim=0)
            concat_labels = torch.cat(labels_list, dim=0)

            if concat_act.size(0) == 1:
                server_model.eval()
            else:
                server_model.train()

            concat_act = concat_act.detach().requires_grad_(True)
            concat_act.retain_grad()

            server_optimizer.zero_grad()
            logits = server_model(concat_act)
            loss = self.hook.compute_loss(logits, concat_labels, criterion, ctx)
            loss.backward()
            server_optimizer.step()

            # --- Phase 4: Get activation gradients ---
            activation_grads = concat_act.grad.clone().detach()

            # --- Phase 5: Gradient processing (shuffle hook) ---
            activation_grads = self.hook.process_gradients(
                activation_grads, activations, concat_labels, ctx
            )

            # --- Phase 6: All clients backward (restore → forward → backward → snapshot) ---
            offset = 0
            for i, cid in enumerate(client_order):
                state = client_states[cid]
                bs = batch_sizes[i]
                grad_slice = activation_grads[offset : offset + bs]
                offset += bs

                # Restore this client's state before backward
                restore_model(client_model, client_snapshots[cid])
                client_backward(state, images_list[i], grad_slice, self.config)
                # Save updated state
                client_snapshots[cid] = snapshot_model(client_model)

        # Collect results from per-client snapshots
        client_results = []
        for cid in client_order:
            result = ClientResult(
                client_id=cid,
                model_state_dict=client_snapshots[cid],
                dataset_size=client_states[cid].dataset_size,
                label_distribution=client_states[cid].label_distribution,
            )
            client_results.append(result)

        return client_results
