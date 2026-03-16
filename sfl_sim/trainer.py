"""
SimTrainer: Lightweight synchronous SFL trainer.

No async, no queues, no communication protocol.
Method-specific behavior injected via Hook pattern.
Metric extraction via callback system (subscribe/fire).

Two server training modes:
- per_client: server forward/backward per client (SFL, SCAFFOLD, GAS)
- concatenated: all activations concatenated, single server step (USFL, Mix2SFL)
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import load_dataset, distribute, get_testloader
from .models import create_model, SplitModel
from .selection import select_clients
from .client_ops import (
    ClientState, ClientResult, RoundContext, RoundResult,
    client_forward, client_backward, get_next_batch,
    snapshot_model, create_client_state, create_server_optimizer, create_criterion,
)


# ------------------------------------------------------------------
# Events fired during training (subscribe to any of these)
# ------------------------------------------------------------------
#
# "round_start"          (round_number, ctx)
# "round_end"            (round_number, ctx, round_result)
# "iteration_start"      (iteration, ctx)
# "iteration_end"        (iteration, ctx)
# "after_client_forward"  (client_id, activation, labels, images, iteration, ctx)
# "after_server_backward" (client_id, logits, loss, activation_grad, iteration, ctx)
# "after_client_backward" (client_id, iteration, ctx)
#
# Concatenated-only:
# "after_concat_forward"  (concat_act, concat_labels, logits, loss, iteration, ctx)
# "after_concat_backward" (activation_grads, iteration, ctx)
# ------------------------------------------------------------------


class SimTrainer:
    """
    Core training orchestrator.

    Hook pattern: method-specific behavior (SFL, USFL, etc.)
    Callback pattern: metric extraction without modifying trainer code.

    Usage:
        trainer.subscribe("after_server_backward", my_callback)
        trainer.train()
    """

    def __init__(self, config: Config, hook):
        self.config = config
        self.hook = hook
        self.device = torch.device(config.device)
        self.rng = np.random.RandomState(config.seed)
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Load data
        self.trainset, testset, self.num_classes = load_dataset(
            config.dataset, data_dir="./data"
        )
        self.testloader = get_testloader(testset, batch_size=config.batch_size)

        # Create model
        self.model = create_model(
            config.model, self.num_classes, config.split_layer, config.dataset
        )
        self.model.to(self.device)

        # Distribute data
        self.client_data_masks = distribute(
            self.trainset,
            config.num_clients,
            config.distribution,
            alpha=config.dirichlet_alpha,
            labels_per_client=config.labels_per_client,
            min_require_size=config.min_require_size,
            seed=config.seed,
        )
        self.all_client_ids = list(range(config.num_clients))

    # ------------------------------------------------------------------
    # Callback system
    # ------------------------------------------------------------------

    def subscribe(self, event: str, callback: Callable):
        """Register a callback for an event."""
        self._callbacks[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable):
        """Remove a callback."""
        self._callbacks[event].remove(callback)

    def fire(self, event: str, **kwargs):
        """Fire an event, calling all registered callbacks."""
        for cb in self._callbacks.get(event, ()):
            cb(**kwargs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> List[RoundResult]:
        """Main training loop."""
        results = []
        total_rounds = self.config.global_round

        for round_num in range(1, total_rounds + 1):
            t0 = time.time()

            ctx = self.hook.pre_round(self, round_num)
            self.fire("round_start", round_number=round_num, ctx=ctx)

            client_results = self.run_sfl_round(ctx)
            round_result = self.hook.post_round(self, round_num, ctx, client_results)

            round_result.metrics["round_time"] = time.time() - t0
            self.fire("round_end", round_number=round_num, ctx=ctx, round_result=round_result)
            results.append(round_result)

        return results

    def run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
        """Dispatch to per-client or concatenated mode based on hook."""
        if self.hook.server_training_mode == "concatenated":
            return self._run_concatenated(ctx)
        return self._run_per_client(ctx)

    # ------------------------------------------------------------------
    # Pattern A: Per-client (SFL, SCAFFOLD, GAS, MultiSFL)
    # ------------------------------------------------------------------

    def _run_per_client(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Interleaved per-client processing.
        One batch per client per iteration, server step per client.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device
        client_order = sorted(client_states.keys())

        max_iters = 0
        for cid in client_order:
            n_batches = len(client_states[cid].dataloader)
            total = self.config.local_epochs * n_batches
            max_iters = max(max_iters, total)

        total_loss = 0.0
        loss_count = 0

        for it in range(max_iters):
            self.fire("iteration_start", iteration=it, ctx=ctx)

            for cid in client_order:
                state = client_states[cid]
                images, labels = get_next_batch(state, device)
                activation, labels = client_forward(state, (images, labels), device)

                self.fire("after_client_forward",
                          client_id=cid, activation=activation, labels=labels,
                          images=images, iteration=it, ctx=ctx)

                if activation.size(0) == 1:
                    server_model.eval()
                else:
                    server_model.train()

                server_optimizer.zero_grad()
                logits = server_model(activation)
                loss = self.hook.compute_loss(logits, labels, criterion, ctx, client_id=cid)
                loss.backward()
                server_optimizer.step()

                total_loss += loss.item()
                loss_count += 1

                activation_grad = activation.grad.clone().detach()

                self.fire("after_server_backward",
                          client_id=cid, logits=logits, loss=loss,
                          activation_grad=activation_grad, iteration=it, ctx=ctx)

                client_backward(state, images, activation_grad, self.config)

                self.fire("after_client_backward",
                          client_id=cid, iteration=it, ctx=ctx)

            self.fire("iteration_end", iteration=it, ctx=ctx)

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)
        return self._collect_results(ctx)

    # ------------------------------------------------------------------
    # Pattern B: Concatenated (USFL, Mix2SFL)
    # ------------------------------------------------------------------

    def _run_concatenated(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Concatenated mode: all clients forward → concat → single server step →
        gradient processing (shuffle) → split back → all clients backward.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device
        client_order = sorted(client_states.keys())

        iterations = ctx.iterations
        total_loss = 0.0
        loss_count = 0

        for it in range(iterations):
            self.fire("iteration_start", iteration=it, ctx=ctx)

            # Phase 1: All clients forward
            activations = []
            labels_list = []
            images_list = []

            for cid in client_order:
                state = client_states[cid]
                images, labels = get_next_batch(state, device)
                activation, labels = client_forward(state, (images, labels), device)
                activations.append(activation)
                labels_list.append(labels)
                images_list.append(images)

                self.fire("after_client_forward",
                          client_id=cid, activation=activation, labels=labels,
                          images=images, iteration=it, ctx=ctx)

            # Concatenate
            concat_act = torch.cat(activations, dim=0)
            concat_labels = torch.cat(labels_list, dim=0)

            # Process activations (hook — for Mix2SFL SmashMix, GAS feature gen)
            concat_act, concat_labels = self.hook.process_activations(
                concat_act, concat_labels, ctx
            )

            concat_act = concat_act.detach().requires_grad_(True)
            concat_act.retain_grad()

            # Phase 2: Server forward + backward
            if concat_act.size(0) == 1:
                server_model.eval()
            else:
                server_model.train()

            server_optimizer.zero_grad()
            logits = server_model(concat_act)
            loss = criterion(logits, concat_labels)
            loss.backward()
            server_optimizer.step()

            total_loss += loss.item()
            loss_count += 1

            # Phase 3: Gradient processing (USFL gradient shuffle)
            activation_grads = concat_act.grad.clone().detach()

            self.fire("after_concat_forward",
                      concat_act=concat_act, concat_labels=concat_labels,
                      logits=logits, loss=loss, iteration=it, ctx=ctx)

            ctx.extra["client_batch_sizes"] = [a.size(0) for a in activations]
            ctx.extra["client_order"] = client_order
            ctx.extra["concat_labels"] = concat_labels

            activation_grads = self.hook.process_gradients(activation_grads, ctx)

            self.fire("after_concat_backward",
                      activation_grads=activation_grads, iteration=it, ctx=ctx)

            # Phase 4: Split grads back, client backward
            offset = 0
            for i, cid in enumerate(client_order):
                batch_size = activations[i].size(0)
                grad_slice = activation_grads[offset:offset + batch_size]
                offset += batch_size
                client_backward(client_states[cid], images_list[i], grad_slice, self.config)

                self.fire("after_client_backward",
                          client_id=cid, iteration=it, ctx=ctx)

            self.fire("iteration_end", iteration=it, ctx=ctx)

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)
        return self._collect_results(ctx)

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------

    def _collect_results(self, ctx: RoundContext) -> List[ClientResult]:
        """Collect client model snapshots after training."""
        client_order = sorted(ctx.client_states.keys())
        results = []
        for cid in client_order:
            state = ctx.client_states[cid]
            results.append(ClientResult(
                client_id=cid,
                model_state_dict=snapshot_model(state.client_model),
                dataset_size=state.dataset_size,
                label_distribution=state.label_distribution,
            ))
        return results
