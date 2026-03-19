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

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn

from .client_ops import (
    ClientResult,
    ClientState,
    RoundContext,
    RoundResult,
    client_backward,
    client_forward,
    get_next_batch,
    snapshot_model,
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

    # Attributes set by entry.py before train():
    #   config, hook, device, rng, _callbacks,
    #   trainset, testloader, num_classes, model,
    #   client_data_masks, selection_schedule, all_client_ids

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

            # Log training volume (first round only)
            if round_num == 1:
                epochs_done = ctx.extra.get("epochs_done", 0)
                bpe = ctx.extra.get("batches_per_epoch_actual", 0)
                print(
                    f"[sfl_sim] epochs={epochs_done}/{self.config.local_epochs}, "
                    f"batches/epoch={bpe}, "
                    f"clients={len(ctx.selected_client_ids)}, "
                    f"policy={self.config.exhaustion_policy}",
                    flush=True,
                )

            round_result = self.hook.post_round(self, round_num, ctx, client_results)

            round_result.metrics["round_time"] = time.time() - t0
            self.fire(
                "round_end", round_number=round_num, ctx=ctx, round_result=round_result
            )
            results.append(round_result)

        return results

    def run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
        """Dispatch based on hook's server_training_mode.

        If the hook provides run_round(), use it (MultiSFL full-round override).
        Otherwise dispatch to per_client / concatenated / concatenated_fused.
        """
        custom = self.hook.run_round(self, ctx)
        if custom is not None:
            return custom

        mode = self.hook.server_training_mode
        if mode == "concatenated":
            return self._run_concatenated(ctx)
        elif mode == "concatenated_fused":
            return self._run_concatenated_fused(ctx)
        return self._run_per_client(ctx)

    # ------------------------------------------------------------------
    # Pattern A: Per-client (SFL, SCAFFOLD, GAS, MultiSFL)
    # ------------------------------------------------------------------

    def _run_per_client(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Interleaved per-client processing.
        One batch per client per iteration, server step per client.

        Uses unified forward-backward: client→server in one graph,
        loss.backward() computes gradients for both models in a single pass.
        No detach, no re-forward.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device
        client_order = sorted(client_states.keys())
        policy = self.config.exhaustion_policy

        max_batches = max(len(s.dataloader) for s in client_states.values())
        local_epochs = self.config.local_epochs

        total_loss = 0.0
        loss_count = 0
        it = 0

        for epoch in range(local_epochs):
            # Reset all clients at epoch boundary
            for state in client_states.values():
                state.data_iter = iter(state.dataloader)
                state.exhausted = False

            for batch_idx in range(max_batches):
                self.fire("iteration_start", iteration=it, ctx=ctx)

                # break policy: stop epoch if any client exhausted
                if policy == "break" and any(s.exhausted for s in client_states.values()):
                    break

                # SFLV2: random client order per iteration (paper spec)
                shuffled_order = list(client_order)
                self.rng.shuffle(shuffled_order)
                for cid in shuffled_order:
                    state = client_states[cid]
                    result = get_next_batch(state, device, policy)
                    if result is None:
                        continue  # skip this client (exhausted this epoch)
                    images, labels = result

                    # Unified forward: client → activation → server → logits
                    # Single computation graph, no detach needed
                    state.client_model.train()
                    state.optimizer.zero_grad()
                    activation = state.client_model(images)
                    activation.retain_grad()

                    self.fire(
                        "after_client_forward",
                        client_id=cid,
                        activation=activation,
                        labels=labels,
                        images=images,
                        iteration=it,
                        ctx=ctx,
                    )

                    if activation.size(0) == 1:
                        server_model.eval()
                    else:
                        server_model.train()

                    server_optimizer.zero_grad()
                    logits = server_model(activation)
                    loss = self.hook.compute_loss(
                        logits, labels, criterion, ctx, client_id=cid
                    )

                    loss.backward()

                    total_loss += loss.item()
                    loss_count += 1

                    self.fire(
                        "after_server_backward",
                        client_id=cid,
                        logits=logits,
                        loss=loss,
                        activation_grad=activation.grad,
                        iteration=it,
                        ctx=ctx,
                    )

                    server_optimizer.step()

                    if self.config.clip_grad:
                        nn.utils.clip_grad_norm_(
                            state.client_model.parameters(), self.config.clip_grad_max_norm
                        )
                    state.optimizer.step()

                    self.fire("after_client_backward", client_id=cid, iteration=it, ctx=ctx)

                self.fire("iteration_end", iteration=it, ctx=ctx)
                it += 1

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)

        ctx.extra["epochs_done"] = local_epochs
        ctx.extra["batches_per_epoch_actual"] = max_batches
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
        policy = self.config.exhaustion_policy

        batches_per_epoch = ctx.extra.get("batches_per_epoch",
            max(len(s.dataloader) for s in client_states.values()))
        local_epochs = self.config.local_epochs

        total_loss = 0.0
        loss_count = 0
        it = 0

        for epoch in range(local_epochs):
            # Reset all clients at epoch boundary
            for state in client_states.values():
                state.data_iter = iter(state.dataloader)
                state.exhausted = False

            for batch_idx in range(batches_per_epoch):
                self.fire("iteration_start", iteration=it, ctx=ctx)

                # Phase 1: All clients forward (skip exhausted ones)
                activations = []
                labels_list = []
                images_list = []
                active_clients = []

                for cid in client_order:
                    state = client_states[cid]
                    result = get_next_batch(state, device, policy)
                    if result is None:
                        continue
                    images, labels = result
                    activation, labels = client_forward(state, (images, labels), device)
                    activations.append(activation)
                    labels_list.append(labels)
                    images_list.append(images)
                    active_clients.append(cid)

                    self.fire(
                        "after_client_forward",
                        client_id=cid,
                        activation=activation,
                        labels=labels,
                        images=images,
                        iteration=it,
                        ctx=ctx,
                    )

                # break policy: stop epoch if any client exhausted
                if policy == "break" and any(s.exhausted for s in client_states.values()):
                    break

                # No active clients left
                if not activations:
                    break

                # Concatenate
                concat_act = torch.cat(activations, dim=0)
                concat_labels = torch.cat(labels_list, dim=0)

                ctx.extra["client_batch_sizes"] = [a.size(0) for a in activations]
                ctx.extra["client_order"] = active_clients

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
                loss = self.hook.compute_loss(logits, concat_labels, criterion, ctx)
                loss.backward()
                server_optimizer.step()

                total_loss += loss.item()
                loss_count += 1

                # Phase 3: Gradient processing
                activation_grads = concat_act.grad.clone().detach()

                if self.config.scale_client_grad:
                    activation_grads *= len(active_clients)

                self.fire(
                    "after_concat_forward",
                    concat_act=concat_act,
                    concat_labels=concat_labels,
                    logits=logits,
                    loss=loss,
                    iteration=it,
                    ctx=ctx,
                )

                ctx.extra["concat_labels"] = concat_labels
                activation_grads = self.hook.process_gradients(activation_grads, ctx)

                self.fire(
                    "after_concat_backward",
                    activation_grads=activation_grads,
                    iteration=it,
                    ctx=ctx,
                )

                # Phase 4: Split grads back, client backward
                offset = 0
                for i, cid in enumerate(active_clients):
                    batch_size = activations[i].size(0)
                    grad_slice = activation_grads[offset : offset + batch_size]
                    offset += batch_size
                    client_backward(
                        client_states[cid], images_list[i], grad_slice, self.config
                    )
                    self.fire("after_client_backward", client_id=cid, iteration=it, ctx=ctx)

                self.fire("iteration_end", iteration=it, ctx=ctx)
                it += 1

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)

        ctx.extra["epochs_done"] = local_epochs
        ctx.extra["batches_per_epoch_actual"] = batches_per_epoch
        return self._collect_results(ctx)

    # ------------------------------------------------------------------
    # Pattern B-fused: Concatenated with register_hook (USFL optimized)
    # ------------------------------------------------------------------

    def _run_concatenated_fused(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Optimized concatenated mode using register_hook.

        Single forward + single backward per client (no re-forward).
        Graph stays connected: client → cat → hook(shuffle) → server → loss.
        Only works when process_activations is pass-through.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device
        client_order = sorted(client_states.keys())
        policy = self.config.exhaustion_policy

        batches_per_epoch = ctx.extra.get("batches_per_epoch",
            max(len(s.dataloader) for s in client_states.values()))
        local_epochs = self.config.local_epochs

        total_loss = 0.0
        loss_count = 0
        it = 0

        _debug_r1 = (ctx.round_number == 1)

        for epoch in range(local_epochs):
            # Reset all clients at epoch boundary
            for state in client_states.values():
                state.data_iter = iter(state.dataloader)
                state.exhausted = False

            _epoch_steps = 0
            _min_active = len(client_order)

            for batch_idx in range(batches_per_epoch):
                self.fire("iteration_start", iteration=it, ctx=ctx)

                # Phase 1: All clients forward (graph stays connected, no detach)
                activations = []
                labels_list = []
                active_clients = []

                # break policy: stop epoch if any client exhausted
                if policy == "break" and any(s.exhausted for s in client_states.values()):
                    break

                for cid in client_order:
                    state = client_states[cid]
                    result = get_next_batch(state, device, policy)
                    if result is None:
                        continue
                    images, labels = result
                    labels_list.append(labels)

                    state.client_model.train()
                    state.optimizer.zero_grad()
                    activation = state.client_model(images)
                    activation.retain_grad()
                    activations.append(activation)
                    active_clients.append(cid)

                    self.fire(
                        "after_client_forward",
                        client_id=cid,
                        activation=activation,
                        labels=labels,
                        images=images,
                        iteration=it,
                        ctx=ctx,
                    )

                # No active clients left
                if not activations:
                    break

                # Concatenate (graph preserved through torch.cat)
                concat_act = torch.cat(activations, dim=0)
                concat_labels = torch.cat(labels_list, dim=0)
                concat_act.retain_grad()

                ctx.extra["client_batch_sizes"] = [a.size(0) for a in activations]
                ctx.extra["client_order"] = active_clients
                ctx.extra["concat_labels"] = concat_labels

                scale_factor = len(active_clients) if self.config.scale_client_grad else 1

                def make_hook(hook_ref, ctx_ref, scale):
                    def shuffle_hook(grad):
                        g = grad.clone()
                        if scale > 1:
                            g *= scale
                        return hook_ref.process_gradients(g, ctx_ref)
                    return shuffle_hook

                concat_act.register_hook(make_hook(self.hook, ctx, scale_factor))

                # Phase 2: Server forward + backward (single pass for everything)
                if concat_act.size(0) == 1:
                    server_model.eval()
                else:
                    server_model.train()

                server_optimizer.zero_grad()
                logits = server_model(concat_act)
                loss = criterion(logits, concat_labels)
                loss.backward()

                total_loss += loss.item()
                loss_count += 1

                self.fire(
                    "after_concat_forward",
                    concat_act=concat_act,
                    concat_labels=concat_labels,
                    logits=logits,
                    loss=loss,
                    iteration=it,
                    ctx=ctx,
                )

                self.fire(
                    "after_concat_backward",
                    activation_grads=concat_act.grad,
                    iteration=it,
                    ctx=ctx,
                )

                # Step all active optimizers (gradients already computed by loss.backward)
                server_optimizer.step()
                for cid in active_clients:
                    state = client_states[cid]
                    if self.config.clip_grad:
                        nn.utils.clip_grad_norm_(
                            state.client_model.parameters(), self.config.clip_grad_max_norm
                        )
                    state.optimizer.step()

                    self.fire("after_client_backward", client_id=cid, iteration=it, ctx=ctx)

                self.fire("iteration_end", iteration=it, ctx=ctx)
                it += 1
                _epoch_steps += 1
                _min_active = min(_min_active, len(active_clients))

            if _debug_r1:
                print(
                    f"  [epoch {epoch+1}/{local_epochs}] "
                    f"steps={_epoch_steps}, "
                    f"active_clients: min={_min_active}/{len(client_order)}",
                    flush=True,
                )

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)

        ctx.extra["epochs_done"] = local_epochs
        ctx.extra["batches_per_epoch_actual"] = batches_per_epoch
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
            results.append(
                ClientResult(
                    client_id=cid,
                    model_state_dict=snapshot_model(state.client_model),
                    dataset_size=state.dataset_size,
                    label_distribution=state.label_distribution,
                )
            )
        return results
