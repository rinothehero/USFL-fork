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

from .client_ops import (
    ClientResult,
    ClientState,
    RoundContext,
    RoundResult,
    client_backward,
    client_forward,
    create_client_state,
    create_criterion,
    create_server_optimizer,
    get_next_batch,
    snapshot_model,
)
from .config import Config
from .data import distribute, get_testloader, load_dataset
from .models import SplitModel, create_model
from .selection import select_clients

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
        # client_data_masks는 딕셔너리
        # {
        #     0: [3401, 7822, 12045, ...],   # 클라이언트 0이 받은 이미지 인덱스들
        #     1: [501, 2233, 8891, ...],     # 클라이언트 1이 받은 이미지 인덱스들
        #     ...
        #     99: [4102, 9923, ...],         # 클라이언트 99
        # }
        self.client_data_masks = distribute(
            self.trainset,
            config.num_clients,
            config.distribution,
            alpha=config.dirichlet_alpha,
            labels_per_client=config.labels_per_client,
            min_require_size=config.min_require_size,
            seed=config.seed,
        )
        # [0, 1, 2, ..., 99]
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
            self.fire(
                "round_end", round_number=round_num, ctx=ctx, round_result=round_result
            )
            results.append(round_result)

        return results

    def run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
        """Dispatch based on hook's server_training_mode."""
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

        max_iters = 0
        for cid in client_order:
            n_batches = len(client_states[cid].dataloader)
            total = self.config.local_epochs * n_batches
            max_iters = max(max_iters, total)

        total_loss = 0.0
        loss_count = 0

        for it in range(max_iters):
            self.fire("iteration_start", iteration=it, ctx=ctx)

            # break policy: stop if any client exhausted
            if policy == "break" and any(s.exhausted for s in client_states.values()):
                break

            # SFLV2: random client order per iteration (paper spec)
            shuffled_order = list(client_order)
            self.rng.shuffle(shuffled_order)
            for cid in shuffled_order:
                state = client_states[cid]
                result = get_next_batch(state, device, policy)
                if result is None:
                    continue  # skip this client (exhausted)
                images, labels = result

                # Unified forward: client → activation → server → logits
                # Single computation graph, no detach needed
                state.client_model.train()
                state.optimizer.zero_grad()
                activation = state.client_model(images)
                activation.retain_grad()  # Capture grad at split point for callbacks

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

                # Single backward: computes gradients for BOTH server and client
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

                # Server step
                server_optimizer.step()

                # Client step (gradients already computed by loss.backward)
                if self.config.clip_grad:
                    nn.utils.clip_grad_norm_(
                        state.client_model.parameters(), self.config.clip_grad_max_norm
                    )
                state.optimizer.step()

                self.fire("after_client_backward", client_id=cid, iteration=it, ctx=ctx)

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
        policy = self.config.exhaustion_policy

        iterations = ctx.iterations
        total_loss = 0.0
        loss_count = 0

        for it in range(iterations):
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
                    continue  # exhausted, skip
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

            # break policy: stop if any client exhausted
            if policy == "break" and any(s.exhausted for s in client_states.values()):
                break

            # No active clients left (all exhausted with skip policy)
            if not activations:
                break

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

            # Scale gradients: CE(mean) divides by total concat batch,
            # but each client should see gradient as if trained on own batch.
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

            ctx.extra["client_batch_sizes"] = [a.size(0) for a in activations]
            ctx.extra["client_order"] = active_clients
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

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)
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

        iterations = ctx.iterations
        total_loss = 0.0
        loss_count = 0

        for it in range(iterations):
            self.fire("iteration_start", iteration=it, ctx=ctx)

            # Phase 1: All clients forward (graph stays connected, no detach)
            activations = []
            labels_list = []
            active_clients = []

            # break policy: stop if any client exhausted
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

            # Register hook: gradient shuffle happens during backward
            # Shuffle에는 array에서 각 클라이언트 경게, 순서, label등이 필요하다.
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

            # concat_act의 gradient가 계산되면 shuffle_hook을 실행하라는 등록
            concat_act.register_hook(make_hook(self.hook, ctx, scale_factor))

            # Phase 2: Server forward + backward (single pass for everything)
            if concat_act.size(0) == 1:
                server_model.eval()
            else:
                server_model.train()

            server_optimizer.zero_grad()
            logits = server_model(concat_act)
            loss = criterion(logits, concat_labels)

            # loss.backward() 동작 과정:
            #
            # 전체 그래프 (forward에서 만들어진 것):
            #   images_0 → [client_0 params] → act_0 ─┐
            #   images_1 → [client_1 params] → act_1 ─┤→ cat → concat_act → [server params] → logits → loss
            #   ...                                    │
            #   images_9 → [client_9 params] → act_9 ─┘
            #
            # backward는 이 그래프를 끝(loss)에서 처음(images)으로 역순 추적:
            #
            # 1) loss → logits
            #    ∂loss/∂logits 계산
            #
            # 2) logits → server params
            #    ∂loss/∂server_params 계산 → server_params.grad에 저장 (server_params를 어떻게 바꿔야 loss가 줄어드는지)
            #
            # 3) logits → concat_act
            #    ∂loss/∂concat_act 계산 → concat_act.grad에 저장 ✓
            #    ★ 이 시점에서 register_hook 발동 → gradient shuffle 실행 ★
            #    → 셔플된 gradient가 원래 gradient를 대체
            #
            # 4) concat_act → torch.cat backward → act_0, act_1, ..., act_9
            #    셔플된 gradient를 10조각으로 분배 (각 activation 크기대로)
            #
            # 5) act_i → client_i params  (10명 동시)
            #    ∂loss/∂client_i_params 계산 → client_i_params.grad에 저장 ✓
            #
            # 결과: server + 10개 client 모든 파라미터의 .grad가 채워짐
            # 이후 optimizer.step()으로 실제 파라미터 업데이트
            # 이 한 줄이 모든 일을 한다!!!
            # loss에서 출발해서 거꾸로 따라가면서 모든 파라미터의 gradient를 계산
            #
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
            results.append(
                ClientResult(
                    client_id=cid,
                    model_state_dict=snapshot_model(state.client_model),
                    dataset_size=state.dataset_size,
                    label_distribution=state.label_distribution,
                )
            )
        return results
