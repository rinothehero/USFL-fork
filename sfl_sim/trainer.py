"""
SimTrainer: Lightweight synchronous SFL trainer.

No async, no queues, no communication protocol.
Method-specific behavior injected via Hook pattern.
"""
from __future__ import annotations

import copy
import time
from typing import List

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


class SimTrainer:
    """
    Core training orchestrator. All methods share this trainer.

    Training loop:
        for round in rounds:
            ctx = hook.pre_round(trainer, round)
            results = trainer.run_sfl_round(ctx)
            round_result = hook.post_round(trainer, round, ctx, results)
    """

    def __init__(self, config: Config, hook):
        self.config = config
        self.hook = hook
        self.device = torch.device(config.device)
        self.rng = np.random.RandomState(config.seed)

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

    def train(self) -> List[RoundResult]:
        """Main training loop."""
        results = []
        total_rounds = self.config.global_round

        for round_num in range(1, total_rounds + 1):
            t0 = time.time()

            ctx = self.hook.pre_round(self, round_num)
            client_results = self.run_sfl_round(ctx)
            round_result = self.hook.post_round(self, round_num, ctx, client_results)

            round_result.metrics["round_time"] = time.time() - t0
            results.append(round_result)

        return results

    def run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Execute one SFL round with interleaved client processing.

        Per-client pattern: one batch per client per iteration,
        cycling through all clients in sorted order.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device
        client_order = sorted(client_states.keys())

        # Compute max iterations across all clients
        max_iters = 0
        for cid in client_order:
            n_batches = len(client_states[cid].dataloader)
            total = self.config.local_epochs * n_batches
            max_iters = max(max_iters, total)

        total_loss = 0.0
        loss_count = 0

        for it in range(max_iters):
            for cid in client_order:
                state = client_states[cid]

                # Get batch
                images, labels = get_next_batch(state, device)

                # Client forward
                activation, labels = client_forward(state, (images, labels), device)

                # Handle BatchNorm edge case (single sample)
                if activation.size(0) == 1:
                    server_model.eval()
                else:
                    server_model.train()

                # Server forward + backward
                server_optimizer.zero_grad()
                logits = server_model(activation)
                loss = self.hook.compute_loss(logits, labels, criterion, ctx, client_id=cid)
                loss.backward()
                server_optimizer.step()

                total_loss += loss.item()
                loss_count += 1

                # Client backward
                activation_grad = activation.grad.clone().detach()
                client_backward(state, images, activation_grad, self.config)

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)

        # Collect results
        client_results = []
        for cid in client_order:
            state = client_states[cid]
            result = ClientResult(
                client_id=cid,
                model_state_dict=snapshot_model(state.client_model),
                dataset_size=state.dataset_size,
                label_distribution=state.label_distribution,
            )
            client_results.append(result)

        return client_results
