"""
SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) hook for SFL.

Control variates correct client gradient drift in Non-IID settings:
- Global control variate `c` and per-client control variates `c_i`
- Gradient correction: g_corrected = g - c_i + c
- Control variate update: c_i_new = c_i - c + (1/(K*lr)) * (x_global - x_local)
- Global update: c_new = c + (1/N) * sum(c_i_new - c_i_old)

In SFL context, SCAFFOLD is applied to the client-side model only.
Server-side model is unchanged (standard SFL per-client training).
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List

import torch

from .base import BaseMethodHook
from ..client_ops import (
    RoundContext, RoundResult, ClientResult, ClientState,
    create_client_state, create_server_optimizer, create_criterion,
    snapshot_model, create_optimizer,
)
from ..aggregation import aggregate
from ..selection import select_clients

if TYPE_CHECKING:
    from ..trainer import SimTrainer


class SCAFFOLDHook(BaseMethodHook):
    """SCAFFOLD adapted for Split Federated Learning.

    Control variates are applied to the client-side model only.
    Uses per_client server training mode (same as SFL).
    """

    def __init__(self, config, trainer: "SimTrainer"):
        super().__init__(config, trainer)
        # Pre-allocate client model pool
        self._client_pool: list = []

        # SCAFFOLD persistent state
        # Global control variate: {param_name: tensor}
        self.global_c: Dict[str, torch.Tensor] = {}
        # Per-client control variates: {client_id: {param_name: tensor}}
        self.client_c: Dict[int, Dict[str, torch.Tensor]] = {}

    @property
    def server_training_mode(self) -> str:
        return "per_client"

    def _ensure_pool(self, n: int, model, device):
        """Ensure pool has at least n client model copies."""
        while len(self._client_pool) < n:
            self._client_pool.append(copy.deepcopy(model.client_model).to(device))

    def _initialize_control_variates(self, client_model):
        """Initialize global control variate to zeros if not set."""
        if not self.global_c:
            for name, param in client_model.named_parameters():
                if param.requires_grad:
                    self.global_c[name] = torch.zeros_like(param.data, device="cpu")

    def _get_client_c(self, cid: int, client_model) -> Dict[str, torch.Tensor]:
        """Get per-client control variate, initializing to zeros if needed."""
        if cid not in self.client_c:
            self.client_c[cid] = {}
            for name, param in client_model.named_parameters():
                if param.requires_grad:
                    self.client_c[cid][name] = torch.zeros_like(param.data, device="cpu")
        return self.client_c[cid]

    def pre_round(self, trainer: "SimTrainer", round_number: int) -> RoundContext:
        config = self.config
        device = trainer.device
        model = trainer.model

        # 1. Select clients (uniform)
        selected = select_clients(
            config.selector,
            config.num_clients_per_round,
            trainer.all_client_ids,
            rng=trainer.rng,
        )

        # 2. Initialize global control variate if first round
        self._initialize_control_variates(model.client_model)

        # 3. Snapshot current client model state
        client_base_state = snapshot_model(model.client_model)

        # 4. Create client states with SCAFFOLD correction info
        self._ensure_pool(len(selected), model, device)

        client_states = {}
        for i, cid in enumerate(selected):
            client_model = self._client_pool[i]
            client_model.load_state_dict(client_base_state)

            indices = trainer.client_data_masks[cid]
            state = create_client_state(
                client_id=cid,
                client_model=client_model,
                trainset=trainer.trainset,
                data_indices=indices,
                config=config,
            )

            # Store SCAFFOLD correction in extra
            # correction[name] = global_c[name] - client_c[cid][name]
            c_i = self._get_client_c(cid, client_model)
            correction = {}
            for name in self.global_c:
                correction[name] = self.global_c[name] - c_i[name]
            state.extra["scaffold_correction"] = correction

            # Store initial params snapshot for control variate update
            state.extra["theta_0"] = {
                name: param.data.clone().detach().cpu()
                for name, param in client_model.named_parameters()
                if param.requires_grad
            }
            state.extra["local_step_count"] = 0

            client_states[cid] = state

        # 5. Server model + optimizer + criterion
        server_model = model.server_model
        server_model.to(device)
        server_optimizer = create_server_optimizer(server_model, config)
        criterion = create_criterion(config)

        # 6. Compute iterations
        max_iters = 0
        for cid in selected:
            n_batches = len(client_states[cid].dataloader)
            max_iters = max(max_iters, config.local_epochs * n_batches)

        # 7. Register callback to apply SCAFFOLD gradient correction
        # The correction is applied after loss.backward() but before optimizer.step()
        def scaffold_correction_callback(client_id, logits, loss, activation_grad, iteration, ctx):
            state = ctx.client_states.get(client_id)
            if state is None:
                return
            correction = state.extra.get("scaffold_correction", {})
            for name, param in state.client_model.named_parameters():
                if param.grad is not None and name in correction:
                    param.grad.data += correction[name].to(param.device)
            state.extra["local_step_count"] = state.extra.get("local_step_count", 0) + 1

        # Store callback reference for cleanup
        self._scaffold_callback = scaffold_correction_callback
        trainer.subscribe("after_server_backward", scaffold_correction_callback)

        return RoundContext(
            round_number=round_number,
            selected_client_ids=selected,
            client_states=client_states,
            server_model=server_model,
            server_optimizer=server_optimizer,
            criterion=criterion,
            iterations=max_iters,
            device=device,
            extra={"client_base_state": client_base_state},
        )

    def post_round(
        self,
        trainer: "SimTrainer",
        round_number: int,
        round_ctx: RoundContext,
        client_results: List[ClientResult],
    ) -> RoundResult:
        model = trainer.model
        device = trainer.device

        # Unsubscribe SCAFFOLD callback
        if hasattr(self, "_scaffold_callback"):
            trainer.unsubscribe("after_server_backward", self._scaffold_callback)

        # 1. Update per-client control variates
        lr = self.config.learning_rate
        N = self.config.num_clients  # Total clients (not just selected)

        delta_c_list = []
        for result in client_results:
            cid = result.client_id
            state = round_ctx.client_states[cid]
            theta_0 = state.extra.get("theta_0", {})
            K = state.extra.get("local_step_count", 0)

            if K == 0:
                continue

            # Current model params (after training)
            theta_K = {
                name: param.data.clone().detach().cpu()
                for name, param in state.client_model.named_parameters()
                if param.requires_grad
            }

            c_i = self._get_client_c(cid, state.client_model)
            c_i_new = {}
            delta_c = {}

            for name in self.global_c:
                if name in theta_0 and name in theta_K:
                    # c_i_new = c_i - c + (theta_0 - theta_K) / (K * lr)
                    # Note: (theta_0 - theta_K) / (K*lr) = -(theta_K - theta_0) / (K*lr)
                    delta_theta = theta_K[name] - theta_0[name]
                    c_i_new[name] = (
                        c_i[name]
                        - self.global_c[name]
                        - delta_theta / (K * lr)
                    )
                    delta_c[name] = c_i_new[name] - c_i[name]

            # Update per-client control variate
            self.client_c[cid] = c_i_new
            delta_c_list.append(delta_c)

        # 2. Update global control variate: c = c + (1/N) * sum(delta_c_i)
        if delta_c_list:
            for name in self.global_c:
                delta_sum = sum(
                    dc[name] for dc in delta_c_list if name in dc
                )
                self.global_c[name] = self.global_c[name] + delta_sum / N

        # 3. Aggregate client models (FedAvg)
        state_dicts = [r.model_state_dict for r in client_results]
        weights = [float(r.dataset_size) for r in client_results]
        agg_state = aggregate(self.config.aggregator, state_dicts, weights)

        # 4. Update global client model
        model.client_model.load_state_dict(agg_state)

        # 5. Evaluate
        accuracy = model.evaluate(trainer.testloader, device)
        avg_loss = round_ctx.extra.get("avg_loss", 0.0)

        print(
            f"[Round {round_number:3d}] "
            f"Acc: {accuracy:.4f}  Loss: {avg_loss:.4f}",
            flush=True,
        )

        return RoundResult(
            round_number=round_number,
            accuracy=accuracy,
            loss=avg_loss,
            metrics={
                "selected_clients": round_ctx.selected_client_ids,
            },
        )
