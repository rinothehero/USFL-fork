"""
SFL Hook — Basic Split Federated Learning.

Server training mode: per_client (Pattern A)
  - Server processes ONE client at a time
  - optimizer.step() called per client per iteration

This is the simplest hook. All other methods build on top of this pattern.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List

import torch
import torch.nn as nn

from .base import BaseMethodHook
from ..client_ops import (
    ClientResult,
    ClientState,
    RoundContext,
    RoundResult,
    create_client_state,
    create_criterion,
    create_server_optimizer,
    get_label_distribution,
    snapshot_model,
    restore_model,
)

if TYPE_CHECKING:
    pass


class SFLHook(BaseMethodHook):
    """Basic Split Federated Learning. Per-client sequential server training."""

    def pre_round(self, trainer, round_number: int) -> RoundContext:
        config = self.config
        model = self.resources["model"]
        splitter = self.resources["splitter"]
        selector = self.resources["selector"]
        client_data_masks = self.resources["client_data_masks"]
        trainset = self.resources["trainset"]
        device = trainer.device

        # 1. Build client_informations for selector
        all_client_ids = list(range(config.num_clients))
        client_informations = self._build_client_informations(
            all_client_ids, trainset, client_data_masks
        )

        # 2. Select clients
        selector_data = {
            "client_informations": client_informations,
            "num_classes": self.resources["num_classes"],
            "batch_size": config.batch_size,
        }
        selected = selector.select(
            config.num_clients_per_round, list(all_client_ids), selector_data
        )

        # 3. Split model
        #    FlexibleResNet: get_split_models() returns references (ERRATA 3)
        #    Standard models: splitter.split() returns ModuleDict list
        if hasattr(model, "get_split_models"):
            client_module, server_module = model.get_split_models()
        else:
            split_result = splitter.split(
                model.get_torch_model(), config.sfl_config.__dict__
            )
            client_module = split_result[0]
            server_module = split_result[1]

        client_module = client_module.to(device)
        server_module = server_module.to(device)

        # 4. Snapshot client model state (all clients start from same global state)
        client_base_state = snapshot_model(client_module)

        # 5. Create server optimizer + criterion
        server_optimizer = create_server_optimizer(server_module, config)
        criterion = create_criterion(config)

        # 6. Create ClientState for each selected client
        client_states = {}
        for cid in sorted(selected):
            # Restore client model to base state before creating state
            restore_model(client_module, client_base_state)
            indices = client_data_masks[cid]

            state = create_client_state(
                client_id=cid,
                client_model=client_module,
                trainset=trainset,
                data_indices=indices,
                config=config,
            )
            # Snapshot this client's starting state
            state.extra["initial_state"] = client_base_state
            client_states[cid] = state

        # 7. Calculate iterations
        #    SFL: local_epochs × ceil(dataset_size / batch_size) per client
        #    But in per-client mode, iteration count = local_epochs * batches_per_epoch
        #    We set iterations=None to let trainer handle epoch-based iteration
        iterations = self._compute_iterations(client_states, config)

        return RoundContext(
            round_number=round_number,
            selected_client_ids=sorted(selected),
            client_states=client_states,
            server_model=server_module,
            server_optimizer=server_optimizer,
            criterion=criterion,
            iterations=iterations,
            device=device,
            extra={
                "client_base_state": client_base_state,
                "client_module_ref": client_module,
            },
        )

    def post_round(
        self,
        trainer,
        round_number: int,
        round_ctx: RoundContext,
        client_results: List[ClientResult],
    ) -> RoundResult:
        config = self.config
        model = self.resources["model"]
        aggregator = self.resources["aggregator"]

        # 1. Build models and params for aggregator
        #    aggregator.aggregate expects: (models: List[Module], params: List[dict])
        client_module_ref = round_ctx.extra["client_module_ref"]
        models = []
        params = []

        for result in client_results:
            # Create a temporary model with this client's state
            m = copy.deepcopy(client_module_ref)
            m.load_state_dict(result.model_state_dict)
            models.append(m)

            params.append({
                "client_id": result.client_id,
                "dataset_size": result.dataset_size,
                "label_distribution": result.label_distribution,
                "round_number": round_number,
            })

        # 2. Aggregate
        if models:
            aggregated = aggregator.aggregate(models, params)

            # 3. Update global model
            if hasattr(model, "get_split_models"):
                client_model, _ = model.get_split_models()
                client_model.load_state_dict(aggregated.state_dict())
                if hasattr(model, "sync_full_model_from_split"):
                    model.sync_full_model_from_split()
            else:
                # For ModuleDict-based splits, update the full model
                torch_model = model.get_torch_model()
                # Merge client aggregated + server model
                full_state = torch_model.state_dict()
                full_state.update(aggregated.state_dict())
                full_state.update(round_ctx.server_model.state_dict())
                torch_model.load_state_dict(full_state)

        # 4. Evaluate
        accuracy = model.evaluate(self.resources["testloader"])

        avg_loss = round_ctx.extra.get("avg_loss", 0.0)
        print(
            f"[Round {round_number}] Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}"
        )

        return RoundResult(
            round_number=round_number,
            accuracy=accuracy,
            loss=avg_loss,
            metrics={
                "selected_clients": round_ctx.selected_client_ids,
            },
        )

    # --- Helpers ---

    def _build_client_informations(
        self, client_ids, trainset, client_data_masks
    ) -> dict:
        """Build client_informations dict matching selector interface."""
        infos = {}
        for cid in client_ids:
            indices = client_data_masks[cid]
            label_dist = get_label_distribution(trainset, indices)
            infos[cid] = {
                "dataset": {
                    "label_distribution": label_dist,
                    "size": len(indices),
                },
            }
        return infos

    def _compute_iterations(self, client_states: Dict[int, ClientState], config) -> int:
        """
        Compute total number of iterations for per-client mode.
        Each client does local_epochs passes through its data.
        iterations = local_epochs * max(batches_per_client)
        """
        max_batches = 0
        for state in client_states.values():
            n_batches = len(state.dataloader)
            max_batches = max(max_batches, n_batches)
        return config.local_epochs * max_batches
