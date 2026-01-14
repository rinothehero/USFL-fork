from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from .config import MultiSFLConfig
from .client import Client, ClientUpdateStats
from .servers import (
    FedServer,
    MainServer,
    BranchClientState,
    BranchServerState,
    ServerTrainResult,
)
from .replay import ScoreVectorTracker, KnowledgeRequestPlanner
from .scheduler import SamplingProportionScheduler
from .models import SplitModel, get_full_model, get_split_models, ClientNet, ServerNet
from .utils import set_seed
from .g_measurement import GMeasurementSystem
from torch.utils.data import DataLoader, ConcatDataset


@dataclass
class RoundStats:
    round_idx: int
    p_r: float
    fgn_r: float
    requested: int
    collected: int
    trials: int
    acc: float
    mean_grad_f_main_norm: float
    mean_client_update_norm: float
    mean_server_update_norm: float


class MultiSFLTrainer:
    def __init__(
        self,
        cfg: MultiSFLConfig,
        clients: List[Client],
        fed: FedServer,
        main: MainServer,
        score_tracker: ScoreVectorTracker,
        planner: KnowledgeRequestPlanner,
        scheduler: SamplingProportionScheduler,
        test_loader: Any = None,
    ):
        self.cfg = cfg
        self.clients = clients
        self.fed = fed
        self.main = main
        self.score_tracker = score_tracker
        self.planner = planner
        self.scheduler = scheduler
        self.test_loader = test_loader

        self.g_system: Optional[GMeasurementSystem] = None
        if cfg.enable_g_measurement:
            # Combine all client datasets for Oracle calculation
            all_datasets = [c.dataset for c in clients]
            full_dataset = ConcatDataset(all_datasets)
            # Use the same batch size as training for fair Oracle averaging.
            full_loader = DataLoader(
                full_dataset,
                batch_size=cfg.batch_size * cfg.n_main_clients_per_round,
                shuffle=False,
                drop_last=False,
                num_workers=2,
            )
            self.g_system = GMeasurementSystem(
                full_loader,
                device=cfg.device,
                diagnostic_frequency=cfg.g_measure_frequency,
                use_variance_g=cfg.use_variance_g,
            )

        self.B = cfg.num_branches or cfg.n_main_clients_per_round
        assert self.B == len(self.fed.branches) == len(self.main.branches)

        self.stats: List[RoundStats] = []

    def sample_main_clients(self) -> List[int]:
        ids = np.random.choice(
            len(self.clients), size=self.cfg.n_main_clients_per_round, replace=False
        ).tolist()
        return ids

    def evaluate_master(self) -> float:
        if self.test_loader is None:
            return float("nan")

        wc_master_sd = self.fed.master_state_dict
        ws_master_sd = self.main.master_state_dict
        if wc_master_sd is None or ws_master_sd is None:
            wc_master_sd = self.fed.compute_master()
            ws_master_sd = self.main.compute_master()

        model = get_full_model(
            self.cfg.model_type, self.cfg.dataset, self.cfg.num_classes
        ).to(self.cfg.device)

        if self.cfg.model_type == "simple":
            from .models import ClientNet, ServerNet

            wc = ClientNet().to(self.cfg.device)
            ws = ServerNet(num_classes=self.cfg.num_classes).to(self.cfg.device)
            wc.load_state_dict(wc_master_sd)
            ws.load_state_dict(ws_master_sd)
            model = SplitModel(wc, ws).to(self.cfg.device)
        else:
            # Merge client and server state dicts into the full model
            full_sd = model.state_dict()

            for k, v in wc_master_sd.items():
                if k in full_sd:
                    full_sd[k] = v

            for k, v in ws_master_sd.items():
                if k in full_sd:
                    full_sd[k] = v

            model.load_state_dict(full_sd)

        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.cfg.device)
                y = y.to(self.cfg.device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        return correct / max(total, 1)

    def run(self) -> List[RoundStats]:
        set_seed(self.cfg.seed)

        # Ensure master model exists for first round G measurement
        self.fed.compute_master()
        self.main.compute_master()

        p_r = self.scheduler.state.p

        for r in range(self.cfg.num_rounds):
            main_ids = self.sample_main_clients()
            inactive_ids = [i for i in range(len(self.clients)) if i not in main_ids]

            perm = np.random.permutation(main_ids)
            mapping = {b: int(perm[b % len(perm)]) for b in range(self.B)}
            print(
                f"\n[Round {r + 1}/{self.cfg.num_rounds}] Branch-Client mapping: {mapping}"
            )

            # G Measurement: Setup PRE-ROUND model and compute Oracle
            is_diagnostic = (
                self.g_system is not None and self.g_system.is_diagnostic_round(r)
            )
            wc_measure = None
            ws_measure = None

            if is_diagnostic:
                wc_master_sd = self.fed.master_state_dict
                ws_master_sd = self.main.master_state_dict

                if self.cfg.model_type == "simple":
                    wc_measure = ClientNet().to(self.cfg.device)
                    ws_measure = ServerNet(num_classes=self.cfg.num_classes).to(
                        self.cfg.device
                    )
                else:
                    wc_measure, ws_measure = get_split_models(
                        self.cfg.model_type,
                        self.cfg.dataset,
                        self.cfg.num_classes,
                        self.cfg.split_layer,
                    )
                    wc_measure = wc_measure.to(self.cfg.device)
                    ws_measure = ws_measure.to(self.cfg.device)

                if wc_master_sd is not None and ws_master_sd is not None:
                    wc_measure.load_state_dict(wc_master_sd)
                    ws_measure.load_state_dict(ws_master_sd)

                full_model = None
                if wc_master_sd is not None and ws_master_sd is not None:
                    full_model = get_full_model(
                        self.cfg.model_type, self.cfg.dataset, self.cfg.num_classes
                    ).to(self.cfg.device)
                    full_sd = full_model.state_dict()
                    for k, v in wc_master_sd.items():
                        if k in full_sd:
                            full_sd[k] = v
                    for k, v in ws_master_sd.items():
                        if k in full_sd:
                            full_sd[k] = v
                    full_model.load_state_dict(full_sd)

                self.g_system.compute_oracle(
                    wc_measure,
                    ws_measure,
                    full_model=full_model,
                    split_layer_name=self.cfg.split_layer,
                    use_sfl_oracle=True,
                )

            grad_norm_sq_list: List[float] = []
            grad_f_main_norm_list: List[float] = []
            client_update_norm_list: List[float] = []
            server_update_norm_list: List[float] = []
            requested_total = 0
            collected_total = 0
            trials_total = 0

            g_f_all_list: List[torch.Tensor] = []
            g_y_server_list: List[torch.Tensor] = []
            g_server_weights: List[int] = []
            g_x_client_list: List[torch.Tensor] = []
            g_y_client_list: List[torch.Tensor] = []
            g_client_ids: List[int] = []
            g_client_weights: List[int] = []

            for b in range(self.B):
                client_id = mapping[b]
                main_client = self.clients[client_id]

                bc = self.fed.get_client_branch(b)
                bs = self.main.get_server_branch(b)

                branch_grad_norm_sq = 0.0
                branch_grad_f_main_norm = 0.0
                branch_server_update_norm = 0.0
                branch_client_update_norm = 0.0
                branch_requested = 0
                branch_collected = 0
                branch_trials = 0
                q_remaining: Optional[np.ndarray] = None

                for local_step in range(self.cfg.local_steps):
                    f_main, y_main, label_dist, cache, base_count_batch = (
                        main_client.forward_main(bc.model, self.cfg.batch_size)
                    )

                    if local_step == 0:
                        self.score_tracker.append_label_dist(b, label_dist)
                        sv = self.score_tracker.score_vector(b)

                        if self.cfg.replay_budget_mode == "batch":
                            base_count = base_count_batch
                        else:
                            base_count = len(main_client.dataset)

                        req = self.planner.plan(
                            sv,
                            p_r,
                            base_count,
                            replay_min_total=self.cfg.replay_min_total,
                        )
                        q_remaining = req.q.copy()
                        branch_requested = int(req.total)

                    f_rep_list: List[torch.Tensor] = []
                    y_rep_list: List[torch.Tensor] = []

                    if local_step == 0 and q_remaining is not None:
                        q_rem: np.ndarray = q_remaining
                        trials = 0
                        while (
                            q_rem.sum() > 0
                            and trials < self.cfg.max_assistant_trials_per_branch
                            and len(inactive_ids) > 0
                        ):
                            assist_id = int(np.random.choice(inactive_ids))
                            assistant = self.clients[assist_id]

                            sampled = assistant.sample_batch_by_quota(q_rem)
                            trials += 1
                            if sampled is None:
                                continue
                            x_a, y_a, provided = sampled

                            f_a = assistant.forward_assistant(bc.model, x_a)
                            f_rep_list.append(f_a)
                            y_rep_list.append(y_a)

                            provided = np.asarray(provided, dtype=np.int64)
                            if provided.shape != q_rem.shape:
                                tmp = np.zeros_like(q_rem)
                                m = min(len(tmp), len(provided))
                                tmp[:m] = provided[:m]
                                provided = tmp
                            q_rem = np.maximum(0, q_rem - provided)
                            branch_collected += int(provided.sum())
                        branch_trials = trials

                    # G Measurement: Collect features for Server G (first step only)
                    if is_diagnostic and local_step == 0:
                        f_main_cpu = f_main.detach().cpu()
                        y_main_cpu = y_main.detach().cpu()

                        # Server G data (per branch)
                        if f_rep_list:
                            f_rep = torch.cat(f_rep_list, dim=0).detach().cpu()
                            y_rep = torch.cat(y_rep_list, dim=0).detach().cpu()
                            g_f_all_list.append(torch.cat([f_main_cpu, f_rep], dim=0))
                            g_y_server_list.append(
                                torch.cat([y_main_cpu, y_rep], dim=0)
                            )
                        else:
                            g_f_all_list.append(f_main_cpu)
                            g_y_server_list.append(y_main_cpu)

                        g_server_weights.append(int(g_y_server_list[-1].size(0)))

                        # Client G data (per client)
                        if main_client.last_batch_x is not None:
                            g_x_client_list.append(main_client.last_batch_x)
                            g_y_client_list.append(y_main_cpu)
                            g_client_ids.append(client_id)
                            g_client_weights.append(int(y_main_cpu.size(0)))

                    train_result: ServerTrainResult = (
                        self.main.train_branch_with_replay(
                            b=b,
                            f_main=f_main,
                            y_main=y_main,
                            f_replay_list=f_rep_list,
                            y_replay_list=y_rep_list,
                        )
                    )

                    client_stats: ClientUpdateStats = main_client.apply_feature_grad(
                        bc.model, bc.optimizer, cache, train_result.grad_f_main
                    )

                    branch_grad_norm_sq += train_result.grad_norm_sq
                    branch_grad_f_main_norm += train_result.grad_f_main_norm
                    branch_server_update_norm += train_result.server_param_update_norm
                    branch_client_update_norm += client_stats.param_update_norm

                grad_norm_sq_list.append(branch_grad_norm_sq / self.cfg.local_steps)
                grad_f_main_norm_list.append(
                    branch_grad_f_main_norm / self.cfg.local_steps
                )
                server_update_norm_list.append(
                    branch_server_update_norm / self.cfg.local_steps
                )
                client_update_norm_list.append(
                    branch_client_update_norm / self.cfg.local_steps
                )
                requested_total += branch_requested
                collected_total += branch_collected
                trials_total += branch_trials

            lr_for_fgn = float(self.cfg.lr_server)
            fgn_r = (
                float(np.mean([-lr_for_fgn * g2 for g2 in grad_norm_sq_list]))
                if grad_norm_sq_list
                else 0.0
            )

            p_prev = p_r
            p_r = self.scheduler.update(fgn_r)

            print(f"[Round {r + 1}] p_r: {p_prev:.6f} -> {p_r:.6f} (FGN={fgn_r:.6f})")
            print(
                f"[Round {r + 1}] Replay: requested={requested_total}, collected={collected_total}, trials={trials_total}"
            )

            mean_grad_f = (
                float(np.mean(grad_f_main_norm_list)) if grad_f_main_norm_list else 0.0
            )
            mean_client_upd = (
                float(np.mean(client_update_norm_list))
                if client_update_norm_list
                else 0.0
            )
            mean_server_upd = (
                float(np.mean(server_update_norm_list))
                if server_update_norm_list
                else 0.0
            )
            print(
                f"[Round {r + 1}] Diagnostics: ||grad_f_main||={mean_grad_f:.6f}, ||client_upd||={mean_client_upd:.6f}, ||server_upd||={mean_server_upd:.6f}"
            )

            self.fed.compute_master()
            self.main.compute_master()
            self.fed.soft_pull_to_master()
            self.main.soft_pull_to_master()

            # G Measurement: Perform measurement using PRE-ROUND models and COLLECTED data
            if (
                is_diagnostic
                and self.g_system is not None
                and wc_measure is not None
                and ws_measure is not None
            ):
                if not g_f_all_list:
                    print(
                        f"[G Measurement] No data collected for round {r + 1}, skipping."
                    )
                else:
                    self.g_system.measure_round(
                        round_idx=r,
                        client_model=wc_measure,
                        server_model=ws_measure,
                        client_ids=g_client_ids,
                        x_all=g_x_client_list,
                        y_all_client=g_y_client_list,
                        f_all=g_f_all_list,
                        y_all_server=g_y_server_list,
                        client_weights=g_client_weights,
                        server_weights=g_server_weights,
                    )

            acc = self.evaluate_master()
            print(f"[Round {r + 1}] Accuracy: {acc:.4f}")

            self.stats.append(
                RoundStats(
                    round_idx=r + 1,
                    p_r=p_r,
                    fgn_r=fgn_r,
                    requested=requested_total,
                    collected=collected_total,
                    trials=trials_total,
                    acc=acc,
                    mean_grad_f_main_norm=mean_grad_f,
                    mean_client_update_norm=mean_client_upd,
                    mean_server_update_norm=mean_server_upd,
                )
            )

        return self.stats
