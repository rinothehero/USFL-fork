from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from .utils import average_state_dicts, blend_state_dict


def _set_batchnorm_eval(module: nn.Module) -> Dict[str, bool]:
    states: Dict[str, bool] = {}
    for name, child in module.named_modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            states[name] = child.training
            child.eval()
    return states


def _restore_batchnorm(module: nn.Module, states: Dict[str, bool]) -> None:
    for name, child in module.named_modules():
        if name in states:
            child.train(states[name])


@dataclass
class BranchClientState:
    model: nn.Module
    optimizer: Any


@dataclass
class BranchServerState:
    model: nn.Module
    optimizer: Any


@dataclass
class ServerTrainResult:
    grad_f_main: Any
    grad_norm_sq: float
    server_param_update_norm: float
    grad_f_main_norm: float


class FedServer:
    def __init__(
        self,
        branch_client_states: List[BranchClientState],
        alpha: float,
        device: str = "cpu",
    ):
        self.branches = branch_client_states
        self.alpha = alpha
        self.device = device
        self.master_state_dict: Optional[dict] = None

    def get_client_branch(self, b: int) -> BranchClientState:
        return self.branches[b]

    def compute_master(self) -> dict:
        sds = [bs.model.state_dict() for bs in self.branches]
        master = average_state_dicts(sds)
        self.master_state_dict = master
        return master

    def soft_pull_to_master(self) -> None:
        if self.master_state_dict is None:
            raise RuntimeError("Master not computed")
        for bs in self.branches:
            new_sd = blend_state_dict(
                bs.model.state_dict(), self.master_state_dict, self.alpha
            )
            bs.model.load_state_dict(new_sd)


class MainServer:
    def __init__(
        self,
        branch_server_states: List[BranchServerState],
        alpha: float,
        device: str = "cpu",
        clip_grad: bool = False,
        clip_grad_max_norm: float = 10.0,
    ):
        self.branches = branch_server_states
        self.alpha = alpha
        self.device = device
        self.clip_grad = clip_grad
        self.clip_grad_max_norm = clip_grad_max_norm
        self.master_state_dict: Optional[dict] = None
        self.criterion = nn.CrossEntropyLoss()

    def get_server_branch(self, b: int) -> BranchServerState:
        return self.branches[b]

    def train_branch_with_replay(
        self,
        b: int,
        f_main: Any,
        y_main: torch.Tensor,
        f_replay_list: List[Any],
        y_replay_list: List[torch.Tensor],
    ) -> ServerTrainResult:
        bs = self.branches[b]
        ws = bs.model
        opt = bs.optimizer

        params_before = {n: p.clone() for n, p in ws.named_parameters()}

        ws.train()
        opt.zero_grad(set_to_none=True)

        f_rep = None
        y_rep = None
        if len(f_replay_list) > 0 and not isinstance(f_replay_list[0], tuple):
            f_rep = torch.cat(f_replay_list, dim=0)
            y_rep = torch.cat(y_replay_list, dim=0)

        y_main = y_main.to(self.device)

        if isinstance(f_main, tuple):
            f_main_act, f_main_id = f_main
            f_main_act = f_main_act.to(self.device)
            f_main_id = f_main_id.to(self.device)
            f_main_act_srv = f_main_act.detach().requires_grad_(True)
            f_main_id_srv = f_main_id.detach().requires_grad_(True)

            if f_rep is not None and y_rep is not None:
                if not all(isinstance(fr, tuple) for fr in f_replay_list):
                    raise ValueError(
                        "Replay features must be tuples for tuple split output"
                    )
                f_rep_act = torch.cat([fr[0] for fr in f_replay_list], dim=0)
                f_rep_id = torch.cat([fr[1] for fr in f_replay_list], dim=0)
                f_rep_act = f_rep_act.to(self.device).detach()
                f_rep_id = f_rep_id.to(self.device).detach()
                f_all = (
                    torch.cat([f_main_act_srv, f_rep_act], dim=0),
                    torch.cat([f_main_id_srv, f_rep_id], dim=0),
                )
                y_rep = y_rep.to(self.device)
                y_all = torch.cat([y_main, y_rep], dim=0)
            else:
                f_all = (f_main_act_srv, f_main_id_srv)
                y_all = y_main

            bn_states: Optional[Dict[str, bool]] = None
            if y_all.size(0) == 1:
                bn_states = _set_batchnorm_eval(ws)

            logits = ws(f_all)
            loss = self.criterion(logits, y_all)
            loss.backward()

            if bn_states is not None:
                _restore_batchnorm(ws, bn_states)

            grad_act = f_main_act_srv.grad
            grad_id = f_main_id_srv.grad
            if grad_act is None or grad_id is None:
                raise RuntimeError("grad_f_main tuple must not be None after backward")
            grad_f_main = (grad_act.detach(), grad_id.detach())
        else:
            f_main = f_main.to(self.device)
            f_main_srv = f_main.detach().requires_grad_(True)

            if f_rep is not None and y_rep is not None:
                f_rep = f_rep.to(self.device).detach()
                y_rep = y_rep.to(self.device)
                f_all = torch.cat([f_main_srv, f_rep], dim=0)
                y_all = torch.cat([y_main, y_rep], dim=0)
            else:
                f_all = f_main_srv
                y_all = y_main

            bn_states: Optional[Dict[str, bool]] = None
            if y_all.size(0) == 1:
                bn_states = _set_batchnorm_eval(ws)

            logits = ws(f_all)
            loss = self.criterion(logits, y_all)
            loss.backward()

            if bn_states is not None:
                _restore_batchnorm(ws, bn_states)

            grad_f_main = f_main_srv.grad
            assert grad_f_main is not None, (
                "grad_f_main must not be None after backward"
            )
            grad_f_main = grad_f_main.detach()

        if isinstance(grad_f_main, tuple):
            grad_f_main_norm = float(torch.norm(grad_f_main[0]).item())
        else:
            grad_f_main_norm = float(torch.norm(grad_f_main).item())

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                ws.parameters(), max_norm=self.clip_grad_max_norm
            )

        grad_norm_sq = 0.0
        for p in ws.parameters():
            if p.grad is not None:
                grad_norm_sq += float((p.grad.detach() ** 2).sum().item())

        opt.step()

        update_norm_sq = 0.0
        for n, p in ws.named_parameters():
            diff = p - params_before[n]
            update_norm_sq += float((diff**2).sum().item())
        server_param_update_norm = float(np.sqrt(update_norm_sq))

        return ServerTrainResult(
            grad_f_main=grad_f_main,
            grad_norm_sq=grad_norm_sq,
            server_param_update_norm=server_param_update_norm,
            grad_f_main_norm=grad_f_main_norm,
        )

    def compute_master(self) -> dict:
        sds = [bs.model.state_dict() for bs in self.branches]
        master = average_state_dicts(sds)
        self.master_state_dict = master
        return master

    def soft_pull_to_master(self) -> None:
        if self.master_state_dict is None:
            raise RuntimeError("Master not computed")
        for bs in self.branches:
            new_sd = blend_state_dict(
                bs.model.state_dict(), self.master_state_dict, self.alpha
            )
            bs.model.load_state_dict(new_sd)
