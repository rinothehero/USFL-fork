from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import count_labels, to_dense_label_dist
from .data import build_class_to_indices
from .log_utils import vprint


SplitOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ForwardCache:
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class ClientUpdateStats:
    param_update_norm: float


class Client:
    def __init__(
        self,
        client_id: int,
        dataset: Any,
        num_classes: int,
        class_to_indices: Optional[Dict[int, List[int]]] = None,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.num_classes = num_classes
        self.device = device

        self._dl: Optional[DataLoader] = None
        self._dl_iter: Any = None

        if class_to_indices is not None:
            self.class_to_indices = class_to_indices
        elif hasattr(dataset, "targets"):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
            self.class_to_indices = build_class_to_indices(targets, num_classes)
        else:
            self.class_to_indices = {c: [] for c in range(num_classes)}

        self.last_batch_x: Optional[torch.Tensor] = None  # For G measurement

    def _get_iter(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._dl is None or self._dl.batch_size != batch_size:
            drop_last = len(self.dataset) >= batch_size
            self._dl = DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
            )
            self._dl_iter = iter(self._dl)

        for _ in range(2):
            try:
                batch = next(self._dl_iter)
                return batch
            except StopIteration:
                self._dl_iter = iter(self._dl)

        batch = next(self._dl_iter)
        return batch

    def forward_main(
        self, w_c: torch.nn.Module, batch_size: int
    ) -> Tuple[SplitOutput, torch.Tensor, np.ndarray, ForwardCache, int]:
        w_c.eval()
        x, y = self._get_iter(batch_size)
        x = x.to(self.device)

        # Save for G measurement
        self.last_batch_x = x.detach().cpu()

        y = y.to(self.device)

        with torch.no_grad():
            f = w_c(x)

        if isinstance(f, tuple):
            f = tuple(t.detach() for t in f)
        else:
            f = f.detach()

        sparse_counts = count_labels(y, self.num_classes)
        dense = to_dense_label_dist(
            {int(k): int(v) for k, v in sparse_counts.items()}, self.num_classes
        )

        assert dense.shape == (self.num_classes,), (
            f"Label dist shape mismatch: {dense.shape} != ({self.num_classes},)"
        )
        assert dense.min() >= 0, f"Label dist has negative values: {dense.min()}"

        cache = ForwardCache(x=x, y=y)
        base_count = len(y)
        return f, y.detach(), dense, cache, base_count

    def apply_feature_grad(
        self,
        w_c: torch.nn.Module,
        opt_c: Any,
        cache: ForwardCache,
        grad_f: SplitOutput,
        clip_grad: bool = False,
        clip_grad_max_norm: float = 10.0,
    ) -> ClientUpdateStats:
        params_before = {n: p.clone() for n, p in w_c.named_parameters()}

        w_c.train()
        opt_c.zero_grad(set_to_none=True)

        x = cache.x
        f = w_c(x)
        if isinstance(f, tuple):
            if not isinstance(grad_f, tuple):
                raise ValueError("grad_f must be a tuple for tuple split output")
            torch.autograd.backward(list(f), list(grad_f))
        else:
            if isinstance(grad_f, tuple):
                raise ValueError("grad_f must be a tensor for tensor split output")
            f.backward(grad_f)

        if clip_grad:
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                w_c.parameters(), max_norm=clip_grad_max_norm
            )
            vprint(f"[Clip][Client] pre_norm={float(pre_clip_norm):.6f}", 2)

        opt_c.step()

        update_norm_sq = 0.0
        for n, p in w_c.named_parameters():
            diff = p - params_before[n]
            update_norm_sq += float((diff**2).sum().item())

        return ClientUpdateStats(param_update_norm=float(np.sqrt(update_norm_sq)))

    def sample_batch_by_quota(
        self, q_remaining: np.ndarray, batch_cap: int = 256
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
        if q_remaining.sum() <= 0:
            return None

        x_list: List[torch.Tensor] = []
        y_list: List[int] = []
        provided = np.zeros(self.num_classes, dtype=np.int64)

        total_requested = int(q_remaining.sum())

        for c in range(self.num_classes):
            needed = int(q_remaining[c])
            if needed <= 0:
                continue

            available_indices = self.class_to_indices.get(c, [])
            if len(available_indices) == 0:
                continue

            n_to_sample = min(needed, len(available_indices))
            sampled_local_indices = np.random.choice(
                available_indices, size=n_to_sample, replace=False
            ).tolist()

            for local_idx in sampled_local_indices:
                sample_x, sample_y = self.dataset[local_idx]
                if not isinstance(sample_x, torch.Tensor):
                    sample_x = torch.tensor(sample_x)
                x_list.append(sample_x)
                y_list.append(
                    int(sample_y) if not isinstance(sample_y, int) else sample_y
                )
                provided[c] += 1

                if len(x_list) >= batch_cap:
                    break

            if len(x_list) >= batch_cap:
                break

        if len(x_list) == 0:
            return None

        x = torch.stack(x_list, dim=0).to(self.device)
        y = torch.tensor(y_list, dtype=torch.long, device=self.device)

        return x, y, provided

    def forward_assistant(self, w_c: torch.nn.Module, x: torch.Tensor) -> SplitOutput:
        w_c.eval()
        with torch.no_grad():
            f = w_c(x)
        if isinstance(f, tuple):
            return tuple(t.detach() for t in f)
        return f.detach()

    def get_local_class_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for c in range(self.num_classes):
            counts[c] = len(self.class_to_indices.get(c, []))
        return counts
