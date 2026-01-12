import random
import numpy as np
import torch
from typing import Dict, List

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_dense_label_dist(label_count: Dict[int, int], num_classes: int) -> np.ndarray:
    """
    Convert sparse dict -> dense normalized distribution vector of length num_classes.
    This is critical to avoid 'requested=0 forever' bugs.
    """
    v = np.zeros(num_classes, dtype=np.float64)
    for k, cnt in label_count.items():
        v[int(k)] = float(cnt)
    s = float(v.sum())
    if s > 0:
        v /= s
    return v

def count_labels(y: torch.Tensor, num_classes: int) -> Dict[int, int]:
    """
    Return sparse count dict from a label tensor.
    """
    y_np = y.detach().cpu().numpy().astype(int)
    counts = {}
    for c in y_np:
        counts[c] = counts.get(c, 0) + 1
    return counts

def average_state_dicts(state_dicts: List[dict]) -> dict:
    """
    Elementwise average of model state dicts (float tensors).
    Assumes identical keys.
    """
    if not state_dicts:
        raise ValueError("No state_dicts to average")

    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        vals = [sd[k] for sd in state_dicts]
        if torch.is_floating_point(vals[0]):
            avg[k] = torch.stack(vals, dim=0).mean(dim=0)
        else:
            # non-float buffers (e.g., num_batches_tracked) - take first
            avg[k] = vals[0]
    return avg

def blend_state_dict(base: dict, master: dict, alpha: float) -> dict:
    """
    base <- (base + alpha * master)/(1+alpha)
    """
    out = {}
    for k, v in base.items():
        if torch.is_floating_point(v):
            out[k] = (v + alpha * master[k]) / (1.0 + alpha)
        else:
            out[k] = v
    return out
