from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class ReplayRequest:
    q: np.ndarray          # length C, int quota per class
    total: int             # total replay requested
    prior: np.ndarray      # length C

class ScoreVectorTracker:
    def __init__(self, num_branches: int, num_classes: int, gamma: float):
        self.B = num_branches
        self.C = num_classes
        self.gamma = gamma
        self.L_hist: List[List[np.ndarray]] = [[] for _ in range(self.B)]

    def append_label_dist(self, b: int, label_dist_dense: np.ndarray) -> None:
        assert label_dist_dense.shape == (self.C,)
        # distribution should sum to 1 or be all zeros (rare)
        self.L_hist[b].append(label_dist_dense.astype(np.float64))

    def score_vector(self, b: int) -> np.ndarray:
        hist = self.L_hist[b]
        if len(hist) == 0:
            return np.ones(self.C, dtype=np.float64) / self.C

        r = len(hist) - 1
        weights = np.array([self.gamma ** (r - j) for j in range(r + 1)], dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)
        stacked = np.stack(hist, axis=0)  # (r+1, C)
        sv = (weights[:, None] * stacked).sum(axis=0)
        return sv

class KnowledgeRequestPlanner:
    def __init__(self, num_classes: int):
        self.C = num_classes

    def plan(self, sv: np.ndarray, p_r: float, base_count: int, replay_min_total: int = 0) -> ReplayRequest:
        """
        Implements prior and q computation with rounding correction.
        """
        assert sv.shape == (self.C,)
        assert base_count >= 0

        mean_sv = float(sv.mean())
        prior = np.maximum(0.0, mean_sv - sv)  # length C

        total = int(round(float(base_count) * float(p_r)))
        if total < replay_min_total and p_r > 0:
            total = replay_min_total

        if total <= 0:
            return ReplayRequest(q=np.zeros(self.C, dtype=np.int64), total=0, prior=prior)

        sum_prior = float(prior.sum())
        if sum_prior <= 0.0:
            return ReplayRequest(q=np.zeros(self.C, dtype=np.int64), total=0, prior=prior)

        q_float = total * (prior / sum_prior)
        q = np.round(q_float).astype(np.int64)

        # rounding correction
        diff = int(total - q.sum())
        if diff != 0:
            # adjust by largest prior
            order = np.argsort(-prior)
            idx = 0
            while diff != 0 and idx < len(order):
                c = int(order[idx])
                if diff > 0:
                    q[c] += 1
                    diff -= 1
                else:
                    if q[c] > 0:
                        q[c] -= 1
                        diff += 1
                idx = (idx + 1) if (idx + 1) < len(order) else 0

        # final safety
        q = np.maximum(q, 0)
        # optional: force sum==total (may differ if all q are 0 due to correction limits)
        return ReplayRequest(q=q, total=int(q.sum()), prior=prior)
