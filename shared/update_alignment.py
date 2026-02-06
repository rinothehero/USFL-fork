"""
Update Alignment Metric (A_cos + M_norm) for Fair Cross-Technique Comparison

Measures pairwise cosine similarity of client update vectors to quantify
how aligned participating clients are in their optimization direction.

Key advantage over G_drift: Scale-invariant — immune to LR/batch_size/step
definition differences across SFL, USFL, GAS, and MultiSFL.

Metrics:
  A_cos:  Weighted mean pairwise cosine similarity of update vectors (higher = more aligned)
  M_norm: Weighted mean L2 norm of update vectors (update magnitude)

Usage:
  deltas = [flatten_delta(end_sd, start_params) for each client]
  result = compute_update_alignment(deltas, weights=agg_weights)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


# BatchNorm buffer names to exclude from trainable params
_BN_BUFFER_KEYWORDS = ("running_mean", "running_var", "num_batches_tracked")


def is_trainable_param(name: str) -> bool:
    """Check if parameter name is trainable (not a BN buffer)."""
    return not any(kw in name for kw in _BN_BUFFER_KEYWORDS)


def flatten_delta(
    end_state_dict: dict,
    start_params: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Compute Δ_i = flatten(θ_end - θ_start) for trainable parameters only.

    Args:
        end_state_dict: Client model state_dict after local training
        start_params: Round-start global model params {name: tensor} (CPU)

    Returns:
        Flattened 1-D tensor of parameter differences, or None if no matching params.
    """
    diffs = []
    for name, start_val in start_params.items():
        if name in end_state_dict and is_trainable_param(name):
            end_val = end_state_dict[name]
            if isinstance(end_val, torch.Tensor):
                diff = end_val.detach().cpu().float() - start_val.float()
            else:
                diff = torch.tensor(end_val, dtype=torch.float32) - start_val.float()
            diffs.append(diff.reshape(-1))
    if not diffs:
        return None
    return torch.cat(diffs)


@dataclass
class AlignmentResult:
    """Result of update alignment computation."""
    A_cos: float = 0.0          # Weighted mean pairwise cosine similarity
    M_norm: float = 0.0         # Weighted mean update norm
    n_valid: int = 0            # Number of clients with ||Δ|| > τ
    n_total: int = 0            # Total clients
    n_pairs: int = 0            # Number of valid pairs computed
    per_pair: List[dict] = field(default_factory=list)  # Optional per-pair details

    def to_dict(self) -> dict:
        return {
            "A_cos": self.A_cos,
            "M_norm": self.M_norm,
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "n_pairs": self.n_pairs,
        }


def compute_update_alignment(
    deltas: List[Tuple[int, torch.Tensor]],
    weights: Optional[Dict[int, float]] = None,
    tau: float = 1e-7,
    record_pairs: bool = False,
) -> AlignmentResult:
    """
    Compute A_cos (pairwise cosine alignment) and M_norm (mean update magnitude).

    Args:
        deltas: List of (client_id, flattened_delta_vector) tuples.
        weights: Optional {client_id: weight} for weighted averaging.
                 If None, uniform weights are used.
        tau: Minimum ||Δ|| threshold. Clients below this are excluded from A_cos.
        record_pairs: If True, store per-pair cosine values (for debugging).

    Returns:
        AlignmentResult with A_cos, M_norm, and metadata.
    """
    result = AlignmentResult(n_total=len(deltas))

    if len(deltas) == 0:
        return result

    # Compute norms and filter by τ
    valid_entries: List[Tuple[int, torch.Tensor, float]] = []  # (id, delta, norm)
    all_norms: List[Tuple[int, float]] = []

    for cid, delta in deltas:
        norm = delta.norm().item()
        all_norms.append((cid, norm))
        if norm > tau:
            valid_entries.append((cid, delta, norm))

    result.n_valid = len(valid_entries)

    # --- M_norm: weighted mean of ALL client norms (including small ones) ---
    if weights is not None:
        w_sum = sum(weights.get(cid, 1.0) for cid, _ in all_norms)
        if w_sum > 0:
            result.M_norm = sum(
                weights.get(cid, 1.0) * norm for cid, norm in all_norms
            ) / w_sum
    else:
        if all_norms:
            result.M_norm = sum(norm for _, norm in all_norms) / len(all_norms)

    # --- A_cos: weighted mean pairwise cosine (valid clients only) ---
    if len(valid_entries) < 2:
        # Not enough valid clients for pairwise comparison
        result.A_cos = float("nan")
        return result

    cos_sum = 0.0
    weight_sum = 0.0
    n_pairs = 0

    for i in range(len(valid_entries)):
        cid_i, delta_i, norm_i = valid_entries[i]
        for j in range(i + 1, len(valid_entries)):
            cid_j, delta_j, norm_j = valid_entries[j]

            # Cosine similarity
            cos_val = F.cosine_similarity(
                delta_i.unsqueeze(0), delta_j.unsqueeze(0)
            ).item()

            # Pair weight: w_i * w_j (or 1.0 if uniform)
            if weights is not None:
                w_ij = weights.get(cid_i, 1.0) * weights.get(cid_j, 1.0)
            else:
                w_ij = 1.0

            cos_sum += w_ij * cos_val
            weight_sum += w_ij
            n_pairs += 1

            if record_pairs:
                result.per_pair.append({
                    "i": cid_i, "j": cid_j,
                    "cos": cos_val, "w": w_ij,
                })

    result.n_pairs = n_pairs
    result.A_cos = cos_sum / weight_sum if weight_sum > 0 else float("nan")

    return result
