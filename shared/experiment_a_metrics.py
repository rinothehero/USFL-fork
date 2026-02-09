from __future__ import annotations

from typing import Dict, Optional

import torch


def _safe_float(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_cosine_distance(
    a: Optional[torch.Tensor], b: Optional[torch.Tensor], epsilon: float
) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.numel() == 0 or b.numel() == 0:
        return None
    if a.shape != b.shape:
        return None
    denom = float(torch.norm(a).item() * torch.norm(b).item()) + float(epsilon)
    if denom <= 0.0:
        return None
    numer = float(torch.dot(a, b).item())
    cos = numer / denom
    return float(1.0 - cos)


def _normalize_weights(
    deltas: Dict[int, torch.Tensor], weights: Optional[Dict[int, float]]
) -> Dict[int, float]:
    if not deltas:
        return {}

    if not weights:
        uniform = 1.0 / float(len(deltas))
        return {cid: uniform for cid in deltas.keys()}

    out: Dict[int, float] = {}
    total = 0.0
    for cid in deltas.keys():
        w = weights.get(cid, 0.0)
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 0.0
        if w < 0.0:
            w = 0.0
        out[cid] = w
        total += w

    if total <= 0.0:
        uniform = 1.0 / float(len(deltas))
        return {cid: uniform for cid in deltas.keys()}

    return {cid: (w / total) for cid, w in out.items()}


def compute_experiment_a_metrics(
    client_deltas: Dict[int, torch.Tensor],
    client_weights: Optional[Dict[int, float]] = None,
    client_probe_direction: Optional[torch.Tensor] = None,
    per_client_probe_directions: Optional[Dict[int, torch.Tensor]] = None,
    server_delta: Optional[torch.Tensor] = None,
    server_probe_direction: Optional[torch.Tensor] = None,
    server_steps: int = 0,
    epsilon: float = 1e-8,
) -> dict:
    """
    Compute Experiment A metrics from already-captured update vectors.

    Definitions follow docs/experiment.md:
    - m2, u2, Var, A_ratio, A_rel
    - B_c, C_c (anchor probe), C_c_per_client_probe (optional)
    - B_s, server magnitude per step
    """
    if not client_deltas:
        return {
            "num_clients": 0,
            "weight_sum": 0.0,
            "m2_c": None,
            "u2_c": None,
            "var_c": None,
            "A_c_ratio": None,
            "A_c_rel": None,
            "B_c": None,
            "C_c": None,
            "C_c_per_client_probe": None,
            "server_steps": int(server_steps),
            "delta_server_norm_sq": None,
            "B_s": None,
            "server_mag_per_step_sq": None,
            "server_mag_per_step": None,
        }

    normalized_weights = _normalize_weights(client_deltas, client_weights)
    weight_sum = float(sum(normalized_weights.values()))

    # Weighted mean update mu_c = Σ w_i * Δ_i
    first_delta = next(iter(client_deltas.values()))
    mu = torch.zeros_like(first_delta, dtype=torch.float32)
    m2 = 0.0
    for cid, delta in client_deltas.items():
        d = delta.float()
        w = float(normalized_weights.get(cid, 0.0))
        mu += w * d
        m2 += w * float(torch.dot(d, d).item())

    u2 = float(torch.dot(mu, mu).item())
    var_c = max(0.0, m2 - u2)
    a_ratio = var_c / (m2 + float(epsilon))
    a_rel = var_c / (u2 + float(epsilon))

    b_c = _safe_cosine_distance(mu, client_probe_direction, float(epsilon))

    c_c = None
    if client_probe_direction is not None:
        c_sum = 0.0
        for cid, delta in client_deltas.items():
            w = float(normalized_weights.get(cid, 0.0))
            dist = _safe_cosine_distance(delta.float(), client_probe_direction, float(epsilon))
            if dist is None:
                continue
            c_sum += w * dist
        c_c = float(c_sum)

    c_c_per_client = None
    if per_client_probe_directions:
        c_sum = 0.0
        used = 0
        for cid, delta in client_deltas.items():
            probe = per_client_probe_directions.get(cid)
            if probe is None:
                continue
            w = float(normalized_weights.get(cid, 0.0))
            dist = _safe_cosine_distance(delta.float(), probe.float(), float(epsilon))
            if dist is None:
                continue
            c_sum += w * dist
            used += 1
        if used > 0:
            c_c_per_client = float(c_sum)

    delta_server_norm_sq = None
    b_s = None
    server_mag_per_step_sq = None
    server_mag_per_step = None
    if server_delta is not None:
        server_delta = server_delta.float()
        delta_server_norm_sq = float(torch.dot(server_delta, server_delta).item())
        b_s = _safe_cosine_distance(server_delta, server_probe_direction, float(epsilon))
        denom_steps = float(max(int(server_steps), 0)) + float(epsilon)
        server_mag_per_step_sq = delta_server_norm_sq / denom_steps
        server_mag_per_step = float(torch.norm(server_delta).item()) / denom_steps

    return {
        "num_clients": int(len(client_deltas)),
        "weight_sum": float(weight_sum),
        "m2_c": _safe_float(m2),
        "u2_c": _safe_float(u2),
        "var_c": _safe_float(var_c),
        "A_c_ratio": _safe_float(a_ratio),
        "A_c_rel": _safe_float(a_rel),
        "B_c": _safe_float(b_c),
        "C_c": _safe_float(c_c),
        "C_c_per_client_probe": _safe_float(c_c_per_client),
        "server_steps": int(server_steps),
        "delta_server_norm_sq": _safe_float(delta_server_norm_sq),
        "B_s": _safe_float(b_s),
        "server_mag_per_step_sq": _safe_float(server_mag_per_step_sq),
        "server_mag_per_step": _safe_float(server_mag_per_step),
    }
