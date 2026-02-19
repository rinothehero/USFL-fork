from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import torch


_ROUND_FILE_RE = re.compile(r"round_(\d+)\.pt$")


def _to_round_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return None


def load_iid_mu_reference(path: str) -> Dict[int, torch.Tensor]:
    """
    Load round-indexed client consensus update vectors (mu_c^t) used as IID reference.

    Supported formats:
    1) Directory with per-round files: round_0001.pt, round_0002.pt, ...
       - file payload can be tensor directly or {"mu": tensor}
    2) Single torch file containing:
       - {"round_mu": {round: tensor, ...}} or
       - {round: tensor, ...}
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    out: Dict[int, torch.Tensor] = {}

    if p.is_dir():
        for child in sorted(p.iterdir()):
            match = _ROUND_FILE_RE.match(child.name)
            if not match:
                continue
            round_number = _to_round_int(match.group(1))
            if round_number is None:
                continue
            try:
                payload = torch.load(child, map_location="cpu")
            except Exception:
                continue
            tensor = None
            if isinstance(payload, dict):
                tensor = _to_tensor(payload.get("mu"))
            if tensor is None:
                tensor = _to_tensor(payload)
            if tensor is not None:
                out[round_number] = tensor
        return out

    # Single-file format
    try:
        payload = torch.load(p, map_location="cpu")
    except Exception:
        return {}

    source = payload
    if isinstance(payload, dict) and isinstance(payload.get("round_mu"), dict):
        source = payload["round_mu"]

    if isinstance(source, dict):
        for k, v in source.items():
            round_number = _to_round_int(k)
            if round_number is None:
                continue
            tensor = _to_tensor(v)
            if tensor is not None:
                out[round_number] = tensor

    return out


def save_iid_mu_round(save_dir: str, round_number: int, mu: torch.Tensor) -> str | None:
    """
    Save per-round IID reference mu_c^t as:
      <save_dir>/round_XXXX.pt

    Returns saved file path on success; None otherwise.
    """
    if not save_dir:
        return None
    if not isinstance(mu, torch.Tensor) or mu.numel() == 0:
        return None

    out_dir = Path(save_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    out_file = out_dir / f"round_{int(round_number):04d}.pt"
    payload = {
        "round": int(round_number),
        "mu": mu.detach().cpu().to(dtype=torch.float16),
    }
    try:
        torch.save(payload, out_file)
    except Exception:
        return None
    return str(out_file)
