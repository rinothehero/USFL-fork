from dataclasses import dataclass
from typing import List, Literal
import numpy as np

from .log_utils import vprint


@dataclass
class SamplingState:
    p: float
    fgn_hist: List[float]


class SamplingProportionScheduler:
    def __init__(
        self,
        p0: float,
        p_min: float,
        p_max: float,
        eps: float,
        mode: Literal["paper", "abs_ratio", "one_plus_delta"] = "abs_ratio",
        delta_clip: float = 0.2,
        verbose: bool = True,
    ):
        self.state = SamplingState(p=float(p0), fgn_hist=[])
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.eps = float(eps)
        self.mode = mode
        self.delta_clip = float(delta_clip)
        self.verbose = verbose

        if self.p_min <= 0.0:
            vprint(
                f"[WARNING] p_min={self.p_min} <= 0. Multiplicative updates may kill p permanently. Setting p_min=1e-4.", 0
            )
            self.p_min = 1e-4

    def update(self, fgn_r: float) -> float:
        st = self.state
        fgn_prev = st.fgn_hist[-1] if len(st.fgn_hist) > 0 else None
        p_prev = st.p

        st.fgn_hist.append(float(fgn_r))

        if fgn_prev is None:
            p_new = p_prev
            factor_or_delta = 0.0
        elif self.mode == "paper":
            denom = (
                fgn_prev
                if abs(fgn_prev) > self.eps
                else (self.eps if fgn_prev >= 0 else -self.eps)
            )
            # Paper Eq: p_{r+1} = p_r * (1 + (FGN_r - FGN_{r-1}) / FGN_{r-1})
            factor = (fgn_r - fgn_prev) / denom
            p_new = p_prev * (1.0 + factor)
            factor_or_delta = factor
        elif self.mode == "abs_ratio":
            factor = abs(fgn_r) / (abs(fgn_prev) + self.eps)
            p_new = p_prev * factor
            factor_or_delta = factor
        elif self.mode == "one_plus_delta":
            delta = (fgn_r - fgn_prev) / (abs(fgn_prev) + self.eps)
            delta = float(np.clip(delta, -self.delta_clip, self.delta_clip))
            p_new = p_prev * (1.0 + delta)
            factor_or_delta = delta
        else:
            raise ValueError(f"Unknown p update mode: {self.mode}")

        p_unclipped = p_new
        p_clipped = float(np.clip(p_new, self.p_min, self.p_max))
        st.p = p_clipped

        if self.verbose:
            vprint(
                f"[p_update] mode={self.mode}, p_prev={p_prev:.8f}, factor/delta={factor_or_delta:.6f}, "
                f"p_unclipped={p_unclipped:.8f}, p_clipped={p_clipped:.8f}, FGN_prev={fgn_prev}, FGN_r={fgn_r:.8f}", 2
            )

        return st.p
