from dataclasses import dataclass

import torch
import torch.nn as nn

from .base_analyzer import BaseAnalyzer


@dataclass
class L2AnalyzerDTO:
    new_model: nn.Module
    prev_model: nn.Module


class L2Analyzer(BaseAnalyzer):
    @torch.no_grad()
    def analyze(self, data: L2AnalyzerDTO) -> float:
        device = next(data.new_model.parameters()).device
        squared_sum = torch.zeros(1, device=device)

        for p_new, p_prev in zip(
            data.new_model.parameters(), data.prev_model.parameters()
        ):
            diff = p_new - p_prev
            squared_sum += torch.sum(diff * diff)

        return torch.sqrt(squared_sum).item()
