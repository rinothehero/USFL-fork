from dataclasses import dataclass

import torch
import torch.nn as nn

from .base_analyzer import BaseAnalyzer


@dataclass
class MeanVarianceAnalyzerDTO:
    model: nn.Module


class MeanVarianceAnalyzer(BaseAnalyzer):
    @torch.no_grad()
    def analyze(self, data: MeanVarianceAnalyzerDTO) -> float:
        parameters = [p.view(-1) for p in data.model.parameters() if p.requires_grad]
        if not parameters:
            return 0.0

        all_params = torch.cat(parameters)

        mean = torch.mean(all_params)
        variance = torch.mean((all_params - mean) ** 2)

        return mean.item(), variance.item()
