from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base_analyzer import BaseAnalyzer


@dataclass
class CosineSimilarityAnalyzerDTO:
    matrix_a: torch.Tensor
    matrix_b: torch.Tensor


class CosineSimilarityAnalyzer(BaseAnalyzer):
    def analyze(self, data: CosineSimilarityAnalyzerDTO):
        a_flat = data.matrix_a.view(-1)
        b_flat = data.matrix_b.view(-1)

        norm_a = a_flat.norm()
        norm_b = b_flat.norm()
        if norm_a == 0 or norm_b == 0:
            return 0.0

        a_batch = a_flat.unsqueeze(0)
        b_batch = b_flat.unsqueeze(0)

        cos_sim = F.cosine_similarity(a_batch, b_batch, dim=1).item()
        return cos_sim
