from dataclasses import dataclass

import torch
import torch.nn as nn

from .base_analyzer import BaseAnalyzer
from .cosine_similarity_analyzer import CosineSimilarityAnalyzer


@dataclass
class ConflictRatioAnalyzerDTO:
    matrix_a: torch.Tensor
    matrix_b: torch.Tensor


class ConflictRatioAnalyzer(BaseAnalyzer):
    def analyze(self, data: ConflictRatioAnalyzerDTO):
        """
        Calculate the conflict ratio between two gradient matrices.

        The conflict ratio measures the degree to which two gradient directions
        are in conflict with each other. It is calculated as the ratio of
        the negative dot product to the product of the norms.

        Args:
            data: ConflictRatioAnalyzerDTO containing the two gradient matrices

        Returns:
            The conflict ratio between the two gradient matrices
        """
        cosine_sim = CosineSimilarityAnalyzer().analyze(data)

        conflict_ratio = max(0.0, -cosine_sim)

        return conflict_ratio
