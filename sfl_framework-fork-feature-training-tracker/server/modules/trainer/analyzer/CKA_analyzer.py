from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .base_analyzer import BaseAnalyzer


@dataclass
class CKAAnalyzerDTO:
    matrix_a: torch.Tensor
    matrix_b: torch.Tensor


# Reference: https://github.com/google-research/google-research/blob/master/representation_similarity/
# Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
# Similarity of Neural Network Representations Revisited.
# In International Conference on Machine Learning (pp. 3519-3529).
class CKAAnalyzer(BaseAnalyzer):
    def center_gram(self, gram, unbiased=False):
        """Center a symmetric Gram matrix.

        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
            gram: A num_examples x num_examples symmetric matrix.
            unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.

        Returns:
            A symmetric matrix with centered columns and rows.
        """
        if not np.allclose(gram, gram.T):
            raise ValueError("Input must be a symmetric matrix.")
        gram = gram.copy()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def cka(self, gram_x, gram_y, debiased=False):
        """Compute CKA.

        Args:
            gram_x: A num_examples x num_examples Gram matrix.
            gram_y: A num_examples x num_examples Gram matrix.
            debiased: Use unbiased estimator of HSIC. CKA may still be biased.

        Returns:
            The value of CKA between X and Y.
        """
        gram_x = self.center_gram(gram_x, unbiased=debiased)
        gram_y = self.center_gram(gram_y, unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    def analyze(self, data: CKAAnalyzerDTO):
        """
        Calculate CKA similarity between two activation matrices.

        Args:
            data: CKAAnalyzerDTO containing the activation matrices to compare

        Returns:
            The CKA similarity score between the activation matrices
        """
        a = data.matrix_a.detach().cpu().numpy()
        b = data.matrix_b.detach().cpu().numpy()

        gram_a = a.reshape(a.shape[0], -1) @ a.reshape(a.shape[0], -1).T
        gram_b = b.reshape(b.shape[0], -1) @ b.reshape(b.shape[0], -1).T

        return self.cka(gram_a, gram_b, debiased=True)
