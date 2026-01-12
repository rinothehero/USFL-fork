import math
import random
from typing import TYPE_CHECKING, List

import numpy as np

from .base_selector import BaseSelector

if TYPE_CHECKING:
    from typing import List

    from server_args import Config


class FedCBSSelector(BaseSelector):
    def __init__(self, config: "Config"):
        self.config = config
        self.client_selected_count = {}
        self.round = 0

        self.beta_start = 1
        self.beta_step = 1
        self.lamda = 10.0
        self.eps = 1e-12

    def compute_local_label_distribution(self, label_distribution, num_classes):
        size = sum(label_distribution.values())
        alpha_i = np.zeros(num_classes, dtype=np.float32)

        for label in label_distribution.keys():
            alpha_i[int(label)] = label_distribution[label] / size

        return alpha_i, size

    def build_S_matrix(self, label_distributions, num_classes=10):
        client_ids = label_distributions.keys()
        N = len(client_ids)

        data_sizes = []
        alphas = []
        for client_id in client_ids:
            label_distribution = label_distributions[client_id]
            alpha_i, size = self.compute_local_label_distribution(
                label_distribution, num_classes=num_classes
            )
            alphas.append(alpha_i)
            data_sizes.append(size)

        alphas = np.array(alphas)
        data_sizes = np.array(data_sizes)

        S = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                dot_ij = np.dot(alphas[i], alphas[j])
                S[i, j] = data_sizes[i] * data_sizes[j] * dot_ij

        return S, data_sizes, label_distributions

    def get_client_selected_count(self, client_id):
        if client_id not in self.client_selected_count:
            self.client_selected_count[client_id] = 0

        return self.client_selected_count[client_id]

    def update_client_selected_count(self, client_id):
        if client_id not in self.client_selected_count:
            self.client_selected_count[client_id] = 0

        self.client_selected_count[client_id] += 1

    def select(self, n: int, client_ids: List[int], data=None):
        self.round += 1

        client_informations = data["client_informations"]
        num_classes = data["num_classes"]

        label_distributions = {
            client_id: client_informations[client_id]["dataset"]["label_distribution"]
            for client_id in client_ids
        }
        dataset_sizes = {
            client_id: client_informations[client_id]["dataset"]["size"]
            for client_id in client_ids
        }

        S, dataset_sizes, label_distributions = self.build_S_matrix(
            label_distributions, num_classes
        )
        selected = []
        for m in range(1, n + 1):
            beta_m = self.beta_start + (m - 1) * self.beta_step
            not_selected = [
                client_id for client_id in client_ids if client_id not in selected
            ]

            if len(not_selected) == 0:
                break

            scores = []

            for candi in not_selected:
                new_selected = selected + [candi]
                sum_sizes = sum(dataset_sizes[client_id] for client_id in new_selected)

                numerator = 0.0
                for i in new_selected:
                    for j in new_selected:
                        numerator += S[i, j]

                val_qcid = (numerator / (sum_sizes**2 + self.eps)) + self.eps
                exploit_score = 1.0 / (val_qcid**beta_m)

                t_c = float(self.get_client_selected_count(candi))
                explore_score = self.lamda * math.sqrt(
                    3.0 * math.log(self.round + 1.0) / (2.0 * (t_c + self.eps))
                )

                total_score = exploit_score + explore_score
                scores.append((candi, total_score))

            total_sum = sum(s for _, s in scores)
            r = random.random() * total_sum
            acc = 0.0
            chosen = scores[-1][0]
            for candi, sc in scores:
                acc += sc
                if acc >= r:
                    chosen = candi
                    break

            selected.append(chosen)
            self.update_client_selected_count(chosen)

        return selected
