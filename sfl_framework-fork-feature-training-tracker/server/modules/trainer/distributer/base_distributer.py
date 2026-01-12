from abc import ABC, abstractmethod

import numpy as np


class BaseDistributer(ABC):
    def _remove_fraction_of_labels(self, label: list[int], fraction: float):
        label_list = list(set(label))
        remove_labels = np.random.choice(
            label_list, int(len(label_list) * fraction), replace=False
        )
        print("remove_labels: ", remove_labels)

        for remove_label in remove_labels:
            indices = [i for i, x in enumerate(label) if x == remove_label]
            half_size = len(indices) // 2
            for idx in indices[:half_size]:
                label[idx] = int("9999999")

        return label

    @abstractmethod
    def distribute(self, clients: list[str]):
        pass
