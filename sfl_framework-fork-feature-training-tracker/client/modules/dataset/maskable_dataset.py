import random
from collections import Counter

from torch.utils.data import Dataset


class MaskableDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.original_dataset = dataset
        self.indices = indices

        if hasattr(dataset, "select"):
            self.dataset = dataset.select(indices)
            self.is_subset = True
            if "label" in dataset.column_names:
                labels = [int(i.item()) for i in self.dataset["label"]]
            else:
                labels = [int(item[1]) for item in self.dataset]
        else:
            self.dataset = dataset
            self.is_subset = False
            labels = []
            for idx in indices:
                _, label = dataset[idx]
                labels.append(label)

        self.label_distribution = dict(Counter(labels))

        self.indices_per_label = {}
        if self.is_subset:
            for label in self.label_distribution:
                self.indices_per_label[label] = [
                    i for i in range(len(labels)) if labels[i] == label
                ]
        else:
            for i, label in enumerate(labels):
                if label not in self.indices_per_label:
                    self.indices_per_label[label] = []
                self.indices_per_label[label].append(self.indices[i])

        self.amount_per_label = self.label_distribution.copy()
        self.limited_indices = []

        self.shard_positions = {label: 0 for label in self.label_distribution}

        self.update_limited_indices()

    def __len__(self):
        return len(self.limited_indices)

    def __getitem__(self, index: int):
        actual_index = self.limited_indices[index]
        return self.dataset[actual_index]

    def update_limited_indices(self):
        self.limited_indices = []

        for label, amount in self.amount_per_label.items():
            if amount == 0:
                continue

            label_int = int(label)
            indices_list = self.indices_per_label[label_int]
            total_indices = len(indices_list)

            if total_indices == 0:
                continue

            if amount > total_indices:
                full_cycles = amount // total_indices
                remainder = amount % total_indices

                self.limited_indices.extend(indices_list * full_cycles)

                if remainder > 0:
                    start_pos = self.shard_positions[label_int]
                    indices_to_add = []

                    if start_pos + remainder > total_indices:
                        indices_to_add.extend(indices_list[start_pos:])
                        remaining = remainder - (total_indices - start_pos)
                        indices_to_add.extend(indices_list[:remaining])
                        self.shard_positions[label_int] = remaining
                    else:
                        indices_to_add.extend(
                            indices_list[start_pos : start_pos + remainder]
                        )
                        self.shard_positions[label_int] = (
                            start_pos + remainder
                        ) % total_indices

                    self.limited_indices.extend(indices_to_add)
            else:
                start_pos = self.shard_positions[label_int]
                indices_to_add = []

                if start_pos + amount > total_indices:
                    indices_to_add.extend(indices_list[start_pos:])
                    remaining = amount - (total_indices - start_pos)
                    indices_to_add.extend(indices_list[:remaining])
                    self.shard_positions[label_int] = remaining
                else:
                    indices_to_add.extend(indices_list[start_pos : start_pos + amount])
                    self.shard_positions[label_int] = (
                        start_pos + amount
                    ) % total_indices

                self.limited_indices.extend(indices_to_add)

        random.shuffle(self.limited_indices)
        #print(f"limited indices: {len(self.limited_indices)}")

    def update_amount_per_label(self, amount_per_label: dict[int, int]):
        self.amount_per_label = amount_per_label
        self.update_limited_indices()

    def get_label_distribution(self) -> dict:
        return {str(k): v for k, v in self.label_distribution.items()}

    def get_original_length(self):
        return len(self.indices)

    def reset_shard_positions(self):
        self.shard_positions = {label: 0 for label in self.label_distribution}
