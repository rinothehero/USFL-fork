"""
MaskableDataset for USFL data balancing (trimming/replication).

Wraps a base dataset with per-label sample count control.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from torch.utils.data import Dataset


class MaskableDataset(Dataset):
    """
    Dataset wrapper supporting per-label trimming and replication.

    - Trimming: reduce samples of a label below a threshold
    - Replication: increase samples by cycling through available indices
    """

    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base = base_dataset
        self.original_indices = list(indices)

        # Extract targets
        if hasattr(base_dataset, "targets"):
            targets = base_dataset.targets
        elif hasattr(base_dataset, "labels"):
            targets = base_dataset.labels
        else:
            raise ValueError("Dataset has no targets or labels")

        # Build label → original index mapping
        self._label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in indices:
            label = int(targets[idx])
            self._label_to_indices[label].append(idx)

        # Active indices (modified by update_amount_per_label)
        self.active_indices = list(indices)
        # Track cycling position per label for reproducible replication
        self._cycle_pos: Dict[int, int] = {}

    def update_amount_per_label(self, amounts: Dict[int, int]):
        """
        Set target sample count per label.

        If target < available: trim (first target samples)
        If target > available: replicate (cycle through samples)
        If target == 0 or label missing: skip that label
        """
        new_indices = []

        for label, target in amounts.items():
            available = self._label_to_indices.get(label, [])
            if not available or target <= 0:
                continue

            n = len(available)
            if target <= n:
                # Trim
                new_indices.extend(available[:target])
            else:
                # Replicate by cycling
                pos = self._cycle_pos.get(label, 0)
                for i in range(target):
                    new_indices.append(available[(pos + i) % n])
                self._cycle_pos[label] = (pos + target) % n

        random.shuffle(new_indices)
        self.active_indices = new_indices

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        return self.base[self.active_indices[idx]]

    @property
    def label_distribution(self) -> Dict[int, int]:
        """Current label distribution of active indices."""
        if hasattr(self.base, "targets"):
            targets = self.base.targets
        else:
            targets = self.base.labels

        dist: Dict[int, int] = defaultdict(int)
        for idx in self.active_indices:
            dist[int(targets[idx])] += 1
        return dict(dist)
