import copy
from typing import TYPE_CHECKING, List

import torch

from .base_aggregator import BaseAggregator
from ..utils.training_tracker import TrainingTracker

if TYPE_CHECKING:
    from typing import List


class USFLAggregator(BaseAggregator):
    """
    Label-capped weighting
      1) 전역 분포 P_Q에서 각 라벨 l 의 최대 가중치  w_hat[l] = n_l / N
      2) 디바이스 j 가 라벨 l 에 대해 기여한 비율 = n_l,j / n_l
      3) 디바이스 가중치   w_j = Σ_l w_hat[l] · (n_l,j / n_l)
      4) Σ_j w_j = 1  → 집계는 확률적 가중 평균
    """

    def aggregate(
        self,
        models: List[torch.nn.Module],
        params: List[dict],
    ):
        if not models:
            raise ValueError("No models to aggregate")

        # Extract round_number from params (if available)
        round_number = params[0].get("round_number", 0) if params else 0

        # Extract client_ids from params (if available)
        client_ids = [p.get("client_id", i) for i, p in enumerate(params)]

        # 1) 각 디바이스의 라벨 분포 (원본)
        label_dists = [p["label_distribution"] for p in params]

        # 1-1) 각 디바이스의 실제 사용 라벨 분포 (augmented)
        augmented_dists = [p.get("augmented_label_counts", {}) for p in params]

        # 2) 전역 라벨 집합과 라벨별 총합 (원본 기준)
        labels = sorted({lab for d in label_dists for lab in d})
        total_per_label = {
            lab: sum(d.get(lab, 0) for d in label_dists) for lab in labels
        }
        global_total = sum(total_per_label.values()) or 1

        # 2-1) 실제 사용 데이터 라벨별 총합 (augmented)
        augmented_total_per_label = {
            lab: sum(d.get(lab, 0) for d in augmented_dists) for lab in labels
        }

        # 3) 라벨별 최대 가중치 w_hat[l]
        max_weight = {lab: total_per_label[lab] / global_total for lab in labels}

        # 4) 디바이스별 최종 가중치
        device_weights = []
        for dist in label_dists:
            weight_j = 0.0
            for lab in labels:
                if total_per_label[lab] == 0:
                    continue
                share = dist.get(lab, 0) / total_per_label[lab]  # 기여율
                weight_j += max_weight[lab] * share
            device_weights.append(weight_j)

        # 정규화(수치 안정성)
        norm = sum(device_weights) or 1.0
        device_weights = [w / norm for w in device_weights]
        print("device_weights:", device_weights)

        # --- Log Aggregation Weights ---
        TrainingTracker.log_aggregation_weights(
            round_number=round_number,
            client_ids=client_ids,
            client_weights=device_weights,
            label_distributions=label_dists,
            total_per_label=total_per_label,
            augmented_label_distributions=augmented_dists,
            augmented_total_per_label=augmented_total_per_label,
        )
        # --- End Log ---

        # 5) 가중 평균 집계
        agg_state = {
            k: torch.zeros_like(p, dtype=torch.float32)
            for k, p in models[0].state_dict().items()
        }
        for model, w in zip(models, device_weights):
            for k, p in model.state_dict().items():
                agg_state[k] += p.float() * w

        new_model = copy.deepcopy(models[0])
        new_model.load_state_dict(agg_state)
        return new_model

