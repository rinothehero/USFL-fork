import random
from typing import List

from .base_selector import BaseSelector


class MissingClassSelector(BaseSelector):
    def __init__(self, config):
        self.config = config

    def select(self, n: int, client_ids: List[int], data=None) -> List[int]:
        client_informations = data["client_informations"]
        num_classes = data["num_classes"]
        target_missing = self.config.num_missing_class

        target_union_size = num_classes - target_missing

        max_attempts = 100

        for attempt in range(max_attempts):
            try:
                client_label_sets = {}
                for client_id in client_ids:
                    distribution = client_informations[client_id]["dataset"][
                        "label_distribution"
                    ]
                    label_set = {
                        int(label) for label, count in distribution.items() if count > 0
                    }
                    client_label_sets[client_id] = label_set

                temp_client_ids = client_ids.copy()
                random.shuffle(temp_client_ids)
                remaining = temp_client_ids.copy()
                selected = []
                aggregated_set = set()

                while len(selected) < n:
                    if len(aggregated_set) < target_union_size:
                        candidate = None
                        for cid in remaining:
                            new_union = aggregated_set | client_label_sets[cid]
                            if (
                                len(new_union) > len(aggregated_set)
                                and len(new_union) <= target_union_size
                            ):
                                candidate = cid
                                break
                        if candidate is None:
                            raise ValueError(
                                "새로운 라벨을 추가하여 목표 union size를 달성할 수 없습니다."
                            )
                        selected.append(candidate)
                        aggregated_set |= client_label_sets[candidate]
                        remaining.remove(candidate)
                    elif len(aggregated_set) == target_union_size:
                        candidate = None
                        for cid in remaining:
                            if client_label_sets[cid].issubset(aggregated_set):
                                candidate = cid
                                break
                        if candidate is None:
                            raise ValueError(
                                "목표 union size를 유지하면서 충분한 클라이언트를 선택할 수 없습니다."
                            )
                        selected.append(candidate)
                        remaining.remove(candidate)

                print(aggregated_set)
                print(target_union_size)
                if len(aggregated_set) != target_union_size:
                    raise ValueError(
                        "최종 aggregated label set이 요구하는 missing class 조건을 만족하지 않습니다."
                    )

                return selected

            except ValueError as e:
                continue

        raise ValueError("최종 조건을 만족하는 클라이언트 조합을 찾지 못했습니다.")
