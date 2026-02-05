import asyncio
import math
import random
from collections import Counter, defaultdict, deque
from itertools import combinations
from typing import TYPE_CHECKING, Tuple, Union

import torch
from torch.utils.data import DataLoader

from ..scheduler.batch_scheduler import create_schedule
from ..utils.usfl_logger import USFLLogger
from ..utils.training_tracker import TrainingTracker
from .base_stage_organizer import BaseStageOrganizer
from .in_round.in_round import InRound
from .post_round.post_round import PostRound
from .pre_round.pre_round import PreRound

# G Measurement V2 (Oracle-based)
from utils.g_measurement import (
    GMeasurementSystem,
    snapshot_model,
    restore_model,
    get_param_names,
    compute_g_metrics,
)

# Drift Measurement (SCAFFOLD-style)
from utils.drift_measurement import DriftMeasurementTracker

if TYPE_CHECKING:
    from server_args import Config

    from ...dataset.base_dataset import BaseDataset
    from ...global_dict.global_dict import GlobalDict
    from ...model.base_model import BaseModel
    from ...ws.connection import Connection
    from ...ws.inmemory_connection import InMemoryConnection
    from ..aggregator.base_aggregator import BaseAggregator
    from ..seletor.base_selector import BaseSelector
    from ..splitter.base_splitter import BaseSplitter


class USFLStageOrganizer(BaseStageOrganizer):
    def __init__(
        self,
        config: "Config",
        connection: Union["Connection", "InMemoryConnection"],
        global_dict: "GlobalDict",
        aggregator: "BaseAggregator",
        model: "BaseModel",
        dataset: "BaseDataset",
        selector: "BaseSelector",
        splitter: "BaseSplitter",
    ):
        self.config = config
        self.connection = connection
        self.global_dict = global_dict

        self.aggregator = aggregator
        self.model = model
        self.testloader = dataset.get_testloader()
        self.num_classes = dataset.get_num_classes()
        self.selector = selector
        self.splitter = splitter

        self.pre_round = PreRound(config, global_dict)
        self.in_round = InRound(config, global_dict)
        self.post_round = PostRound(config, global_dict)

        self.round_start_time = 0
        self.round_end_time = 0
        self.selected_clients = []
        self.split_models = []

        self.selected_count = {
            str(client_id): 0 for client_id in range(self.config.num_clients)
        }

        # τ : temperature for g-LA (default -1.0)
        self.tau: float = getattr(self.config, "logit_tau", -0.5)
        self.usage_decay_factor: float = getattr(
            self.config, "usage_decay_factor", 0.99
        )

        # per-client prior cache set during _pre_round
        self.client_priors: dict[str, dict[str, float]] = {}
        self.client_label_counts: dict[str, dict[str, int]] = {}

        # Initialize USFL logger with unique filename
        USFLLogger.initialize(config)
        # Initialize Training Tracker for iteration-level logging
        TrainingTracker.initialize(config)

        # G Measurement V2 (Oracle-based) - lazy initialization
        self._dataset = dataset  # Store reference for lazy loading (G measurement)
        self.g_measurement_system = None
        if getattr(config, "enable_g_measurement", False):
            diagnostic_rounds = getattr(config, "diagnostic_rounds", "1,3,5")
            if isinstance(diagnostic_rounds, str):
                diagnostic_rounds = [int(x) for x in diagnostic_rounds.split(",")]
            measurement_mode = getattr(config, "g_measurement_mode", "single")
            measurement_k = getattr(config, "g_measurement_k", 5)
            self.g_measurement_system = GMeasurementSystem(
                diagnostic_rounds=diagnostic_rounds,
                device=config.device,
                use_variance_g=getattr(config, "use_variance_g", False),
                measurement_mode=measurement_mode,
                measurement_k=measurement_k,
            )

        # Drift Measurement (SCAFFOLD-style)
        self.drift_tracker = None
        if getattr(config, "enable_drift_measurement", False):
            self.drift_tracker = DriftMeasurementTracker()
            print("[Drift] DriftMeasurementTracker initialized (USFL)")

    @staticmethod
    def _get_exponential_bin(usage_count: int) -> int:
        """
        Maps a usage count to an exponential bin.

        Bins: 0 → 0, 1 → 1, 2-3 → 2, 4-7 → 4, 8-15 → 8, etc.
        Formula: bin = 2^floor(log2(usage_count)) for usage_count > 1

        This ensures logarithmic growth instead of linear growth.
        """
        if usage_count <= 1:
            return usage_count
        # Get the highest power of 2 less than or equal to usage_count
        return 1 << (usage_count.bit_length() - 1)

    @staticmethod
    def _get_bin_range(bin_key: int) -> tuple:
        """
        Returns the (min, max) usage count range for a bin.

        Examples:
        - bin 0 → (0, 0)
        - bin 1 → (1, 1)
        - bin 2 → (2, 3)
        - bin 4 → (4, 7)
        - bin 8 → (8, 15)
        """
        if bin_key <= 1:
            return (bin_key, bin_key)
        return (bin_key, 2 * bin_key - 1)

    def _print_memory_usage(self, step):
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{step} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")

    def _calculate_batch_size(
        self, dataset_sizes: dict[int, int], selected_clients: list[int]
    ):
        total_dataset_size = sum(dataset_sizes.values())
        if total_dataset_size == 0:
            # 데이터가 없는 경우 0으로 나누기 오류 방지
            return (
                0,
                {str(cid): 0 for cid in selected_clients},
                {str(cid): 0 for cid in selected_clients},
            )

        global_batch_size = int(
            self.config.batch_size
            * (len(selected_clients) / self.config.num_clients_per_round)
        )

        print(
            f"batch size: {self.config.batch_size}, selected_count: {len(selected_clients)}, num_clients_per_round: {self.config.num_clients_per_round}, global batch size: {global_batch_size}"
        )

        # 1. 각 클라이언트의 이상적인 (소수점 포함) 배치 사이즈와 소수부 계산
        ideal_sizes = {}
        for client_id, dataset_size in dataset_sizes.items():
            ideal_size = global_batch_size * dataset_size / total_dataset_size
            ideal_sizes[str(client_id)] = (ideal_size, ideal_size - int(ideal_size))

        # 2. 기본 배치 사이즈 할당 (소수점 버림)
        batch_sizes = {cid: int(val[0]) for cid, val in ideal_sizes.items()}

        zero_batch_clients = []
        # 5. 최종 배치 사이즈 보정 (0이하 방지)
        for client_id in batch_sizes:
            if batch_sizes[client_id] <= 0:
                batch_sizes[client_id] = 1
                zero_batch_clients.append(client_id)
                print(f"Warning: Client {client_id} had zero batch size, set to 1")

        # 3. 버려진 소수점들의 합 (나머지) 계산
        remainder = global_batch_size - sum(batch_sizes.values())

        # 4. 소수부가 가장 컸던 클라이언트 순으로 나머지 분배
        sorted_clients = sorted(
            ideal_sizes.keys(), key=lambda cid: ideal_sizes[cid][1], reverse=True
        )

        for i in range(remainder):
            client_to_increment = sorted_clients[i % len(sorted_clients)]
            if client_to_increment not in zero_batch_clients:
                batch_sizes[client_to_increment] += 1

        iterations_per_client = {
            str(client_id): (
                dataset_sizes[str(client_id)] // batch_sizes[str(client_id)]
                if batch_sizes.get(str(client_id), 0) > 0
                else 0
            )
            for client_id in dataset_sizes.keys()
        }

        return global_batch_size, batch_sizes, iterations_per_client

    def _concatenate_activations(self, concatenated_activations, activation):
        if not concatenated_activations:
            if isinstance(activation["outputs"], tuple):
                concatenated_activations["outputs"] = [
                    torch.tensor([], device=self.config.device)
                    for _ in range(len(activation["outputs"]))
                ]
            else:
                concatenated_activations["outputs"] = torch.tensor(
                    [], device=self.config.device
                )

            for key in ["labels", "attention_mask"]:
                if key in activation:
                    concatenated_activations[key] = torch.tensor(
                        [], device=self.config.device
                    )

        if isinstance(activation["outputs"], tuple):
            outputs_list = list(activation["outputs"])
            if isinstance(concatenated_activations["outputs"], tuple):
                concatenated_outputs = list(concatenated_activations["outputs"])
            else:
                concatenated_outputs = concatenated_activations["outputs"]

            for i in range(len(outputs_list)):
                if isinstance(outputs_list[i], torch.Tensor):
                    outputs_list[i] = outputs_list[i].to(self.config.device)

                    if len(concatenated_outputs[i]) == 0:
                        concatenated_outputs[i] = (
                            outputs_list[i].detach().clone().requires_grad_(True)
                        )
                    else:
                        concatenated_outputs[i] = (
                            torch.cat(
                                [
                                    concatenated_outputs[i].to(self.config.device),
                                    outputs_list[i],
                                ],
                                dim=0,
                            )
                            .detach()
                            .clone()
                            .requires_grad_(True)
                        )

            activation["outputs"] = tuple(outputs_list)
            concatenated_activations["outputs"] = tuple(concatenated_outputs)

        else:
            activation_output = activation["outputs"].to(self.config.device)

            if len(concatenated_activations["outputs"]) == 0:
                concatenated_activations["outputs"] = activation_output
            else:
                concatenated_activations["outputs"] = torch.cat(
                    [concatenated_activations["outputs"], activation_output],
                    dim=0,
                )

        for key in ["labels", "attention_mask"]:
            if key in activation:
                tensor_on_device = activation[key].to(self.config.device)

                if (
                    key not in concatenated_activations
                    or len(concatenated_activations[key]) == 0
                ):
                    concatenated_activations[key] = tensor_on_device
                else:
                    concatenated_activations[key] = torch.cat(
                        [concatenated_activations[key], tensor_on_device],
                        dim=0,
                    )

        return concatenated_activations

    def _calculate_scaled_kl_divergence(self, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        min_label = labels.min().item()
        if min_label == 1:
            adjusted_labels = labels - 1
        else:
            adjusted_labels = labels

        counts = torch.bincount(adjusted_labels, minlength=self.num_classes)
        total = counts.sum()
        empirical_dist = counts.float() / total

        uniform_dist = torch.ones(self.num_classes) / self.num_classes

        kl_div = 0.0
        for p, q in zip(empirical_dist, uniform_dist):
            if p > 0:
                kl_div += p * torch.log(p / q)

        max_kl_div = torch.log(torch.tensor(self.num_classes, dtype=torch.float))

        scaled_kl_div = kl_div / max_kl_div

        return scaled_kl_div.item()

    def _shuffle_gradients(self, activations, grads):
        C = self.num_classes
        is_tuple = isinstance(grads, tuple)
        grad_list = list(grads) if is_tuple else [grads]
        grad_list = [g.clone() for g in grad_list]
        device = grad_list[0].device

        # ---------- 1) 메타 수집 ----------
        all_labels, pos2cid = [], {}
        cid2pos, cid2labels = defaultdict(list), defaultdict(list)

        for act in activations:
            cid = act["client_id"]
            labs = act["labels"]
            labs = labs.tolist() if isinstance(labs, torch.Tensor) else labs

            start = len(all_labels)
            all_labels.extend(labs)
            end = len(all_labels)

            cid2pos[cid].extend(range(start, end))
            cid2labels[cid].extend(labs)
            for p in range(start, end):
                pos2cid[p] = cid

        N_per_cid = {cid: len(v) for cid, v in cid2pos.items()}
        keep_limit = {cid: N // C for cid, N in N_per_cid.items()}

        # ---------- 2) 고정ㆍ도너 큐 구성 ----------
        fixed_pos = set()
        donor_queue = {
            lbl: deque() for lbl in range(C)
        }  # label -> deque[(tensor, donor)]
        cid_donor_lbls = defaultdict(set)  # 추후 검사용(원한다면)

        for cid in cid2pos:
            limit = keep_limit[cid]
            lbl2pos = defaultdict(list)
            for p in cid2pos[cid]:
                lbl2pos[all_labels[p]].append(p)

            for lbl, plist in lbl2pos.items():
                keep_cnt = min(len(plist), limit)
                fixed_pos.update(plist[:keep_cnt])

                # 초과분 → donor_queue
                for p in plist[keep_cnt:]:
                    donor_queue[lbl].append(
                        (tuple(g[p].clone() for g in grad_list), cid)
                    )
                    cid_donor_lbls[cid].add(lbl)

        # 각 클라이언트가 현재 가진(고정) per-class 카운트
        cid_class_cnt = {
            cid: Counter(all_labels[p] for p in fixed_pos if pos2cid[p] == cid)
            for cid in cid2pos
        }

        # 남은 빈 슬롯(포지션) 저장
        empty_pos = defaultdict(list)  # cid -> [position,...]
        for p in range(len(all_labels)):
            if p not in fixed_pos:
                empty_pos[pos2cid[p]].append(p)

        # ---------- 3) 클래스별 필요량 계산 ----------
        cid_needs = defaultdict(lambda: defaultdict(int))  # cid -> lbl -> deficit
        for cid in cid2pos:
            for lbl in range(C):
                need = keep_limit[cid] - cid_class_cnt[cid][lbl]
                cid_needs[cid][lbl] = max(0, need)

        # ---------- 4) 부족 클래스 채우기 ----------
        # 라벨 순회하며 각 클라이언트에 필요한 수만큼 donor_queue에서 pop
        for lbl in range(C):
            queue = donor_queue[lbl]
            # 클라이언트 순서를 매 라운드 섞어 공정성 확보
            clients = list(cid2pos.keys())
            random.shuffle(clients)

            for cid in clients:
                need = cid_needs[cid][lbl]
                if need == 0:
                    continue

                tries = 0
                # 큐를 돌면서 donor != cid 인 항목 pop
                while need > 0 and queue and tries < len(queue):
                    tensor, donor_id = queue.popleft()
                    if donor_id == cid:
                        # 자신이 기부한 gradient → 뒤로 보내고 skip
                        queue.append((tensor, donor_id))
                        tries += 1
                        continue

                    # 빈 슬롯 하나 꺼내 채움
                    pos = empty_pos[cid].pop()
                    for idx, grad_item in enumerate(grad_list):
                        grad_item[pos] = tensor[idx].to(device)
                    cid_class_cnt[cid][lbl] += 1
                    need -= 1
                    tries = 0  # 성공했으면 다시 0으로

                cid_needs[cid][lbl] = need  # 남은 deficit(일반적으로 0)

        # ---------- 5) 남은 빈 슬롯은 donor 제약 안걸린 아무 클래스 채우기 ----------
        # (불가피하게 donor-class 도 들어갈 수 있지만, 실험상 대부분 1차에서 해결)
        leftovers = []
        for lbl, q in donor_queue.items():
            leftovers.extend(list(q))
        random.shuffle(leftovers)

        for cid in cid2pos:
            while empty_pos[cid]:
                if not leftovers:
                    raise RuntimeError("도너 큐 고갈: 슬롯을 모두 채울 수 없습니다.")
                tensor, donor_id = leftovers.pop()
                # donor == cid 면 그냥 넣을지 말지 선택. 여기선 허용(필요에 따라 skip 가능).
                pos = empty_pos[cid].pop()
                for idx, grad_item in enumerate(grad_list):
                    grad_item[pos] = tensor[idx].to(device)

        return tuple(grad_list) if is_tuple else grad_list[0]

    async def _pre_round(self, round_number: int):
        # 클라이언트 정보 및 준비 대기
        await self.pre_round.wait_for_client_informations()
        await self.pre_round.wait_for_clients()

        # 클라이언트 정보와 선택된 클라이언트 설정
        client_informations = self.global_dict.get("client_informations")

        # "시간 감쇠" 기능이 켜져 있으면, 기억(누적 사용량)을 감쇠시킴
        if getattr(self.config, "use_cumulative_usage", False):
            # --- Initialize or decay cumulative usage ---
            cumulative_usage = self.global_dict.get("cumulative_usage")
            if not cumulative_usage:  # First time initialization
                USFLLogger.log_cumulative_usage_init()
                for cid, info in client_informations.items():
                    client_id_str = str(cid)
                    cumulative_usage[client_id_str] = {}
                    ld = info["dataset"]["label_distribution"]
                    for lbl, count in ld.items():
                        # {bin_key: count} - exponential bins for memory efficiency
                        # bin 0: data used 0 times
                        cumulative_usage[client_id_str][lbl] = {0: count}
            else:
                # Time decay logic (if needed, can be re-implemented for the new structure)
                # For now, we focus on the freshness logic, so decay is disabled.
                pass

        # Log round separator for feature logging
        USFLLogger.log_round_separator(round_number)

        selection_data = {
            "client_informations": client_informations,
            "num_classes": self.num_classes,
            "batch_size": self.config.batch_size,
            "fresh_scoring": self.config.use_fresh_scoring,
            "round_number": round_number,  # For logging purposes
        }
        # "시간 감쇠" 기능이 켜져 있으면, 선택에 필요한 추가 정보로 감쇠된 기억을 전달
        if getattr(self.config, "use_cumulative_usage", False):
            selection_data["cumulative_usage"] = self.global_dict.get(
                "cumulative_usage"
            )

        # Select clients (selector has internal retry logic with up to 1000 attempts)
        print(
            f"[Round {round_number}] Selecting clients..."
        )  # Terminal progress update
        self.selected_clients = self.pre_round.select_clients(
            self.selector,
            self.connection,
            selection_data,
        )

        # Sanity check: Verify all class labels are covered by selected clients
        global_dataset_sizes = {str(label): 0 for label in range(self.num_classes)}
        for client_id in self.selected_clients:
            label_distribution = client_informations[client_id]["dataset"][
                "label_distribution"
            ]
            for label, count in label_distribution.items():
                global_dataset_sizes[label] += count

        # If any label is missing, the selector failed (should not happen with proper retry logic)
        if any(size == 0 for size in global_dataset_sizes.values()):
            missing_labels = [
                label for label, size in global_dataset_sizes.items() if size == 0
            ]
            raise RuntimeError(
                f"[Round {round_number}] CRITICAL ERROR: Selector returned invalid client group. "
                f"Selected clients {self.selected_clients} are missing labels: {missing_labels}. "
                "This indicates a bug in the selector's retry logic or insufficient data distribution. "
                "Consider increasing dirichlet_alpha or num_clients_per_round."
            )

        # Terminal: Show selected clients summary
        print(f"[Round {round_number}] Selected clients: {self.selected_clients}")

        # 전역 및 각 클라이언트별 라벨별 데이터셋 크기 초기화
        local_dataset_sizes = {
            str(client_id): {str(label): 0 for label in range(self.num_classes)}
            for client_id in self.selected_clients
        }
        local_dataset_proportion_per_label = {
            str(client_id): {str(label): 0 for label in range(self.num_classes)}
            for client_id in self.selected_clients
        }

        # 각 클라이언트별 데이터셋 크기 집계
        for client_id in self.selected_clients:
            client_id_str = str(client_id)
            self.selected_count[client_id_str] += 1

            label_distribution = client_informations[client_id]["dataset"][
                "label_distribution"
            ]
            for label, count in label_distribution.items():
                local_dataset_sizes[client_id_str][label] += count

            total_cnt = sum(label_distribution.values())
            self.client_priors[client_id_str] = {
                str(lbl): (
                    label_distribution.get(str(lbl), 0) / total_cnt
                    if total_cnt > 0
                    else 0.0
                )
                for lbl in range(self.num_classes)
            }

        # 각 클라이언트가 전체 데이터셋의 각 레이블(클래스)을 얼마나 차지하고 있는지 그 비율을 계산하는 부분
        for client_id in self.selected_clients:
            client_id_str = str(client_id)
            # 모든 레이블 각각에 대해
            for label in range(self.num_classes):
                label_str = str(label)
                # 만약 전체 데이터셋(현재 선택된 클라이언트 분포 합집합)에 해당 레이블의 데이터가 존재한다면,
                if global_dataset_sizes[label_str] > 0:
                    # 현재 클라이언트가 가진 해당 레이블의 데이터 수를 전체 데이터셋(현재 라운드에 선택된 클라이언트들의 데이터 분포 합집합)에 있는 해당 레이블의 데이터 수로 나누어 비율을 계산
                    # 즉, 현 라운드에서 각 클라이언트의 분포의 기여도 비율을 구하는 과정
                    local_dataset_proportion_per_label[client_id_str][label_str] = (
                        local_dataset_sizes[client_id_str][label_str]
                        / global_dataset_sizes[label_str]
                    )
                # added divisionByZero correction code
                else:
                    local_dataset_proportion_per_label[client_id_str][label_str] = 0.0

        # 클라이언트 별 label 갯수 할당
        client_label_counts = {
            str(client_id): {str(label): 0 for label in range(self.num_classes)}
            for client_id in self.selected_clients
        }
        client_total_counts = {str(client_id): 0 for client_id in self.selected_clients}

        # Determine balancing strategy
        # Backward compatibility: use_data_replication=true → "replication", false → "trimming"
        strategy = getattr(self.config, "balancing_strategy", None)
        if strategy is None:
            # Fallback to legacy option
            strategy = (
                "replication"
                if getattr(self.config, "use_data_replication", False)
                else "trimming"
            )

        added_count = 0
        removed_count = 0

        if strategy == "target":
            # --- Target-based Balancing: Hybrid approach ---
            target_type = getattr(self.config, "balancing_target", "mean")
            sizes = list(global_dataset_sizes.values())

            if target_type == "mean":
                target_size = int(sum(sizes) / len(sizes))
            elif target_type == "median":
                sorted_sizes = sorted(sizes)
                mid = len(sorted_sizes) // 2
                target_size = int(sorted_sizes[mid])
            else:
                target_size = int(target_type)  # Fixed number

            USFLLogger.log_debug(
                f"Target-based balancing: strategy={target_type}, target_size={target_size}"
            )
            print(
                f"[Round {round_number}] Target-based balancing: target={target_size} (strategy={target_type})"
            )

            # Calculate per-label adjustment
            should_add_count = {}
            should_remove_count = {}
            for label in range(self.num_classes):
                label_str = str(label)
                diff = global_dataset_sizes[label_str] - target_size
                if diff > 0:
                    should_remove_count[label_str] = diff
                    should_add_count[label_str] = 0
                else:
                    should_add_count[label_str] = -diff
                    should_remove_count[label_str] = 0

            # Apply adjustments per client
            for client_id in self.selected_clients:
                client_id_str = str(client_id)
                for label in range(self.num_classes):
                    label_str = str(label)
                    label_count = local_dataset_sizes[client_id_str][label_str]

                    if label_count > 0:
                        proportion = local_dataset_proportion_per_label[client_id_str][
                            label_str
                        ]

                        if should_add_count[label_str] > 0:
                            # Replication for this label
                            add_for_client = int(
                                should_add_count[label_str] * proportion
                            )
                            client_label_counts[client_id_str][label_str] = (
                                label_count + add_for_client
                            )
                            added_count += add_for_client
                        elif should_remove_count[label_str] > 0:
                            # Trimming for this label
                            remove_for_client = int(
                                should_remove_count[label_str] * proportion
                            )
                            client_label_counts[client_id_str][label_str] = (
                                label_count - remove_for_client
                            )
                            removed_count += remove_for_client
                        else:
                            client_label_counts[client_id_str][label_str] = label_count

                        client_total_counts[client_id_str] += client_label_counts[
                            client_id_str
                        ][label_str]
                    else:
                        client_label_counts[client_id_str][label_str] = 0

            USFLLogger.log_debug(
                f"Target balancing: added={added_count}, removed={removed_count}"
            )

        elif strategy == "replication":
            # --- Data Replication: Augment to max size ---
            max_dataset_size = max(global_dataset_sizes.values())
            USFLLogger.log_debug(f"Max dataset size: {max_dataset_size}")

            should_add_count = {
                str(label): max_dataset_size - global_dataset_sizes[str(label)]
                for label in range(self.num_classes)
            }
            USFLLogger.log_debug(f"Should add count: {should_add_count}")

            for client_id in self.selected_clients:
                client_id_str = str(client_id)
                for label in range(self.num_classes):
                    label_str = str(label)
                    label_count = local_dataset_sizes[client_id_str][label_str]

                    if label_count > 0 and should_add_count[label_str] > 0:
                        should_add_count_for_client = int(
                            should_add_count[label_str]
                            * local_dataset_proportion_per_label[client_id_str][
                                label_str
                            ]
                        )
                        client_label_counts[client_id_str][label_str] = (
                            local_dataset_sizes[client_id_str][label_str]
                            + should_add_count_for_client
                        )
                        client_total_counts[client_id_str] += client_label_counts[
                            client_id_str
                        ][label_str]

                        added_count += should_add_count_for_client
                    else:
                        client_label_counts[client_id_str][label_str] = label_count
                        client_total_counts[client_id_str] += label_count

            USFLLogger.log_debug(f"Total added count (via replication): {added_count}")
            print(
                f"[Round {round_number}] Data replication: {added_count} samples added (max_size={max_dataset_size})"
            )

        else:  # "trimming" (default)
            # --- Original Trimming: Reduce to min size ---
            min_dataset_size = min(global_dataset_sizes.values())
            USFLLogger.log_debug(f"Min dataset size: {min_dataset_size}")

            should_remove_count = {
                str(label): global_dataset_sizes[str(label)] - min_dataset_size
                for label in range(self.num_classes)
            }
            USFLLogger.log_debug(f"Should remove count: {should_remove_count}")

            for client_id in self.selected_clients:
                client_id_str = str(client_id)
                for label in range(self.num_classes):
                    label_str = str(label)
                    label_count = local_dataset_sizes[client_id_str][label_str]

                    if label_count > 0:
                        should_remove_count_for_client = int(
                            should_remove_count[label_str]
                            * local_dataset_proportion_per_label[client_id_str][
                                label_str
                            ]
                        )
                        client_label_counts[client_id_str][label_str] = (
                            local_dataset_sizes[client_id_str][label_str]
                            - should_remove_count_for_client
                        )
                        client_total_counts[client_id_str] += client_label_counts[
                            client_id_str
                        ][label_str]

                        removed_count += should_remove_count_for_client

                    else:
                        client_label_counts[client_id_str][label_str] = 0

            USFLLogger.log_debug(f"Total removed count: {removed_count}")

        self.client_label_counts = client_label_counts

        payload_extension = {}
        final_local_epochs = self.config.local_epochs

        if self.config.use_dynamic_batch_scheduler:
            # --- New Dynamic Batch Scheduling Logic ---
            ordered_client_ids = sorted(self.selected_clients)
            C_list = [client_total_counts[str(cid)] for cid in ordered_client_ids]

            k, schedule_by_index = create_schedule(self.config.batch_size, C_list)

            schedule_by_client_id = {str(cid): [] for cid in ordered_client_ids}
            for iter_schedule in schedule_by_index:
                for client_idx, batch_size in enumerate(iter_schedule):
                    client_id = ordered_client_ids[client_idx]
                    schedule_by_client_id[str(client_id)].append(batch_size)

            # Log detailed batch schedule to file
            client_data_usage = {
                str(cid): C_list[i] for i, cid in enumerate(ordered_client_ids)
            }
            USFLLogger.log_batch_schedule(
                round_number=round_number,
                client_data_usage=client_data_usage,
                total_iterations=k,
                schedule_by_client=schedule_by_client_id,
            )

            payload_extension = {
                "iterations": k,
                "batch_schedule": schedule_by_client_id,
            }
            # Dynamic scheduler uses the dataset once.
            self.global_dict.set(f"round_{round_number}_effective_epochs", 1)
            # --- End of New Logic ---
        else:
            # --- Original Min_Iterations Logic ---
            global_batch_size, batch_sizes, iterations_per_client = (
                self._calculate_batch_size(client_total_counts, self.selected_clients)
            )

            USFLLogger.log_debug(
                f"Using original min_iterations scheduler: {iterations_per_client}"
            )

            min_iterations = (
                min(iterations_per_client.values()) if iterations_per_client else 0
            )

            if self.config.use_additional_epoch:
                USFLLogger.log_debug("Calculating additional epochs.")
                removed_iteration_per_epoch = (
                    removed_count // global_batch_size if global_batch_size > 0 else 0
                )
                total_removed_iteration = (
                    removed_iteration_per_epoch * self.config.local_epochs
                )
                additional_epoch = 0
                if min_iterations != 0:
                    additional_epoch = total_removed_iteration / min_iterations
                if additional_epoch > 0.0:
                    additional_epoch += 1
                additional_epoch = int(additional_epoch)
                final_local_epochs += additional_epoch
                USFLLogger.log_debug(
                    f"Additional Epoch: {additional_epoch}, Final Epochs: {final_local_epochs}"
                )
            else:
                USFLLogger.log_debug("Not using additional epochs.")

            payload_extension = {
                "batch_sizes": batch_sizes,
                "iterations": min_iterations,
            }
            self.global_dict.set(
                f"round_{round_number}_effective_epochs", final_local_epochs
            )
            # --- End of Original Logic ---

        # 모델 분할 및 모델 큐 설정
        # FlexibleResNet uses pre-built split models (layer boundary support)
        if hasattr(self.model, "get_split_models"):
            client_model, server_model = self.model.get_split_models()
            self.split_models = [client_model, server_model]
        else:
            self.split_models = self.splitter.split(
                self.model.get_torch_model(), self.config.__dict__
            )

        # Drift Measurement: Snapshot client and server models at round start
        if self.drift_tracker is not None:
            self.drift_tracker.on_round_start(self.split_models[0], self.split_models[1])

        model_queue = self.global_dict.get("model_queue")
        model_queue.start_insert_mode()

        # 라운드 시작 및 종료 시간 계산
        self.round_start_time, self.round_end_time = (
            self.pre_round.calculate_round_end_time()
        )

        # 선택된 클라이언트에게 커스터마이즈된 글로벌 모델 전송
        base_payload = {
            "round_number": round_number,
            "round_end_time": self.round_end_time,
            "round_start_time": self.round_start_time,
            "signiture": model_queue.get_signiture(),
            "local_epochs": final_local_epochs,
            "split_count": len(self.split_models),
            "model_index": 0,
            "augmented_dataset_sizes": client_label_counts,
        }
        base_payload.update(payload_extension)

        # === G MEASUREMENT: Compute oracle for diagnostic rounds ===
        if (
            self.g_measurement_system is not None
            and self.g_measurement_system.is_diagnostic_round(round_number)
        ):
            import psutil, os

            process = psutil.Process(os.getpid())
            mem_before_oracle = process.memory_info().rss / (1024**3)
            print(
                f"\n[G Measurement] === Round {round_number}: Computing Oracle === (mem={mem_before_oracle:.2f}GB)"
            )

            # Lazy initialize oracle_calculator if needed
            if self.g_measurement_system.oracle_calculator is None:
                full_trainset = self._dataset.get_trainset()
                oracle_batch_size = self.config.oracle_batch_size
                full_trainloader = DataLoader(
                    dataset=full_trainset,
                    batch_size=oracle_batch_size
                    if oracle_batch_size is not None
                    else self.config.batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                self.g_measurement_system.initialize(full_trainloader)
                print(
                    f"[G Measurement] Oracle calculator initialized with {len(full_trainloader.dataset)} samples"
                )

            # Set param names for gradient splitting
            self.g_measurement_system.set_param_names(
                self.split_models[0],  # client model
                self.split_models[1],  # server model
            )

            if hasattr(self.model, "sync_full_model_from_split"):
                self.model.sync_full_model_from_split()

            # Compute oracle gradient using split models (includes split layer gradient)
            full_model = self.model.get_torch_model().to(self.config.device)
            self.g_measurement_system.compute_oracle_split_for_round(
                self.split_models[0],  # client model
                self.split_models[1],  # server model
                full_model,  # full model for split layer gradient
                split_layer_name=self.config.split_layer,
                config=self.config,
            )

            mem_after_oracle = process.memory_info().rss / (1024**3)
            print(
                f"[G Measurement] Oracle computed (mem={mem_after_oracle:.2f}GB, Δ={mem_after_oracle - mem_before_oracle:+.2f}GB)"
            )

        await self.pre_round.send_customized_global_model(
            self.selected_clients,
            [self.split_models[0] for _ in range(len(self.selected_clients))],
            self.connection,
            base_payload,
        )

    async def _in_round(self, round_number: int):
        # Start round tracking
        TrainingTracker.start_round(round_number)
        iteration_count = 0

        # G Measurement: Start accumulated/k_batch round
        if (
            self.g_measurement_system is not None
            and self.g_measurement_system.is_diagnostic_round(round_number)
            and self.g_measurement_system.measurement_mode in ("accumulated", "k_batch")
        ):
            self.g_measurement_system.start_accumulated_round()

        # SFLV2: Create persistent optimizer and criterion once per round
        server_model = self.split_models[1].to(self.config.device)

        # Server learning rate: explicit value or default to client_lr * num_clients_per_round
        server_lr = self.config.server_learning_rate
        if server_lr is None:
            server_lr = self.config.learning_rate * self.config.num_clients_per_round
        print(
            f"[USFL] Server LR: {server_lr} "
            f"(client_lr={self.config.learning_rate} x {self.config.num_clients_per_round} clients)"
        )

        server_optimizer = self.in_round._get_optimizer(
            server_model, self.config, lr_override=server_lr
        )
        server_criterion = self.in_round._get_criterion(self.config)

        async def __server_side_training():
            nonlocal iteration_count
            while True:
                activations = await self.in_round.wait_for_concatenated_activations(
                    self.selected_clients
                )

                activation_length_per_client = {
                    activation["client_id"]: len(activation["labels"])
                    for activation in activations
                }

                # --- Iteration Tracking ---
                iteration_count += 1
                client_data = {}
                server_total_labels = Counter()

                for act in activations:
                    cid = act["client_id"]
                    labels = act["labels"]
                    labels_list = (
                        labels.tolist() if hasattr(labels, "tolist") else labels
                    )
                    label_counts = Counter(labels_list)

                    client_data[cid] = {
                        "batch_size": len(labels_list),
                        "label_distribution": {
                            str(k): v for k, v in label_counts.items()
                        },
                    }
                    server_total_labels.update(label_counts)

                # Log client-level data
                TrainingTracker.log_iteration_data(
                    round_number, iteration_count, client_data
                )

                # Log server total
                total_samples = sum(len(act["labels"]) for act in activations)
                TrainingTracker.log_server_iteration(
                    round_number,
                    iteration_count,
                    total_samples,
                    {str(k): v for k, v in server_total_labels.items()},
                )
                # --- End Iteration Tracking ---

                # Filter out empty activations (batch_size=0)
                non_empty_activations = [
                    act for act in activations if len(act["labels"]) > 0
                ]

                # Only process if there are non-empty activations
                if non_empty_activations:
                    concatenated_activations = {}
                    for activation in non_empty_activations:
                        concatenated_activations = self._concatenate_activations(
                            concatenated_activations, activation
                        )
                    logits = await self.in_round.forward(
                        self.split_models[1], concatenated_activations
                    )

                    # G Measurement: only collect on first batch of diagnostic rounds
                    is_diagnostic = (
                        self.g_measurement_system is not None
                        and self.g_measurement_system.is_diagnostic_round(round_number)
                    )

                    grad, loss, server_grad = await self.in_round.backward_from_label(
                        self.split_models[1],
                        logits,
                        concatenated_activations,
                        collect_server_grad=is_diagnostic,
                        optimizer=server_optimizer,
                        criterion=server_criterion,
                    )

                    # Drift Measurement: Accumulate server drift after optimizer.step()
                    if self.drift_tracker is not None:
                        self.drift_tracker.accumulate_server_drift(self.split_models[1])

                    # Scale gradient by number of participating clients
                    # CE(mean) on concatenated batch divides by total_batch = N * client_batch
                    # We multiply by N to restore scale to 1/client_batch (same as SFL)
                    num_participating_clients = len(non_empty_activations)
                    if num_participating_clients > 1:
                        if isinstance(grad, tuple):
                            grad = tuple(g * num_participating_clients for g in grad)
                        else:
                            grad = grad * num_participating_clients

                    # G Measurement: Collect server gradient
                    if is_diagnostic and server_grad:
                        batch_weight = sum(
                            len(act["labels"]) for act in non_empty_activations
                        )
                        if self.g_measurement_system.measurement_mode in (
                            "accumulated",
                            "k_batch",
                        ):
                            # Accumulated/K-batch mode: collect (k_batch will stop after K batches internally)
                            self.g_measurement_system.accumulate_server_gradient(
                                server_grad, batch_weight
                            )
                        elif iteration_count == 1:
                            # Single mode: only first batch
                            self.g_measurement_system.store_server_gradient(
                                server_grad, batch_weight
                            )
                            print(
                                f"[G Measurement] Server gradient collected (batch_size={batch_weight})"
                            )

                    # G Measurement: Store split layer gradient ONLY on first batch
                    if (
                        is_diagnostic
                        and grad is not None
                        and self.g_measurement_system.split_g_tilde is None
                    ):
                        if isinstance(grad, tuple):
                            split_grad = tuple(
                                g.clone().detach().cpu() for g in grad if g is not None
                            )
                            split_grad = tuple(
                                g.mean(dim=0) if g.dim() >= 1 else g for g in split_grad
                            )
                            self.g_measurement_system.split_g_tilde = split_grad
                            split_shapes = [g.shape for g in split_grad]
                        else:
                            split_grad = grad.clone().detach().cpu()
                            if split_grad.dim() >= 1:
                                self.g_measurement_system.split_g_tilde = (
                                    split_grad.mean(dim=0)
                                )
                            else:
                                self.g_measurement_system.split_g_tilde = split_grad
                            split_shapes = [
                                self.g_measurement_system.split_g_tilde.shape
                            ]
                        print(
                            f"[G Measurement] Split layer gradient collected, shape: {split_shapes}"
                        )

                    if self.config.gradient_shuffle:
                        # print(f"[Gradient Shuffle] Applying strategy: {self.config.gradient_shuffle_strategy}")
                        target = getattr(self.config, "gradient_shuffle_target", "all")

                        if self.config.gradient_shuffle_strategy == "inplace":
                            # print(f"  → Class-balanced shuffle applied")
                            if isinstance(grad, tuple):
                                if target == "activation_only":
                                    grad = (
                                        self._shuffle_gradients(
                                            non_empty_activations, grad[0]
                                        ),
                                        *grad[1:],
                                    )
                                else:
                                    grad = self._shuffle_gradients(
                                        non_empty_activations, grad
                                    )
                            else:
                                grad = self._shuffle_gradients(
                                    non_empty_activations, grad
                                )
                        elif self.config.gradient_shuffle_strategy == "random":
                            if isinstance(grad, tuple):
                                perm = torch.randperm(grad[0].size(0))
                                if target == "activation_only":
                                    grad = (grad[0][perm], *grad[1:])
                                else:
                                    grad = tuple(g[perm] for g in grad)
                            else:
                                # print(f"  → Random permutation applied (grad shape: {grad.shape})")
                                perm = torch.randperm(grad.size(0))
                                grad = grad[perm]
                        elif self.config.gradient_shuffle_strategy == "average":
                            weight = getattr(
                                self.config, "gradient_average_weight", 0.5
                            )
                            if isinstance(grad, tuple):
                                if target == "activation_only":
                                    mean_grad = grad[0].mean(dim=0, keepdim=True)
                                    grad = (
                                        (1 - weight) * grad[0] + weight * mean_grad,
                                        *grad[1:],
                                    )
                                else:
                                    mixed = []
                                    for g in grad:
                                        mean_grad = g.mean(dim=0, keepdim=True)
                                        mixed.append(
                                            (1 - weight) * g + weight * mean_grad
                                        )
                                    grad = tuple(mixed)
                            else:
                                # print(f"  → Average mixing: {(1-weight)*100:.0f}% original + {weight*100:.0f}% global mean")
                                mean_grad = grad.mean(dim=0, keepdim=True)
                                grad = (1 - weight) * grad + weight * mean_grad
                        elif (
                            self.config.gradient_shuffle_strategy
                            == "average_adaptive_alpha"
                        ):
                            # Adaptive mixing based on cosine similarity
                            # High similarity → keep local, Low similarity → use global
                            beta = getattr(self.config, "adaptive_alpha_beta", 2.0)
                            base_grad = grad[0] if isinstance(grad, tuple) else grad
                            mean_grad = base_grad.mean(
                                dim=0, keepdim=True
                            )  # [1, feature_dim]

                            # Compute cosine similarity for each sample
                            # grad: [num_samples, feature_dim], mean_grad: [1, feature_dim]
                            grad_norm = torch.norm(
                                base_grad, dim=1, keepdim=True
                            )  # [num_samples, 1]
                            mean_norm = torch.norm(
                                mean_grad, dim=1, keepdim=True
                            )  # [1, 1]

                            # Avoid division by zero
                            grad_norm = torch.clamp(grad_norm, min=1e-8)
                            mean_norm = torch.clamp(mean_norm, min=1e-8)

                            # Cosine similarity: (grad · mean_grad) / (||grad|| * ||mean_grad||)
                            dot_product = (base_grad * mean_grad).sum(
                                dim=1, keepdim=True
                            )  # [num_samples, 1]
                            cos_sim = dot_product / (
                                grad_norm * mean_norm
                            )  # [num_samples, 1]

                            # Dynamic alpha using sigmoid
                            # High similarity → high alpha → more local
                            # Low similarity → low alpha → more global
                            alpha_dynamic = torch.sigmoid(
                                beta * cos_sim
                            )  # [num_samples, 1]

                            # Log statistics
                            print(f"  → Adaptive alpha (β={beta}):")
                            print(
                                f"     Cosine similarity: min={cos_sim.min().item():.3f}, mean={cos_sim.mean().item():.3f}, max={cos_sim.max().item():.3f}"
                            )
                            print(
                                f"     Alpha (local weight): min={alpha_dynamic.min().item():.3f}, mean={alpha_dynamic.mean().item():.3f}, max={alpha_dynamic.max().item():.3f}"
                            )

                            # Mix: alpha * local + (1 - alpha) * global
                            if isinstance(grad, tuple):
                                if target == "activation_only":
                                    mean_grad = grad[0].mean(dim=0, keepdim=True)
                                    grad = (
                                        alpha_dynamic * grad[0]
                                        + (1 - alpha_dynamic) * mean_grad,
                                        *grad[1:],
                                    )
                                else:
                                    mixed = []
                                    for g in grad:
                                        mean_g = g.mean(dim=0, keepdim=True)
                                        mixed.append(
                                            alpha_dynamic * g
                                            + (1 - alpha_dynamic) * mean_g
                                        )
                                    grad = tuple(mixed)
                            else:
                                grad = (
                                    alpha_dynamic * grad
                                    + (1 - alpha_dynamic) * mean_grad
                                )
                        else:
                            raise ValueError(
                                f"Unknown gradient shuffle strategy: {self.config.gradient_shuffle_strategy}"
                            )

                    # Distribute gradients to non-empty clients
                    start = 0
                    for act in non_empty_activations:
                        cid = act["client_id"]
                        length = activation_length_per_client[cid]
                        end = start + length

                        if isinstance(grad, tuple):
                            g_slice = tuple(g[start:end].clone() for g in grad)
                        else:
                            g_slice = grad[start:end].clone()
                        await self.in_round.send_gradients(
                            self.connection, g_slice, cid, 0
                        )
                        start = end

                # Send empty gradients to clients that sent empty activations
                for act in activations:
                    cid = act["client_id"]
                    if activation_length_per_client[cid] == 0:
                        empty_grad = torch.tensor([]).to(self.config.device)
                        await self.in_round.send_gradients(
                            self.connection, empty_grad, cid, 0
                        )

        wait_for_models = asyncio.create_task(
            self.in_round.wait_for_model_submission(
                self.selected_clients, self.round_end_time
            )
        )
        server_side_training = asyncio.create_task(__server_side_training())

        done, pending = await asyncio.wait(
            [wait_for_models, server_side_training], return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        await asyncio.gather(*pending, return_exceptions=True)

        # G Measurement: Finalize accumulated/k_batch round
        if (
            self.g_measurement_system is not None
            and self.g_measurement_system.is_diagnostic_round(round_number)
            and self.g_measurement_system.measurement_mode in ("accumulated", "k_batch")
        ):
            self.g_measurement_system.finalize_accumulated_round()

    async def _post_round(self, round_number: int):
        model_queue = self.global_dict.get("model_queue")
        model_queue.end_insert_mode()

        # ===== G MEASUREMENT V2 (Oracle-based) =====
        if self.g_measurement_system is not None:
            import psutil, os, gc, pickle

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024**3)

            # Extract client gradients from model_queue (diagnostic rounds)
            if self.g_measurement_system.is_diagnostic_round(round_number):
                client_grads = {}
                client_weights = {}
                for item in model_queue.queue:
                    if len(item) >= 3:
                        client_id, model, num_samples = item[0], item[1], item[2]
                        # num_samples might be a dict containing client_gradient
                        if (
                            isinstance(num_samples, dict)
                            and "client_gradient" in num_samples
                        ):
                            grad_hex = num_samples["client_gradient"]
                            if isinstance(grad_hex, str):
                                client_grads[client_id] = pickle.loads(
                                    bytes.fromhex(grad_hex)
                                )
                            else:
                                client_grads[client_id] = grad_hex
                            # IMPORTANT: Delete client_gradient after extraction to prevent bloated result JSON
                            del num_samples["client_gradient"]

                        weight = None
                        if isinstance(num_samples, dict):
                            measurement_weight = num_samples.get(
                                "measurement_gradient_weight"
                            )
                            if measurement_weight is not None:
                                weight = measurement_weight
                            else:
                                augmented_counts = num_samples.get(
                                    "augmented_label_counts", {}
                                )
                                if augmented_counts:
                                    weight = sum(augmented_counts.values())
                                else:
                                    weight = num_samples.get("dataset_size", 0)
                        else:
                            weight = num_samples

                        if weight is not None:
                            client_weights[client_id] = float(weight)

                if client_grads:
                    self.g_measurement_system.client_g_tildes = client_grads
                    if client_weights:
                        sorted_sizes = [
                            int(client_weights[cid])
                            for cid in sorted(client_weights.keys())
                        ]
                        print(
                            "[G Measurement] Collected gradients from "
                            f"{len(client_grads)} clients (batch_sizes={sorted_sizes})"
                        )
                    else:
                        print(
                            f"[G Measurement] Collected gradients from {len(client_grads)} clients"
                        )

                # Compute G metrics
                result = self.g_measurement_system.compute_g(
                    round_number, client_weights=client_weights
                )
                if result:
                    self.global_dict.add_event("G_MEASUREMENT", result.to_dict())
                    print(f"[G Measurement] Round {round_number} G computed and logged")

            # Clear gradient data to release memory
            self.g_measurement_system.clear_round_data()
            gc.collect()

            mem_after = process.memory_info().rss / (1024**3)
            print(
                f"[G Measurement] Memory: {mem_before:.2f}GB → {mem_after:.2f}GB (freed {mem_before - mem_after:.2f}GB)"
            )

        # ===== CLEANUP: Always remove client_gradient from model_queue to prevent JSON bloat =====
        for item in model_queue.queue:
            if len(item) >= 3:
                num_samples = item[2]
                if isinstance(num_samples, dict) and "client_gradient" in num_samples:
                    del num_samples["client_gradient"]
        # ===== END G MEASUREMENT =====

        # ===== DRIFT MEASUREMENT: Collect drift metrics from clients =====
        if self.drift_tracker is not None:
            for item in model_queue.queue:
                if len(item) >= 3:
                    client_id, model, num_samples = item[0], item[1], item[2]
                    if isinstance(num_samples, dict):
                        drift_trajectory_sum = num_samples.get("drift_trajectory_sum", 0.0)
                        drift_batch_steps = num_samples.get("drift_batch_steps", 0)
                        drift_endpoint = num_samples.get("drift_endpoint", 0.0)
                        if drift_batch_steps > 0:
                            self.drift_tracker.collect_client_drift(
                                client_id,
                                drift_trajectory_sum,
                                drift_batch_steps,
                                drift_endpoint,
                            )
        # ===== END DRIFT MEASUREMENT COLLECTION =====

        client_ids = [model[0] for model in model_queue.queue]
        self.global_dict.add_event(
            "MODEL_AGGREGATION_START", {"client_ids": client_ids}
        )
        updated_torch_model = self.post_round.aggregate_models(
            self.aggregator, model_queue
        )
        self.global_dict.add_event("MODEL_AGGREGATION_END", {"client_ids": client_ids})

        if isinstance(updated_torch_model, torch.nn.ModuleDict):
            updated_torch_model.update(self.split_models[1])

        if updated_torch_model != None:
            updated_torch_model = self.aggregator.model_reshape(updated_torch_model)
            self.post_round.update_global_model(updated_torch_model, self.model)

        # ===== DRIFT MEASUREMENT: Compute G_drift after aggregation =====
        if self.drift_tracker is not None:
            # Get the new global client and server models after aggregation
            if hasattr(self.model, "get_split_models"):
                new_client_model, new_server_model = self.model.get_split_models()
            else:
                new_client_model = self.split_models[0]
                new_server_model = self.split_models[1]

            drift_result = self.drift_tracker.on_round_end(
                round_number, new_client_model, new_server_model
            )
            if drift_result:
                self.global_dict.add_event("DRIFT_MEASUREMENT", drift_result.to_dict())
        # ===== END DRIFT MEASUREMENT =====

        model_queue.clear()
        accuracy = self.post_round.evaluate_global_model(self.model, self.testloader)
        self.global_dict.add_event("MODEL_EVALUATED", {"accuracy": accuracy})

        # 새로운 이벤트 추가: 각 클라이언트가 실제로 사용한 데이터 개수
        self.global_dict.add_event(
            "CLIENT_DATA_USAGE_PER_ROUND",
            {
                "round_number": round_number,
                "client_label_counts": self.client_label_counts,
            },
        )

        # "시간 감쇠" 기능이 켜져 있으면, 이번 라운드의 실제 사용량을 누적하여 기억을 업데이트
        if getattr(self.config, "use_cumulative_usage", False):
            cumulative_usage = self.global_dict.get("cumulative_usage")

            # Get the effective epochs for this round
            # Use internal dict for .get() with default value support
            effective_epochs = self.global_dict.global_dict.get(
                f"round_{round_number}_effective_epochs", 1
            )

            # Log to file
            USFLLogger.log_cumulative_usage_update(round_number, effective_epochs)

            for client_id_str, class_counts in self.client_label_counts.items():
                if client_id_str not in cumulative_usage:
                    continue  # Should be initialized, but as a safeguard

                for class_label, amount_to_use in class_counts.items():
                    if amount_to_use == 0:
                        continue

                    total_amount_to_use = amount_to_use * effective_epochs

                    usage_bins = cumulative_usage[client_id_str].get(class_label, {})
                    if not usage_bins:
                        continue

                    # Move data from lower usage bins to higher ones (exponential bin structure)
                    # We need to process bins in order and calculate target bins
                    sorted_bins = sorted(usage_bins.keys())

                    # Temporary dict to accumulate changes
                    bin_changes = {}

                    remaining_to_move = total_amount_to_use
                    for current_bin in sorted_bins:
                        if remaining_to_move <= 0:
                            break

                        available_in_bin = usage_bins.get(current_bin, 0)
                        if available_in_bin <= 0:
                            continue

                        can_move = min(remaining_to_move, available_in_bin)

                        # Calculate target bin: current usage count + effective_epochs
                        # For bin 0: moves to bin 1 (or 2, 4, 8... depending on effective_epochs)
                        # For bin 1: usage_count was 1, now becomes 1 + effective_epochs
                        # For bin 2: average usage was 2.5, now becomes ~2.5 + effective_epochs
                        bin_min, bin_max = self._get_bin_range(current_bin)
                        avg_current_usage = (bin_min + bin_max) / 2.0
                        new_usage_count = int(avg_current_usage + effective_epochs)
                        target_bin = self._get_exponential_bin(new_usage_count)

                        # Record changes
                        bin_changes[current_bin] = (
                            bin_changes.get(current_bin, 0) - can_move
                        )
                        bin_changes[target_bin] = (
                            bin_changes.get(target_bin, 0) + can_move
                        )

                        remaining_to_move -= can_move

                    # Apply changes to usage_bins
                    for bin_key, change in bin_changes.items():
                        usage_bins[bin_key] = usage_bins.get(bin_key, 0) + change
                        # Remove bins with 0 data to save memory
                        if usage_bins[bin_key] <= 0:
                            del usage_bins[bin_key]

        # Terminal: Show round completion with accuracy
        print(f"[Round {round_number}] Completed - Accuracy: {accuracy:.4f}")

        if self.config.device == "cuda":
            self._print_memory_usage(f"[Round {round_number} Before]")
            torch.cuda.empty_cache()
            self._print_memory_usage(f"[Round {round_number} After]")
