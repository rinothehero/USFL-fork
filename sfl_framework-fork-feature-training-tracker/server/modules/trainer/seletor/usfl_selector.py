import math
import random
from collections import deque
from typing import TYPE_CHECKING, Dict, List

from .base_selector import BaseSelector
from ..utils.usfl_logger import USFLLogger

if TYPE_CHECKING:
    from server_args import Config


class USFLSelector(BaseSelector):
    def __init__(self, config: "Config"):
        self.config = config
        self.selected_counts = {}
        self.client_selected_counts: Dict[int, int] = {}
        self.selection_history = deque(
            maxlen=self.config.num_clients // self.config.num_clients_per_round
        )
        self.freshness_decay_rate = getattr(self.config, "freshness_decay_rate", 0.5)
        self._trimming_cache = {}  # Cache for trimming simulations (performance optimization)

    def _calculate_batch_size(
        self, dataset_sizes: Dict[int, int], selected_clients: List[int]
    ) -> (Dict[int, int], Dict[int, int]):
        total_dataset_size = sum(dataset_sizes.values())
        batch_size = (
            int(
                self.config.batch_size
                * (len(selected_clients) / self.config.num_clients_per_round)
            )
            if len(selected_clients) >= self.config.num_clients_per_round
            else self.config.batch_size
        )

        batch_sizes = {
            client_id: int(batch_size * dataset_size / total_dataset_size)
            for client_id, dataset_size in dataset_sizes.items()
        }

        iterations_per_client = {
            client_id: (
                (dataset_sizes[client_id] // batch_sizes[client_id])
                if batch_sizes[client_id] != 0
                else 0
            )
            for client_id in batch_sizes
        }

        return batch_size, batch_sizes, iterations_per_client

    def kl_divergence(self, counts: Dict[str, int], num_classes: int) -> float:
        total = sum(counts.values())
        if total == 0:
            return float("inf")
        kl = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                kl += p * math.log(p * num_classes)
        return kl

    def min_max_scale(self, val, min_val, max_val):
        if max_val == min_val:
            return 0.0
        return (val - min_val) / (max_val - min_val)

    def _simulate_trimming(
        self,
        client_group: List[int],
        client_informations: Dict,
        num_classes: int,
    ) -> Dict[str, Dict[str, int]]:
        """
        주어진 클라이언트 그룹에 대해 USFL의 트리밍 로직을 시뮬레이션하여,
        각 클라이언트가 실제로 사용하게 될 데이터 양을 예측합니다.
        """
        if not client_group:
            return {}

        # 1. 그룹의 합집합 데이터 분포 계산
        global_dataset_sizes = {str(label): 0 for label in range(num_classes)}
        local_dataset_sizes = {}
        for cid in client_group:
            cid_str = str(cid)
            local_dataset_sizes[cid_str] = {}
            ld = client_informations[cid]["dataset"]["label_distribution"]
            for label, count in ld.items():
                global_dataset_sizes[label] += count
                local_dataset_sizes[cid_str][label] = count

        # 2. 트리밍 기준(min_dataset_size) 계산
        min_dataset_size = min(global_dataset_sizes.values())
        should_remove_count = {
            lbl: global_dataset_sizes[lbl] - min_dataset_size
            for lbl in global_dataset_sizes
        }

        # 3. 각 클라이언트별 기여도에 따라 제거할 양 계산
        predicted_label_counts = {str(cid): {} for cid in client_group}
        for cid in client_group:
            cid_str = str(cid)
            for label in range(num_classes):
                label_str = str(label)
                client_label_count = local_dataset_sizes[cid_str].get(label_str, 0)
                
                if client_label_count > 0 and global_dataset_sizes[label_str] > 0:
                    proportion = client_label_count / global_dataset_sizes[label_str]
                    should_remove = int(should_remove_count[label_str] * proportion)
                    predicted_label_counts[cid_str][label_str] = client_label_count - should_remove
                else:
                    predicted_label_counts[cid_str][label_str] = 0

        return predicted_label_counts

    def _simulate_trimming_incremental(
        self,
        base_group: List[int],
        new_candidate: int,
        client_informations: Dict,
        num_classes: int,
        base_result: Dict = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Incrementally simulate trimming by adding one client to existing group.
        Much faster than recalculating from scratch (~10x performance improvement).

        Args:
            base_group: Currently selected clients
            new_candidate: New client to add
            client_informations: Client data
            num_classes: Number of classes
            base_result: Previous trimming result to build upon (if available)

        Returns:
            Predicted label counts after trimming with new candidate added
        """
        full_group = base_group + [new_candidate]
        cache_key = tuple(sorted(full_group))

        # Check cache first
        if cache_key in self._trimming_cache:
            return self._trimming_cache[cache_key]

        # If base_result provided, compute incrementally
        if base_result and base_group:
            result = self._compute_incremental_trimming(
                base_result, base_group, new_candidate, client_informations, num_classes
            )
        else:
            # Full computation for first candidate or when no base available
            result = self._simulate_trimming(full_group, client_informations, num_classes)

        # Cache result (limit cache size to prevent memory bloat)
        if len(self._trimming_cache) > 100:
            self._trimming_cache.clear()
        self._trimming_cache[cache_key] = result

        return result

    def _compute_incremental_trimming(
        self,
        base_result: Dict[str, Dict[str, int]],
        base_group: List[int],
        new_candidate: int,
        client_informations: Dict,
        num_classes: int,
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute trimming result by updating base result with new candidate.
        Avoids recalculating everything from scratch.
        """
        # Recalculate global sizes with new candidate
        global_sizes = {str(label): 0 for label in range(num_classes)}

        # Sum existing clients from base_result
        for cid_str in base_result.keys():
            cid = int(cid_str)
            ld = client_informations[cid]["dataset"]["label_distribution"]
            for label_str, count in ld.items():
                global_sizes[label_str] += count

        # Add new candidate
        new_cid_str = str(new_candidate)
        new_ld = client_informations[new_candidate]["dataset"]["label_distribution"]
        for label_str, count in new_ld.items():
            global_sizes[label_str] += count

        # Recompute trimming with updated global sizes
        min_size = min(global_sizes.values())
        should_remove = {lbl: global_sizes[lbl] - min_size for lbl in global_sizes}

        # Calculate new predicted counts for all clients (including new one)
        result = {}
        all_clients = base_group + [new_candidate]

        for cid in all_clients:
            cid_str = str(cid)
            result[cid_str] = {}
            ld = client_informations[cid]["dataset"]["label_distribution"]

            for label in range(num_classes):
                label_str = str(label)
                local_count = ld.get(label_str, 0)

                if local_count > 0 and global_sizes[label_str] > 0:
                    proportion = local_count / global_sizes[label_str]
                    to_remove = int(should_remove[label_str] * proportion)
                    result[cid_str][label_str] = local_count - to_remove
                else:
                    result[cid_str][label_str] = 0

        return result

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

    def _calculate_freshness_score(
        self,
        predicted_counts: Dict[str, int],
        usage_history: Dict[str, Dict[int, int]],  # Changed from List to Dict
    ) -> float:
        """
        예측된 데이터 사용량과 상세한 누적 사용량 이력을 바탕으로 "신선도 점수"를 계산합니다.
        과거에 덜 사용된 데이터일수록 높은 가중치를 부여합니다.

        NOTE: Now uses exponential bins (Dict) instead of linear list for memory efficiency.
        """
        score = 0.0
        decay_rate = self.freshness_decay_rate

        for label, amount_to_use in predicted_counts.items():
            if amount_to_use == 0:
                continue

            usage_bins = usage_history.get(label, {})
            if not usage_bins:
                continue

            # Iterate through bins in ascending order
            sorted_bins = sorted(usage_bins.keys())
            for bin_key in sorted_bins:
                if amount_to_use <= 0:
                    break

                available_in_bin = usage_bins[bin_key]
                if available_in_bin <= 0:
                    continue

                can_use = min(amount_to_use, available_in_bin)

                # Weight based on bin's average usage count
                bin_min, bin_max = self._get_bin_range(bin_key)
                avg_usage_count = (bin_min + bin_max) / 2.0
                weight = decay_rate ** avg_usage_count

                score += can_use * weight
                amount_to_use -= can_use

        return score

    def select(
        self,
        n: int,
        client_ids: List[int],
        data=None,
        alpha: float = 0.5,
        max_retry: int = 1000,
        top_n: int = 3,
    ) -> List[int]:
        client_informations = data["client_informations"]
        num_classes = data["num_classes"]
        cumulative_usage = data.get("cumulative_usage") or {}  # Ensure it's a dict
        fresh_scoring = data.get("fresh_scoring", False)

        # Safety check: Freshness scoring requires cumulative usage data
        if fresh_scoring and not cumulative_usage:
            warning_msg = (
                "Fresh scoring enabled but cumulative usage is empty! "
                "This should have been caught by config validation. "
                "Falling back to alpha-based scoring for this round."
            )
            print(f"[WARNING] {warning_msg}")  # Keep in terminal as it's a critical warning
            USFLLogger.log_warning(warning_msg)
            fresh_scoring = False
            data["fresh_scoring"] = False  # Update for this round

        valid_client_ids = []
        for cid in client_ids:
            ld = client_informations[cid]["dataset"]["label_distribution"]
            dataset_size = sum(ld.values())
            if dataset_size >= num_classes:
                valid_client_ids.append(cid)

        if len(valid_client_ids) < n:
            raise RuntimeError(
                f"충분한 규모의 데이터(num_classes={num_classes})를 가진 클라이언트가 없습니다: "
                f"유효 클라이언트 수={len(valid_client_ids)}, 요구 수={n}"
            )

        client_ids = valid_client_ids

        if not self.selected_counts:
            self.selected_counts = {str(lbl): 0 for lbl in range(num_classes)}
            self.client_selected_counts = {cid: 0 for cid in client_ids}

        for retry_idx in range(max_retry):
            aggregated_counts = {str(lbl): 0 for lbl in range(num_classes)}
            selected_dataset_sizes: Dict[int, int] = {}
            selected_ids: List[int] = []

            candidates = client_ids.copy()
            random.shuffle(candidates)
            if not candidates:
                raise RuntimeError("후보 클라이언트가 없습니다.")

            first = candidates.pop(0)
            selected_ids.append(first)
            first_ld = client_informations[first]["dataset"]["label_distribution"]
            selected_dataset_sizes[first] = sum(first_ld.values())
            for lbl, cnt in first_ld.items():
                aggregated_counts[lbl] += cnt

            missing_labels = True
            min_dataset_under_batch_size = True

            while candidates and (
                len(selected_ids) < n or missing_labels or min_dataset_under_batch_size
            ):
                best_cand = None

                if fresh_scoring:
                    # --- Improved "Missing Labels + Temp_min + Freshness" Selection Logic ---

                    # Calculate current missing labels
                    current_missing = sum(1 for v in aggregated_counts.values() if v == 0)

                    # Log current state
                    print(f"\n[Selection Step {len(selected_ids) + 1}/{n}] Current missing labels: {current_missing}")
                    if current_missing > 0:
                        missing_labels_list = [lbl for lbl, cnt in aggregated_counts.items() if cnt == 0]
                        print(f"  Missing label IDs: {missing_labels_list}")

                    # 1. Calculate both missing_labels and temp_min for all candidates
                    candidate_scores = []
                    for cand in candidates:
                        temp = aggregated_counts.copy()
                        ld = client_informations[cand]["dataset"]["label_distribution"]
                        for lbl, cnt in ld.items():
                            temp[lbl] += cnt

                        temp_min = min(temp.values())
                        missing_after = sum(1 for v in temp.values() if v == 0)

                        candidate_scores.append({
                            "id": cand,
                            "missing_after": missing_after,  # Primary: fewer missing is better
                            "temp_min": temp_min              # Secondary: higher min is better
                        })

                    # 2. Phase 1 Filter: Sort by missing_labels first, then temp_min
                    # Lexicographic ordering: (missing_after ASC, temp_min DESC)
                    candidate_scores.sort(key=lambda x: (x["missing_after"], -x["temp_min"]))

                    # Log top candidates after sorting
                    print(f"  Top 5 candidates by (missing_after, temp_min):")
                    for i, cs in enumerate(candidate_scores[:5]):
                        reduction = current_missing - cs["missing_after"]
                        print(f"    {i+1}. Client {cs['id']}: missing_after={cs['missing_after']} (reduces {reduction}), temp_min={cs['temp_min']}")

                    # 3. Phase 2 Filter: Get top-N candidates (by the combined criteria)
                    # Take candidates with best (missing_after, temp_min) combinations
                    primary_candidates = [cs["id"] for cs in candidate_scores[:top_n]]
                    print(f"  Primary candidates (top-{top_n}): {primary_candidates}")

                    # 4. Phase 3: Freshness scoring among primary candidates
                    # "신선도" 점수 기반 최종 선택
                    best_cand = -1
                    max_freshness = -1.0
                    freshness_scores_dict = {}

                    # Compute base trimming result once for incremental updates
                    base_trimming = None
                    if selected_ids:
                        base_trimming = self._simulate_trimming(selected_ids, client_informations, num_classes)

                    for cand_id in primary_candidates:
                        # 가상 선택 및 가상 트리밍 수행 (Incremental computation for performance)
                        predicted_counts = self._simulate_trimming_incremental(
                            selected_ids, cand_id, client_informations, num_classes, base_result=base_trimming
                        )

                        # 후보 클라이언트의 예측된 사용량과 이력으로 신선도 점수 계산
                        cand_predicted_counts = predicted_counts.get(str(cand_id), {})
                        cand_usage_history = cumulative_usage.get(str(cand_id), {})
                        freshness_score = self._calculate_freshness_score(cand_predicted_counts, cand_usage_history)
                        freshness_scores_dict[cand_id] = freshness_score

                        if freshness_score > max_freshness:
                            max_freshness = freshness_score
                            best_cand = cand_id

                    if best_cand == -1 and primary_candidates:
                        best_cand = primary_candidates[0]

                    # Log detailed freshness selection to file
                    USFLLogger.log_freshness_selection(
                        round_number=data.get("round_number", 0),
                        selection_step=len(selected_ids) + 1,
                        total_selections=n,
                        candidate_scores=candidate_scores,
                        primary_candidates=primary_candidates,
                        freshness_scores=freshness_scores_dict,
                        selected_client=best_cand,
                    )
                
                else:
                    # --- Original Alpha-based Scoring Logic ---
                    scores = []
                    for cand in candidates:
                        client_selected_before = sum(
                            1
                            for past_selected in self.selection_history
                            if cand in past_selected
                        )

                        temp = aggregated_counts.copy()
                        ld = client_informations[cand]["dataset"]["label_distribution"]

                        distribution_to_use = ld
                        if cumulative_usage:
                            # 기존 로직은 상세 이력을 사용하지 않으므로, 가상 분포를 단순화하여 계산
                            virtual_distribution = ld.copy()
                            cand_usage = cumulative_usage.get(str(cand), {})
                            for label, usage_list in cand_usage.items():
                                # 0번 사용된 데이터만 남긴다고 가정
                                virtual_distribution[label] = usage_list[0] if usage_list else 0
                            distribution_to_use = virtual_distribution

                        for lbl, cnt in distribution_to_use.items():
                            temp[lbl] += cnt

                        temp_min = min(temp.values()) if temp else 0
                        temp_kl = self.kl_divergence(temp, num_classes)

                        global_counts = {
                            lbl: self.selected_counts[lbl] + temp[lbl] for lbl in temp
                        }
                        global_kl = self.kl_divergence(global_counts, num_classes)

                        scores.append(
                            (cand, temp_min, temp_kl, global_kl, client_selected_before + 1)
                        )

                    if not scores:
                        break

                    mins = [s[1] for s in scores]
                    csb = [s[4] for s in scores]
                    min_min, max_min = min(mins), max(mins)
                    min_csb, max_csb = min(csb), max(csb)

                    best_score = -float("inf")
                    for cand, tmin, _, _, tcsb in scores:
                        s_min = self.min_max_scale(tmin, min_min, max_min)
                        s_csb = self.min_max_scale(tcsb, min_csb, max_csb)

                        score = alpha * s_min + (1 - alpha) * (1 - s_csb)
                        if score > best_score:
                            best_score, best_cand = score, cand

                if best_cand is None:
                    break

                candidates.remove(best_cand)
                selected_ids.append(best_cand)

                # Log the final selection
                if fresh_scoring and len(candidate_scores) > 0:
                    selected_score = next((cs for cs in candidate_scores if cs["id"] == best_cand), None)
                    if selected_score:
                        freshness = freshness_scores_dict.get(best_cand, "N/A")
                        freshness_str = f"{freshness:.2f}" if isinstance(freshness, float) else str(freshness)
                        print(f"  ✓ Selected: Client {best_cand} (missing_after={selected_score['missing_after']}, temp_min={selected_score['temp_min']}, freshness={freshness_str})")
                else:
                    print(f"  ✓ Selected: Client {best_cand}")

                ld = client_informations[best_cand]["dataset"]["label_distribution"]
                selected_dataset_sizes[best_cand] = sum(ld.values())
                for lbl, cnt in ld.items():
                    aggregated_counts[lbl] += cnt

                g_bs, batch_sizes, _ = self._calculate_batch_size(
                    selected_dataset_sizes, selected_ids
                )
                if self.config.batch_size > 64 :
                    for cid in selected_ids.copy():
                        if batch_sizes.get(cid, 0) < num_classes:
                            selected_ids.remove(cid)
                            for lbl, cnt in client_informations[cid]["dataset"][
                                "label_distribution"
                            ].items():
                                aggregated_counts[lbl] -= cnt
                            del selected_dataset_sizes[cid]

                missing_labels = any(c == 0 for c in aggregated_counts.values())
                min_size = min(aggregated_counts.values()) if aggregated_counts else 0
                min_criteria = max(1, int(g_bs / num_classes)) if g_bs > 0 else 1
                min_dataset_under_batch_size = not (
                    len(selected_ids) >= n and min_size >= min_criteria
                )

            over_num_classes = len(selected_ids) > num_classes
            
            if (
                len(selected_ids) >= n
                and not missing_labels
                and not min_dataset_under_batch_size
                #and not over_num_classes
            ):
                # Log final selection summary
                final_missing = sum(1 for v in aggregated_counts.values() if v == 0)
                print(f"\n✓ Selection Complete: {len(selected_ids)} clients selected (target: {n})")
                print(f"  Final missing labels: {final_missing}")
                print(f"  Selected clients: {selected_ids}")

                self.selection_history.append(list(selected_ids))

                for lbl in range(num_classes):
                    self.selected_counts[str(lbl)] += aggregated_counts[str(lbl)]

                for cid in selected_ids:
                    self.client_selected_counts[cid] += 1

                return selected_ids
            else:
                # Log retry attempts to file instead of cluttering terminal
                USFLLogger.log_debug(
                    f"Selection retry {retry_idx + 1}/{max_retry}: "
                    f"missing_labels={missing_labels}, "
                    f"min_dataset_under_batch_size={min_dataset_under_batch_size}, "
                    f"over_num_classes={over_num_classes}"
                )

        raise RuntimeError("USFLSelector: failed to select clients.")
