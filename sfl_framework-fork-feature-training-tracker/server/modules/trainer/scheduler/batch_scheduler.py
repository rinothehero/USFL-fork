import math
from typing import List, Tuple

from utils.log_utils import vprint


def create_schedule(B: int, C_list: list[int]) -> Tuple[int, List[List[int]]]:
    """
    서버의 목표 배치 크기(B)와 각 클라이언트의 데이터 양(C_list)을 기반으로,
    모든 클라이언트가 데이터를 모두 소진하면서 서버의 배치 크기를 최대한 일정하게 유지하는
    최적의 iteration 수(k)와 각 iteration별 클라이언트 배치 크기 스케줄을 생성합니다.

    Args:
        B (int): 서버의 목표 배치 크기 (config.batch_size).
        C_list (list[int]): 이번 라운드에 각 클라이언트가 사용할 데이터 양의 리스트.

    Returns:
        Tuple[int, List[List[int]]]:
            - k (int): 최적화된 총 iteration 수.
            - schedule (List[List[int]]): iteration별, 클라이언트별 배치 크기 스케줄.
                                           (e.g., schedule[iteration][client_index])
    """
    n_clients = len(C_list)
    if sum(C_list) == 0:
        return 0, []

    # --- Phase 1: B에 가장 가까운 sum을 만드는 k 찾기 ---
    best_k = -1
    min_diff = float("inf")
    best_sum_for_k = -1

    # k의 탐색 범위를 합리적으로 제한합니다.
    # 최소 k는 1, 최대 k는 모든 데이터 양의 합입니다.
    # 현실적으로 k는 max(C_list)를 넘는 경우가 거의 없습니다.
    search_limit = max(max(C_list), 1) + 3
    for k_candidate in range(1, search_limit):
        current_sum = sum(math.ceil(c / k_candidate) for c in C_list)
        current_diff = abs(current_sum - B)

        if current_diff < min_diff:
            min_diff = current_diff
            best_k = k_candidate
            best_sum_for_k = current_sum
        elif current_diff == min_diff:
            # 차이가 같다면, B를 초과하는 것보다 작은 것을 선호
            if current_sum < best_sum_for_k:
                best_k = k_candidate
                best_sum_for_k = current_sum

    if best_k == -1:
        # 만약 C_list가 모두 0인 경우 등 엣지 케이스 처리
        if sum(C_list) > 0:
             best_k = 1 # 최소 1번의 iteration은 수행
        else:
             return 0, []


    k = best_k

    # --- Phase 2: 스케줄 생성 ---
    schedule = []
    remaining_data = C_list.copy()
    cumulative_consumed = [0] * n_clients
    total_data = sum(C_list)
    
    # 목표 배치 크기를 최대한 균등하게 분배
    base_batch_size = total_data // k
    remainder_batches = total_data % k
    target_batch_sizes = [base_batch_size + 1] * remainder_batches + [
        base_batch_size
    ] * (k - remainder_batches)

    for round_num in range(1, k + 1):
        target_batch_size = target_batch_sizes[round_num - 1]
        total_remaining_data = sum(remaining_data)
        if total_remaining_data == 0:
            # 데이터가 모두 소진되었으면 빈 스케줄 추가
            schedule.append([0] * n_clients)
            continue

        # 남은 데이터 양에 비례하여 이번 라운드의 할당량 계산
        quotas = [
            (rd / total_remaining_data) * target_batch_size for rd in remaining_data
        ]
        consumption_this_round = [min(math.floor(q), rd) for q, rd in zip(quotas, remaining_data)]
        
        # 할당하고 남은 양을 소수부가 큰 순서대로 분배
        reminders_to_distribute = target_batch_size - sum(consumption_this_round)
        fractional_parts = sorted(
            [
                (i, quotas[i] - consumption_this_round[i])
                for i in range(n_clients)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for i in range(reminders_to_distribute):
            client_index = fractional_parts[i][0]
            # 클라이언트가 가진 데이터 이상으로 할당할 수 없음
            if remaining_data[client_index] > consumption_this_round[client_index]:
                consumption_this_round[client_index] += 1
        
        schedule.append(consumption_this_round)

        for i in range(n_clients):
            consumed = min(consumption_this_round[i], remaining_data[i])
            remaining_data[i] -= consumed
            cumulative_consumed[i] += consumed

    # 최종 검증
    if C_list != cumulative_consumed:
        # 스케줄링 오류가 발생할 경우 경고
        vprint(f"Warning: Schedule consumption mismatch!", 0)
        vprint(f"Initial Data: {C_list}", 2)
        vprint(f"Total Consumed: {cumulative_consumed}", 2)


    return k, schedule
