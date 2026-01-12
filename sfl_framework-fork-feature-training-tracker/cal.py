import math
import numpy as np

def create_proportional_federated_schedule(B: int, C_list: list[int]):
    """
    [방법 1] B와 가장 근사한 값을 만드는 k를 선택하여 스케줄을 생성합니다.
    |sum(ceil(C_i/k)) - B|가 최소가 되는 k를 찾습니다.
    """
    n_clients = len(C_list)
    print("="*60)
    print("방법 1: 'B에 가장 가까운 값'을 만드는 k를 선택합니다.")
    print(f"Initial Data (C_list): {C_list}")
    print(f"Target Constraint Sum (B): {B}\n")

    # --- Phase 1: B에 가장 가까운 sum을 만드는 k 찾기 ---
    best_k = -1
    min_diff = float('inf')
    best_sum_for_k = -1
    
    search_limit = max(C_list) + 3
    for k_candidate in range(2, search_limit):
        current_sum = sum(math.ceil(c / k_candidate) for c in C_list)
        current_diff = abs(current_sum - B)

        # 더 작은 차이를 찾았거나, 차이는 같은데 B를 초과하지 않는 더 나은 해를 찾았을 경우
        if current_diff < min_diff or \
           (current_diff == min_diff and current_sum <= B and best_sum_for_k > B):
            min_diff = current_diff
            best_k = k_candidate
            best_sum_for_k = current_sum

    if best_k == -1:
        print("Error: 스케줄을 생성할 k를 찾을 수 없습니다.")
        return

    print(f"--- Phase 1: 최적의 k 찾기 완료 ---")
    print(f"엄격한 조건(sum == {B})을 만족하는 k는 없었습니다.")
    print(f"대신, B에 가장 가까운 sum 값({best_sum_for_k})을 만드는 k = {best_k}를 선택했습니다.\n")
    k = best_k

    # --- Phase 2: 스케줄 생성 (기존 로직과 동일) ---
    print(f"--- Phase 2: 선택된 k={k}로 스케줄 생성 ---")
    # (이하 스케줄 생성 로직은 원래 함수와 동일합니다)
    remaining_data = C_list.copy()
    cumulative_consumed = [0] * n_clients
    total_data = sum(C_list)
    base_batch_size = total_data // k
    remainder_batches = total_data % k
    target_batch_sizes = [base_batch_size + 1] * remainder_batches + [base_batch_size] * (k - remainder_batches)

    for round_num in range(1, k + 1):
        target_batch_size = target_batch_sizes[round_num - 1]
        total_remaining_data = sum(remaining_data)
        if total_remaining_data == 0: continue

        quotas = [(rd / total_remaining_data) * target_batch_size for rd in remaining_data]
        consumption_this_round = [math.floor(q) for q in quotas]
        reminders_to_distribute = target_batch_size - sum(consumption_this_round)
        fractional_parts = sorted([(i, quotas[i] - consumption_this_round[i]) for i in range(n_clients)], key=lambda x: x[1], reverse=True)

        for i in range(reminders_to_distribute):
            client_index = fractional_parts[i][0]
            consumption_this_round[client_index] += 1

        print(f"--- Round {round_num}/{k} ---")
        print(f"Consumption: {' '.join([f'C{i+1}:{c}' for i, c in enumerate(consumption_this_round)])}")

        for i in range(n_clients):
            consumed = min(consumption_this_round[i], remaining_data[i])
            remaining_data[i] -= consumed
            cumulative_consumed[i] += consumed
    
    print("\n--- 최종 검증 ---")
    print(f"초기 데이터: {C_list}")
    print(f"총 소모량:   {cumulative_consumed}")
    assert C_list == cumulative_consumed, "최종 소모량이 초기 데이터와 일치하지 않습니다."
    print("✅ 성공: 스케줄이 성공적으로 생성되었습니다.\n")
# --- Example Usage from the problem description ---
if __name__ == '__main__':
    # B: Max total batch size for the constraint sum
    # C_list: Initial data amount for each client
    B_example = 256
    C_list_example = [1204, 11235, 220, 1225, 10, 635, 940]
    
    create_proportional_federated_schedule(B_example, C_list_example)