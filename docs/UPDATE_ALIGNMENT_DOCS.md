# Update Alignment Metric (A_cos + M_norm) — 구현 문서

## 1. 개요

### 배경: 왜 새로운 메트릭이 필요한가

기존 `G_drift` 메트릭은 기법 간 1:1 정합 비교가 구조적으로 불가능합니다:

| 문제 | 영향 |
|------|------|
| **Step 정의 차이** | SFL은 activation 1개마다 서버 step, USFL은 concat 후 1회, GAS는 버퍼 full일 때만 → 라운드당 서버 step 수가 10배 이상 차이 |
| **Gradient scale 차이** | `CE(mean)` + concat → USFL은 `1/(N·B)` 스케일, SFL은 `1/B` → 같은 lr로도 effective update가 N배 차이 |
| **Server LR 정책 차이** | USFL은 `lr × N`, GAS는 `lr`, MultiSFL은 별도 `lr_server` |
| **데이터량 차이** | USFL은 trimming/replication, MultiSFL은 고정 local_steps, GAS는 drop_last 설정이 다름 |
| **Aggregation 규칙 차이** | FedAvg(가중) vs 균등(1/N) vs soft-pull(α 블렌딩) |

이 모든 차이가 `G_drift = S/B`와 `G_end = ||Δ||²` 값에 직접 영향을 미치므로, 기법 A와 B의 G_drift 숫자를 직접 비교하는 것은 의미가 없습니다.

### 해결: Scale-invariant한 방향 정렬 메트릭

**A_cos** (Update Alignment Cosine)는 클라이언트 업데이트 벡터들의 **pairwise cosine similarity 평균**입니다:

- LR, batch size, step 정의에 **무관** (방향만 비교)
- 높을수록 클라이언트들이 **같은 방향으로** 업데이트 → Non-IID divergence가 적음
- **M_norm** (평균 업데이트 크기)과 함께 보면 완전한 그림

| 상태 | A_cos | M_norm | 해석 |
|------|-------|--------|------|
| 건강한 학습 | 높음 (>0.5) | 적절 | 클라이언트들이 합의하며 유의미하게 학습 |
| Non-IID divergence | 낮음 (<0.2) | 높음 | 클라이언트들이 서로 다른 방향으로 크게 업데이트 |
| 학습 정체 | 높음 | ≈0 | 방향은 맞지만 업데이트가 거의 없음 |
| 초기 혼란 | 낮음 | 낮음 | 아직 수렴 방향을 못 찾음 |

---

## 2. 수학적 정의

### 입력

- `θ_start`: 라운드 시작 시 글로벌 모델 파라미터 (trainable params only, BN 버퍼 제외)
- `θ_i_end`: 클라이언트 `i`의 로컬 학습 후 모델 파라미터
- `Δ_i = flatten(θ_i_end - θ_start)`: 클라이언트 `i`의 업데이트 벡터 (1D)

### A_cos (Pairwise Cosine Alignment)

```
A_cos = Σ_{i<j} w_{ij} · cos(Δ_i, Δ_j) / Σ_{i<j} w_{ij}
```

- `cos(a, b) = a·b / (||a|| · ||b||)`
- `w_{ij} = w_i · w_j` (aggregation weight 기반) 또는 `1.0` (uniform)
- `||Δ_i|| < τ` (기본 `1e-7`)인 클라이언트는 A_cos 계산에서 제외

### M_norm (Mean Update Magnitude)

```
M_norm = Σ_i w_i · ||Δ_i|| / Σ_i w_i
```

- 모든 클라이언트 포함 (τ 필터 미적용)
- A_cos의 크기 정보 보완용

### 범위

- `A_cos ∈ [-1, 1]`: 1 = 완전 정렬, 0 = 직교, -1 = 반대 방향
- `M_norm ≥ 0`: 0 = 업데이트 없음
- 유효 클라이언트 < 2일 경우: `A_cos = NaN`

---

## 3. 파일 구조

### 신규 생성

```
USFL-fork/
├── shared/
│   ├── __init__.py                    # 빈 파일
│   └── update_alignment.py            # 핵심 연산 모듈 (169줄)
```

### 수정된 파일

```
sfl_framework-fork-feature-training-tracker/
├── server/
│   ├── utils/
│   │   └── drift_measurement.py       # DriftMetrics + collect_client_model() 추가
│   └── modules/trainer/stage/
│       ├── sfl_stage_organizer.py      # _post_round에 collect_client_model() 호출 추가
│       └── usfl_stage_organizer.py     # _post_round에 collect_client_model() 호출 추가

GAS_implementation/
├── utils/
│   └── drift_measurement.py           # DriftMetrics + collect_client_delta() 추가
├── GAS_main.py                        # finalize_client 후 collect_client_delta() 호출 추가

multisfl_implementation/
├── multisfl/
│   └── drift_measurement.py           # DriftMetrics + collect_branch_delta() + collect_branch_server_delta()
│   └── trainer.py                     # finalize_branch 후 양쪽 delta 수집 호출 추가
```

---

## 4. 핵심 모듈: `shared/update_alignment.py`

### API

#### `is_trainable_param(name: str) -> bool`

BatchNorm 버퍼(`running_mean`, `running_var`, `num_batches_tracked`)를 제외하는 필터.

#### `flatten_delta(end_state_dict, start_params) -> Optional[torch.Tensor]`

```python
flatten_delta(
    end_state_dict: dict,       # 클라이언트 학습 후 state_dict
    start_params: Dict[str, torch.Tensor],  # 라운드 시작 글로벌 파라미터 (CPU)
) -> Optional[torch.Tensor]     # 1D 차이 벡터, 또는 None
```

- `start_params`의 키 순서를 기준으로 매칭 (일관된 벡터 순서 보장)
- trainable params만 포함
- 결과는 항상 CPU, float32
- 매칭 파라미터가 없으면 `None` 반환

#### `compute_update_alignment(deltas, weights, tau, record_pairs) -> AlignmentResult`

```python
compute_update_alignment(
    deltas: List[Tuple[int, torch.Tensor]],  # [(client_id, flattened_delta), ...]
    weights: Optional[Dict[int, float]],     # {client_id: aggregation_weight}
    tau: float = 1e-7,                       # ||Δ|| 최소 threshold
    record_pairs: bool = False,              # 디버깅용 per-pair 기록
) -> AlignmentResult
```

#### `AlignmentResult` dataclass

```python
@dataclass
class AlignmentResult:
    A_cos: float         # Weighted mean pairwise cosine similarity
    M_norm: float        # Weighted mean update norm
    n_valid: int         # ||Δ|| > τ인 유효 클라이언트 수
    n_total: int         # 전체 클라이언트 수
    n_pairs: int         # 계산된 pair 수 = C(n_valid, 2)
    per_pair: List[dict] # record_pairs=True 시 [{i, j, cos, w}, ...]
```

---

## 5. 트랙별 통합 상세

### 5.1 SFL/USFL Unified Framework

**Drift Tracker 수정** (`server/utils/drift_measurement.py`):

| 추가 항목 | 위치 | 설명 |
|----------|------|------|
| `DriftMetrics.A_cos` | dataclass 필드 | `float = float("nan")` |
| `DriftMetrics.M_norm` | dataclass 필드 | `float = 0.0` |
| `DriftMetrics.n_valid_alignment` | dataclass 필드 | `int = 0` |
| `_client_deltas` | `__init__` | `List[Tuple[int, torch.Tensor]]` — Δ_i 벡터 저장 |
| `_alignment_weights` | `__init__` | `Dict[int, float]` — aggregation weight 저장 |
| `collect_client_model()` | 새 메서드 | `flatten_delta()` 호출 후 `_client_deltas`에 추가 |

**호출 지점:**

SFL (`sfl_stage_organizer.py:~502`):
```python
# _post_round에서 model_queue 순회 중, collect_client_drift() 직후:
if hasattr(model, 'state_dict'):
    self.drift_tracker.collect_client_model(client_id, model.state_dict(), client_weight=client_weight)
elif isinstance(model, dict):
    self.drift_tracker.collect_client_model(client_id, model, client_weight=client_weight)
```

USFL (`usfl_stage_organizer.py:~1402`) — 동일 패턴.

**데이터 흐름:**
```
_pre_round: on_round_start(client_model, server_model) → θ_start 저장, _client_deltas 초기화
_in_round:  (학습 진행, 기존 drift 누적)
_post_round: model_queue에서 각 client model 수신
             → collect_client_drift(S, B, E)           # 기존 scalar drift
             → collect_client_model(state_dict, weight)  # 신규 A_cos용 Δ_i
             → on_round_end() → compute_update_alignment() → DriftMetrics에 A_cos 포함
```

**가중치:** `augmented_label_counts` 합계 (USFL의 aggregation weight과 동일).

### 5.2 GAS

**Drift Tracker 수정** (`GAS_implementation/utils/drift_measurement.py`):

| 추가 항목 | 설명 |
|----------|------|
| `DriftMetrics.A_cos/M_norm/n_valid_alignment` | SFL과 동일 |
| `_client_deltas` | `List[Tuple[int, torch.Tensor]]` |
| `collect_client_delta(client_id, state_dict)` | flatten_delta 후 저장 |

**호출 지점** (`GAS_main.py:~1207`):
```python
# finalize_client() 직후, 아직 sumClientParam에 합산 전:
drift_tracker.collect_client_delta(
    selected_client,
    usersParam[np.where(order == selected_client)[0][0]]
)
```

**데이터 흐름:**
```
epoch 시작: on_round_start(userParam, server_model.state_dict()) → θ_start 저장
각 클라 step: accumulate_client_drift() / accumulate_server_drift()
클라 교체 시: finalize_client() → collect_client_delta()  # Δ_i 수집
전원 교체 시: on_round_end() → compute_update_alignment() → A_cos
```

**가중치:** Uniform (GAS는 균등 `1/user_parti_num` aggregation).

### 5.3 MultiSFL

**Drift Tracker 수정** (`multisfl_implementation/multisfl/drift_measurement.py`):

| 추가 항목 | 설명 |
|----------|------|
| `DriftMetrics.A_cos_client` | Client-side pairwise alignment |
| `DriftMetrics.M_norm_client` | Client-side mean update norm |
| `DriftMetrics.A_cos_server` | **Server-side** pairwise alignment (MultiSFL 고유) |
| `DriftMetrics.M_norm_server` | Server-side mean update norm |
| `DriftMetrics.n_valid_alignment` | 유효 브랜치 수 |
| `_branch_deltas` | Client Δ 저장 |
| `_branch_server_deltas` | Server Δ 저장 |
| `collect_branch_delta()` | Client model delta 수집 |
| `collect_branch_server_delta()` | Server model delta 수집 |

**MultiSFL이 server-side A_cos를 지원하는 이유:**
다른 트랙에서는 서버 모델이 하나뿐이라 pairwise 비교 불가능. MultiSFL은 브랜치별로 독립적인 서버 모델이 있으므로 브랜치 간 서버 업데이트 방향 정렬을 측정할 수 있음.

**호출 지점** (`trainer.py:~521`):
```python
# finalize_branch() 직후, compute_master() 전:
self.drift_tracker.collect_branch_delta(b, bc.model.state_dict())
bs_for_delta = self.main.get_server_branch(b)
self.drift_tracker.collect_branch_server_delta(b, bs_for_delta.model.state_dict())
```

**데이터 흐름:**
```
라운드 시작: on_round_start(master_client_sd, master_server_sd) → θ_start 저장
브랜치 학습: accumulate_branch_drift() / accumulate_server_drift()
브랜치 완료: finalize_branch()
             → collect_branch_delta(b, client_model)        # Client Δ_b
             → collect_branch_server_delta(b, server_model)  # Server Δ_b
compute_master() + soft_pull()
on_round_end() → compute_update_alignment(client_deltas)  → A_cos_client
              → compute_update_alignment(server_deltas)  → A_cos_server
```

---

## 6. 출력 포맷

### 콘솔 로그

**SFL/USFL/GAS:**
```
[Drift] Round 5: Client(G_drift=0.012000, G_end=0.008000) Server(G_drift=0.006000, G_end=0.004000, steps=50) Total(G_drift=0.018000) A_cos=0.7234 M_norm=0.003421 (n_valid=10)
```

**MultiSFL:**
```
[Drift] Round 5: Client(G_drift=0.012000, G_end=0.008000) Server(G_drift=0.006000, G_end=0.004000, steps=15) Total(G_drift=0.018000) A_cos(c=0.7234, s=0.8901)
```

### JSON 결과 (to_dict)

**SFL/USFL/GAS:**
```json
{
  "G_drift_client": 0.012,
  "G_drift_server": 0.006,
  "G_drift_total": 0.018,
  "G_end_client": 0.008,
  "G_end_server": 0.004,
  "G_drift_norm_client": 1.5,
  "A_cos": 0.7234,
  "M_norm": 0.003421,
  "n_valid_alignment": 10,
  "D_dir_client_weighted": 0.002,
  "D_rel_client_weighted": 0.15,
  "num_clients": 10,
  "server_steps": 50
}
```

**MultiSFL:**
```json
{
  "G_drift_client": 0.012,
  "G_drift_server": 0.006,
  "G_drift_total": 0.018,
  "A_cos_client": 0.7234,
  "M_norm_client": 0.003421,
  "A_cos_server": 0.8901,
  "M_norm_server": 0.001234,
  "n_valid_alignment": 3,
  "num_branches": 3,
  "server_steps": 15
}
```

### History (시계열)

```python
history = drift_tracker.get_history()  # or get_all_measurements()
# history["A_cos"] = [0.45, 0.52, 0.61, 0.68, 0.72, ...]  # 라운드별 추이
# history["M_norm"] = [0.008, 0.006, 0.005, 0.004, 0.003, ...]
```

---

## 7. 리소스 추정

### ResNet18 기준 (split_layer = "layer1.1.bn2")

| 항목 | Client Model | Server Model |
|------|-------------|-------------|
| Trainable params (BN 제외) | ~2.3M | ~8.9M |
| Δ_i 벡터 크기 | ~9.2 MB | ~35.6 MB |

| 구성 | 메모리 (Δ 저장) | 연산 (cosine) |
|------|----------------|---------------|
| 10 clients | ~92 MB (client) | 45 pairs × O(2.3M) ≈ 0.1초 |
| 10 branches + server | ~92 + 356 MB | 45 + 45 pairs ≈ 0.3초 |

모든 Δ 벡터는 CPU에 저장되며, `on_round_end()` 이후 `on_round_start()`에서 초기화됨.

---

## 8. 기법 간 비교 시 해석 가이드

### 비교 가능한 메트릭 조합

| 메트릭 | 기법 간 비교 가능 | 이유 |
|--------|:--:|------|
| **A_cos** | O | Scale-invariant, step 정의에 무관 |
| **M_norm** | △ | lr/batch_size에 비례하므로 동일 설정일 때만 |
| G_drift_client | X | step 수, gradient scale에 의존 |
| G_drift_server | X | 서버 step 횟수가 기법마다 극적으로 다름 |
| G_drift_total | X | 위 두 개의 합 |
| D_dir_client | △ | aggregation weight 방식에 의존 |

### 실험 보고 권장 포맷

```
| Method | Accuracy | A_cos↑ | M_norm | G_drift_total (참고) |
|--------|----------|--------|--------|---------------------|
| SFL    | 72.3%    | 0.42   | 0.0051 | 0.082               |
| USFL   | 78.1%    | 0.68   | 0.0034 | 0.021               |
| GAS    | 76.5%    | 0.61   | 0.0042 | 0.045               |
| MultiSFL| 77.8%   | 0.65   | 0.0038 | 0.031               |
```

→ A_cos와 Accuracy의 상관관계가 G_drift보다 높을 것으로 예상됨.

---

## 9. 주의사항 및 한계

### ||Δ|| ≈ 0 불안정성
- `tau = 1e-7` 기본값으로 필터링
- `n_valid < 2`이면 `A_cos = NaN` 반환
- M_norm에는 모든 클라이언트 포함 (필터 미적용)

### 가중치 선택
- **SFL/USFL Framework**: `augmented_label_counts` 기반 (aggregation weight과 동일)
- **GAS**: Uniform (균등 aggregation이므로)
- **MultiSFL**: Uniform (브랜치 단위, 동일 가중)

### Client/Server 분리
- Client-side A_cos: 모든 트랙에서 계산 가능
- Server-side A_cos: **MultiSFL만 가능** (브랜치별 독립 서버 모델)
- 다른 트랙의 서버는 단일 모델 → pairwise 불가능 → 기존 `G_drift_server` 유지

### soft_pull 영향 (MultiSFL)
- `on_round_end`의 `Δx = new_master - old_master`에 soft_pull 효과가 포함됨
- 하지만 `collect_branch_delta()`는 soft_pull **전에** 호출되므로 A_cos는 순수 학습 drift만 반영
- 이것이 MultiSFL에서 A_cos가 기존 G_drift보다 더 정확한 이유

### 메모리 관리
- Δ 벡터는 CPU에 저장 (GPU 메모리 미사용)
- `on_round_start()`에서 이전 라운드 Δ 자동 해제
- 대규모 모델 (>100M params)에서는 `sample_interval` 고려 필요

---

## 10. 향후 확장 가능성

1. **시각화 스크립트**: `history["A_cos"]` 시계열 + `history["M_norm"]` 이중 축 플롯
2. **Per-class A_cos**: 클라이언트 보유 label 기준으로 그룹화하여 class-level alignment 측정
3. **A_cos decay 분석**: 라운드 진행에 따른 A_cos 추이로 수렴 속도 비교
4. **Cross-track 통합 실험 설정**: `use_sfl_transform`, `load_torchvision_resnet18_init()` + A_cos 기록으로 공정 벤치마크
