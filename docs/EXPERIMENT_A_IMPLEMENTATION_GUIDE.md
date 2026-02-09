# Experiment A Implementation Guide (실구현/실행/논문 리포팅)

이 문서는 `docs/experiment.md`의 **정의(spec)** 를 현재 코드에 실제로 어떻게 구현했는지, 그리고 실험을 어떻게 돌려서 논문용 지표를 뽑는지 정리한 실행 가이드다.

관련 기준 문서:
- `docs/experiment.md` (실험 정의/수식)
- `docs/TRAINING_DYNAMICS_COMPARISON.md` (기법별 동역학 차이)

---

## 1) 구현 범위 요약

현재 코드베이스는 Experiment A의 핵심을 다음 형태로 구현한다.

- 공통 메트릭 엔진 분리
  - `shared/experiment_a_metrics.py`
- probe set(Q) 전용 로더/방향 계산 추가
  - SFL: `sfl_framework-fork-feature-training-tracker/server/utils/experiment_a_probe.py`
  - GAS: `GAS_implementation/utils/experiment_a_probe.py`
  - MultiSFL: `multisfl_implementation/multisfl/experiment_a_probe.py`
- 프레임워크별 드리프트 트래커에서 Experiment A 계산/로그
  - SFL: `sfl_framework-fork-feature-training-tracker/server/utils/drift_measurement.py`
  - GAS: `GAS_implementation/utils/drift_measurement.py`
  - MultiSFL: `multisfl_implementation/multisfl/drift_measurement.py`
- 공통 실행 프레임워크(`experiment_core`)에 probe/schedule 전달 및 정규화 연동
  - `experiment_core/sfl_runner.py`
  - `experiment_core/adapters/gas_adapter.py`
  - `experiment_core/adapters/multisfl_adapter.py`
  - `experiment_core/normalization.py`

---

## 2) 메트릭 정의와 코드 매핑

Experiment A 공통 계산 함수는 `compute_experiment_a_metrics(...)` 하나로 통일되어 있다.

위치:
- `shared/experiment_a_metrics.py`

입력:
- `client_deltas`: 각 참여자 업데이트 벡터 `Δ_i`
- `client_weights`: 가중치(`B_i` 기반)
- `client_probe_direction`: 공통 probe 방향 `c_c^t`
- `per_client_probe_directions`: 비동기 보강용 `c_{c,i}^t`
- `server_delta`: 서버 업데이트 `Δ_s^t`
- `server_probe_direction`: 서버 probe 방향 `c_s^t`
- `server_steps`: 서버 optimizer step 수 `K_s^t`

출력(핵심):
- `A_c_ratio = var_c / (m2_c + eps)`  
  - 메인 스케일 불변 client drift ratio
- `B_c = 1 - cos(mu_c, c_c)`  
  - 합의 업데이트의 중앙 정렬 오차
- `C_c_per_client_probe`  
  - 비동기 기법에서 참여자별 시작점 기반 probe 정렬 오차
- `B_s = 1 - cos(Δ_s, c_s)`  
  - 서버 중앙 정렬 오차
- `server_mag_per_step = ||Δ_s|| / (K_s + eps)`  
  - 서버 step 정규화 크기
- `server_mag_per_step_sq = ||Δ_s||^2 / (K_s + eps)`  
  - 제곱 버전(보조)

참고:
- `C_c`(공통 anchor probe 기준)도 같이 기록된다.
- `m2_c`, `u2_c`, `var_c`를 함께 기록하므로 “업데이트 전체 억제”와 “실제 정렬 개선”을 분리해서 해석 가능하다.

---

## 3) 비동기 정확성(중요)

GAS/MultiSFL는 라운드 시작 client 상태가 참여자마다 다를 수 있으므로, Experiment A에서 다음을 별도 처리한다.

1. 각 참여자 시작점 기록 `x_{c,i}^{t,0}`
- GAS: `record_client_start_state(...)`
- MultiSFL: `record_branch_start_state(...)`

2. `Δ_i = x_{c,i}^{t,end} - x_{c,i}^{t,0}`로 계산
- GAS/MultiSFL 드리프트 트래커 내부에서 반영

3. 참여자별 probe 방향 `c_{c,i}^t` 계산
- round-start participant state에 대해 probe gradient 계산
- `C_c_per_client_probe`로 로깅

즉 비동기 기법에서도 “각 참여자의 자기 시작점 기준 정의”를 따르도록 구현되어 있다.

---

## 4) Probe set(Q) 외부 설정/전용 로더 동작

공통 함수:
- `build_probe_loader(...)`
- `compute_split_probe_directions(...)`

### 4.1 설정 파라미터

- `probe_source`: `test` 또는 `train`
- `probe_indices_path`: 고정 인덱스 파일(JSON/TXT)
- `probe_num_samples`: 인덱스 파일이 없을 때, 고정 시드 샘플 수
- `probe_batch_size`: 0이면 기본 loader batch 재사용
- `probe_max_batches`: 라운드당 probe 배치 수
- `probe_seed`: subset 샘플 시드
- `probe_class_balanced`: 라벨 접근 가능 시 class-balanced subset 샘플링
- `probe_class_balanced_batches`: 가능하면 probe batch도 class-balanced가 되도록 인덱스 재정렬

### 4.2 인덱스 파일 포맷

JSON:
- 리스트
  - 예: `[0, 7, 11, 52]`
- 또는 객체
  - 키 중 하나를 자동 인식: `indices`, `probe_indices`, `q_indices`
  - 예: `{"q_indices": [0, 7, 11, 52]}`

TXT:
- 한 줄에 하나의 인덱스

전처리 규칙:
- 음수 제거
- 중복 제거
- dataset 범위 밖 인덱스 제거

### 4.3 선택 우선순위

1. `probe_indices_path` 유효하면 해당 인덱스 사용
2. 아니면 `probe_num_samples > 0`이면 시드 기반 랜덤 subset
3. 둘 다 아니면 해당 source split 전체 사용

### 4.4 방향 계산 방식

- 모델을 `eval()`로 두고 probe 배치를 순회
- 각 배치 gradient를 샘플 수 가중합 후 평균
- `c = -∇L_Q`를 flatten 벡터로 반환
- round당 최대 `probe_max_batches`까지만 사용

---

## 5) 프레임워크별 실제 연결 지점

### 5.1 SFL/USFL/Mix2SFL

- Probe loader 구성:
  - `sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/sfl_stage_organizer.py`
  - `.../usfl_stage_organizer.py`
  - `.../mix2sfl_stage_organizer.py`
- round 시작 시 `c_c^t, c_s^t` 계산 후 drift tracker 주입
- round 종료 시 `DRIFT_MEASUREMENT` 이벤트에 `experiment_a` 포함 저장
- 결과 집계:
  - `sfl_framework-fork-feature-training-tracker/server/modules/global_dict/global_dict.py`
  - `drift_history.expA_*` 키로 정리

### 5.2 GAS

- 환경변수 기반 probe/schedule 설정:
  - `GAS_PROBE_SOURCE`
  - `GAS_PROBE_INDICES_PATH`
  - `GAS_PROBE_NUM_SAMPLES`
  - `GAS_PROBE_BATCH_SIZE`
  - `GAS_PROBE_MAX_BATCHES`
  - `GAS_PROBE_SEED`
  - `GAS_CLIENT_SCHEDULE_PATH`
- 구현:
  - `GAS_implementation/GAS_main.py`
- 비동기 participant start-state 및 per-client probe 방향 계산 포함
- 출력:
  - `results_gas_*.json`의 `drift_history.expA_*`

### 5.3 MultiSFL

- CLI 기반 probe/schedule 설정:
  - `--probe_source`
  - `--probe_indices_path`
  - `--probe_num_samples`
  - `--probe_batch_size`
  - `--probe_max_batches`
  - `--probe_seed`
  - `--client_schedule_path`
- 구현:
  - `multisfl_implementation/run_multisfl.py`
  - `multisfl_implementation/multisfl/trainer.py`
- branch start-state/per-branch probe 방향 계산 포함
- 출력:
  - `results_multisfl_*.json`의 `drift_history.expA_*`

---

## 6) Experiment A 실행 플로우 (권장)

### Step 1. 고정 참여자 스케줄 파일 준비 (`P_t` 고정)

예시 1: list 형식
```json
[
  [0,1,2,3,4,5,6,7,8,9],
  [10,11,12,13,14,15,16,17,18,19]
]
```

예시 2: dict 형식
```json
{
  "rounds": [
    [0,1,2,3,4,5,6,7,8,9],
    [10,11,12,13,14,15,16,17,18,19]
  ]
}
```

### Step 2. 고정 probe 인덱스 파일 준비 (`Q` 고정)

예시:
```json
{ "q_indices": [5, 12, 30, 44, 101, 333] }
```

참고:
- `probe_source=test`를 쓰면 probe 5000장을 사용해도 학습(train partition)에서 샘플을 제외하지 않는다.

### Step 3. `experiment_configs/common.json`에 반영

최소 필수:
- `enable_drift_measurement: true`
- `client_schedule_path: <스케줄 파일 경로>`
- `probe_source: test`
- `probe_indices_path: ""` (인덱스 파일 미사용 시)
- `probe_num_samples: 5000`
- `probe_batch_size: 500`
- `probe_max_batches: 10`
- `probe_seed: 42`
- `probe_class_balanced: true`
- `probe_class_balanced_batches: true`

### Step 4. batch spec 생성

```bash
python -m experiment_core.generate_spec \
  --config-dir experiment_configs \
  --methods sfl usfl gas mix2sfl multisfl \
  --gpu-map '{"sfl":0,"usfl":1,"gas":2,"mix2sfl":3,"multisfl":0}' \
  --output /tmp/expA_batch.json \
  --output-dir results/expA_run
```

### Step 5. 실행

```bash
python -m experiment_core.batch_runner \
  --spec /tmp/expA_batch.json \
  --repo-root .
```

### Step 6. 산출물

- Raw
  - SFL 계열: `results/expA_run/result-*.json`
  - GAS: `results/expA_run/results_gas_*.json`
  - MultiSFL: `results/expA_run/results_multisfl_*.json`
- Normalized
  - `results/expA_run/*.normalized.json`

---

## 7) 논문용 지표를 실제로 어디서 뽑는가

권장: normalized 결과(`*.normalized.json`)의 `experiment_a_history` 사용.

키 매핑:

- Client drift (A-메인)
  - `experiment_a_history.A_c_ratio`
- Client 합의-중앙 정렬
  - `experiment_a_history.B_c`
- Client 개별-중앙 정렬(비동기 보강)
  - `experiment_a_history.C_c_per_client_probe`
- Server 중앙 정렬
  - `experiment_a_history.B_s`
- Server step 정규화 크기
  - `experiment_a_history.server_mag_per_step`
  - (보조) `experiment_a_history.server_mag_per_step_sq`
- 보조 진단
  - `experiment_a_history.m2_c`, `u2_c`, `var_c`

raw에서도 동일하게 `drift_history.expA_*` 키로 접근 가능하다.

---

## 8) 논문 리포팅 템플릿 (권장)

### 8.1 메인 표

기법별로 아래 3개를 같은 조건에서 비교:
- `mean_t(A_c_ratio^t)` (낮을수록 좋음)
- `mean_t(B_c^t)` (낮을수록 좋음)
- `mean_t(B_s^t)` (낮을수록 좋음)

보조:
- `mean_t(server_mag_per_step^t)`로 서버 업데이트 규모 보정 확인

### 8.2 메인 그림

- Round-wise 곡선:
  - `A_c_ratio^t`
  - `B_c^t`
  - `B_s^t`
  - `server_mag_per_step^t`

### 8.3 권장 집계

- seed별 round curve 계산 후
- 최종 리포트는 seed 평균±표준편차
- 필요 시 warm-up 구간 제외 평균 병행

---

## 9) “기법 간 1:1 정합 비교가 가능한가?”에 대한 현재 결론

이 구현은 아래를 충족한다.

- client grad scale 차이의 직접 영향 축소:
  - `A_c_ratio`, `B_c`, `B_s`는 cosine/ratio 기반
- server update 횟수 차이 보정:
  - `server_mag_per_step` 제공
- 참여자 구성 변동 통제:
  - `client_schedule_path`로 `P_t` 고정 가능
- 중앙 방향 정의 통일:
  - 고정 `Q`(probe indices) 사용 가능

단, 완전한 “절대 동일 실험”은 아니다. 남는 차이:
- 모델 구조/optimizer 내부상태/branch 구조 차이
- 각 기법 고유의 학습 궤적 차이

그래서 논문에서는 메인 결론을 `A_c_ratio/B_c/B_s`로 두고, `server_mag_per_step`, `m2/u2/var`, 정확도 곡선을 함께 제시하는 구성이 안전하다.

---

## 10) 재현 체크리스트

- `enable_drift_measurement=true`
- 고정 `client_schedule_path` 사용
- 고정 `probe_indices_path` 사용
- `probe_source`, `probe_max_batches`, `probe_batch_size`를 모든 기법 동일 설정
- seed 통일(`common.seed`, `probe_seed`)
- 동일 분할/round/clients_per_round/local_epochs
- normalized JSON 기반으로 동일 post-processing
