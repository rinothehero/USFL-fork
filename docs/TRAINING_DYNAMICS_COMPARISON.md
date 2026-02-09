# Training Dynamics Comparison Across Methods

이 문서는 현재 코드 구현 기준으로, 기법별 학습 동역학 차이를 정리한다.
핵심 비교축은 아래 3가지다.

1. client 모델에 전달/적용되는 gradient scale
2. client 모델 상태 동기화 방식 (라운드 시작 시 동일/상이)
3. 서버 모델 업데이트 횟수

## Scope and Notation

- 기법: `SFL`, `USFL`, `GAS`, `Mix2SFL`, `MultiSFL`
- 예시 전제: 총 100명 중 라운드당 10명 참여
- 기호:
  - `n`: 라운드 참여 클라이언트 수 (예: 10)
  - `B_i`: i번째 클라이언트 main 배치 크기
  - `sumB = Σ_i B_i`: concat된 총 배치 크기
  - `R_i`: MultiSFL branch i의 replay 샘플 수
  - `Np`: non-empty 참여자 수 (USFL의 `scale_client_grad`에서 사용)
- `client update 1회`: client optimizer `step()` 1회

## One-Page Summary

| Method | Client gradient scale (default) | Client state at round start | Server updates while `n` clients each do 1 update |
|---|---|---|---|
| SFL | `~ 1 / B_i` | 동일 global client model | `n`회 (1:1) |
| USFL | `~ 1 / sumB` (slice 후 전달) | 동일 global client model | 1회 (concat 1회 기준) |
| GAS | `~ 1 / B_i` (client-side local loss) | 비동기 슬롯 상태(실행 중 서로 다를 수 있음) | 1회 (`count_concat == n`) |
| Mix2SFL | 기본 `~ 1 / sumB` (GradMix 대상은 sum/mean로 변형) | 동일 global client model | 1회 (concat 1회 기준) |
| MultiSFL | main 기준 `~ 1 / (B_i + R_i)` (`R_i` replay 영향) | 브랜치별 상이 (persistent + soft-pull) | `n`회 per local step (라운드 총량은 `n * steps_to_run`) |

주의:
- 위 scale은 CE(`reduction="mean"`) 기본과 전달 경로 기준이다.
- 실제 수치는 gradient clipping, shuffle, optimizer state 등에 의해 추가 변형될 수 있다.

## 1) SFL

### 1.1 Gradient scale

SFL 서버는 activation을 클라이언트 도착 순서대로 1개씩 처리한다.
서버 loss는 해당 클라이언트 배치만으로 계산되며 CE(mean) 기준이므로, cut-layer gradient scale은 기본적으로 `1 / B_i`다.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/sfl_stage_organizer.py:243`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/in_round/in_round.py:283`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/sfl_stage_organizer.py:343`

### 1.2 Client state

라운드 시작 시 선택 클라이언트들에게 동일한 client split model을 전송한다.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/sfl_stage_organizer.py:197`

### 1.3 Server update count

activation 1개 처리당 서버 backward/step 1회이므로, 참여자 `n`명이 각 1회 client update를 하면 서버도 `n`회 update된다(1:1 대응).

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/in_round/in_round.py:297`

## 2) USFL

### 2.1 Gradient scale

USFL은 non-empty activation들을 concat해서 서버에서 한 번에 CE(mean) backward한다.
그 뒤 생성된 큰 cut-layer gradient를 `start:end`로 잘라 각 client에 전달한다.
따라서 기본 scale은 `~ 1 / sumB`다.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/usfl_stage_organizer.py:959`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/usfl_stage_organizer.py:1026`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/usfl_stage_organizer.py:1242`

옵션:
- `--scale-client-grad`가 켜지면 전달 전 gradient에 `Np`를 곱한다.
  - scale: `~ Np / sumB`
  - 배치가 균일하면 대략 `~ 1 / B`

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/usfl_stage_organizer.py:1042`

### 2.2 Client state

라운드 시작 시 선택된 클라이언트는 동일 global client model에서 시작한다.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/usfl_stage_organizer.py:915`

### 2.3 Server update count

concat 1회 처리당 서버 update 1회다.
즉 “참여 `n`명이 동기 1step씩 수행하는 iteration” 기준 서버 1회 update.

## 3) GAS

### 3.1 Gradient scale

GAS의 client-side update는 각 selected client에 대해 로컬 loss(CE mean)로 수행되므로 client gradient scale은 기본 `~ 1 / B_i`다.

관련 코드:
- `/GAS_implementation/GAS_main.py:1009`
- `/GAS_implementation/GAS_main.py:1018`
- `/GAS_implementation/GAS_main.py:1078`

서버 update는 concat된 feature 버퍼(`concat_features`)로 CE(mean) 1회 수행한다.

관련 코드:
- `/GAS_implementation/GAS_main.py:1093`
- `/GAS_implementation/GAS_main.py:1166`
- `/GAS_implementation/GAS_main.py:1215`

### 3.2 Client state

GAS는 active slot별 `usersParam`을 유지하며 비동기적으로 진전한다.
실행 중 슬롯별 상태는 일반적으로 동일하지 않다.

관련 코드:
- `/GAS_implementation/GAS_main.py:852`
- `/GAS_implementation/GAS_main.py:960`
- `/GAS_implementation/GAS_main.py:1078`
- `/GAS_implementation/GAS_main.py:1262`

### 3.3 Server update count

`count_concat == user_parti_num`일 때 서버가 1회 update된다.
즉 “client-side iteration `n`회 누적”당 서버 1회.
주의: 이 `n`회는 반드시 서로 다른 `n`명 1회씩을 의미하지는 않는다(스케줄링에 따라 중복 가능).

관련 코드:
- `/GAS_implementation/GAS_main.py:1093`

## 4) Mix2SFL

### 4.1 Gradient scale

기본 cut-layer gradient는 USFL과 동일하게 concat CE(mean)에서 생성되어 slice 배포된다.
즉 기본 scale은 `~ 1 / sumB`.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:355`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:375`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:534`

중요:
- SmashMix loss는 client 전송 grad에 직접 반영되지 않는다.
  - 원본 grad를 먼저 캡처하고,
  - SmashMix는 detached output으로 서버 파라미터 업데이트에만 추가된다.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:387`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:465`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:512`

GradMix:
- `C'` 대상에게는 per-client slice 대신 broadcast gradient 전달
- `reduce=sum` (default): 크기 증폭 가능
- `reduce=mean`: 평균 broadcast

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:549`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:581`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:595`

### 4.2 Client state

라운드 시작 시 선택된 클라이언트는 동일 global client model에서 시작한다.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:301`

### 4.3 Server update count

concat 원본 loss backward + SmashMix backward를 누적한 뒤 `optimizer.step()` 1회 수행한다.
즉 동기 iteration 기준 서버 1회 update.

관련 코드:
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:378`
- `/sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/mix2sfl_stage_organizer.py:514`

## 5) MultiSFL

### 5.1 Gradient scale

브랜치 서버는 main + replay를 합친 `y_all`로 CE(mean) backward하고,
클라이언트로는 main feature(`f_main`)의 gradient만 반환한다.
따라서 replay가 있으면 main 샘플 기준 effective scale이 작아진다(대략 `~ 1 / (B_i + R_i)`).

관련 코드:
- `/multisfl_implementation/multisfl/servers.py:170`
- `/multisfl_implementation/multisfl/servers.py:181`
- `/multisfl_implementation/multisfl/servers.py:187`
- `/multisfl_implementation/multisfl/trainer.py:497`

### 5.2 Client state

브랜치 client/server 모델은 persistent하고, 라운드마다 master 평균 + soft-pull만 적용된다.
즉 브랜치 간 초기 상태는 일반적으로 서로 다르다.

관련 코드:
- `/multisfl_implementation/multisfl/trainer.py:566`
- `/multisfl_implementation/multisfl/trainer.py:568`

### 5.3 Server update count

브랜치 b, local_step마다:
- 서버 branch update 1회
- 클라이언트 branch update 1회

관련 코드:
- `/multisfl_implementation/multisfl/trainer.py:357`
- `/multisfl_implementation/multisfl/trainer.py:482`
- `/multisfl_implementation/multisfl/servers.py:209`
- `/multisfl_implementation/multisfl/client.py:137`

따라서:
- 동기 local step 1번 기준: 서버 총 `n`회
- 라운드 총량 기준: 서버 총 `n * steps_to_run`회

`steps_to_run` 정의:
- 기본(`use_full_epochs=False`): `steps_to_run = local_steps`
- full-epochs 모드(`use_full_epochs=True`): `steps_to_run = n_batches * local_steps`

관련 코드:
- `/multisfl_implementation/multisfl/trainer.py:345`
- `/multisfl_implementation/multisfl/trainer.py:352`

## Why This Causes Cross-Method Mismatch

동일 LR를 써도 아래가 다르면 실제 update 크기/빈도는 달라진다.

1. 분모 차이:
- `1/B_i` vs `1/sumB` vs `1/(B_i+R_i)`

2. 서버 step budget 차이:
- client 1회당 서버 1:1 (SFL, MultiSFL branch-step 기준)
- client `n`회당 서버 1회 (USFL, Mix2SFL, GAS concat-step 기준)

3. 초기 상태 차이:
- 매 라운드 동일 global 시작(SFL/USFL/Mix2SFL) vs 비동기/persistent 상태(GAS/MultiSFL)

즉, metric 비교 시 “알고리즘 차이” 외에 “학습 동역학 차이(스케일/step 빈도/초기상태)”가 성능 차이에 섞여 들어갈 수 있다.
