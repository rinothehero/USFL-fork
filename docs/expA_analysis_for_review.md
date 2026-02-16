# Experiment A 분석 결과 및 논문 모티베이션 정리

## 1. 실험 개요

6개 SFL 기법을 동일 조건에서 비교. SFL IID를 upper bound baseline으로 포함.

**SFL에서 Non-IID가 특히 문제인 이유**: 기존 FL(Federated Learning)은 client만 Non-IID 영향을 받지만, SFL(Split FL)은 모델이 client/server로 분할되어 activation과 gradient를 교환하므로, **server side도 Non-IID에 직접 노출**됨. Client가 편향된 데이터로 생성한 activation을 server가 그대로 받아 학습하기 때문에, client drift뿐 아니라 server drift도 발생.

**실험 조건**:
- Dataset: CIFAR-10
- Model: ResNet18 (split at layer2)
- Non-IID: shard_dirichlet, α=0.3, labels_per_client=2
- Clients: 100 total, 10 per round
- Rounds: 300
- Local epochs: 5 (full epoch)
- Learning rate: 0.001
- Client batch size: 50 (USFL의 bs=500은 server-side concat 크기이며, client 기대 배치는 ~50으로 동일)
- 동일 client schedule (P_t 고정), 동일 probe set (Q 고정, test set 5000장)

---

## 2. 기존 논문 가설 (6단계)

우리가 원래 논문에서 세우려 했던 모티베이션 논리:

1. **[관찰]** 기존 SFL 연구들은 Non-IID + partial participation에서 정확도가 크게 하락한다.
2. **[원인 분석]** 기존 연구들은 Non-IID에서 SFL IID 대비 model drift가 매우 나쁘다.
3. **[방향 설정]** 따라서 정확도 개선을 위해서는 model drift를 낮춰야 한다. 특히 client drift를 낮춰야 한다.
4. **[우리 방법]** USFL은 activation balancing with concatenation + gradient shuffle을 통해 client 측 모델의 split-layer gradient batch를 전역과 align 시킬 수 있다.
5. **[효과]** 따라서 client drift와 server drift를 해결했다.
6. **[결과]** 결과적으로 기존 SOTA 기법들에 비해 정확도가 높았고, SFL IID와 비슷한 수준까지 올라왔다.

**용어 정리 — drift vs alignment**:
- **Model drift** (G_drift 계열): 학습 중 모델 파라미터가 얼마나 이동했는지의 크기 (방향 무관, scalar)
- **Alignment** (B_c, B_s 계열): 모델 업데이트가 전역 최적 방향과 얼마나 일치하는지 (cosine 기반, 방향성)
- 가설에서는 "drift를 줄여야 한다"고 했으나, 실제 측정에서는 drift(크기)뿐 아니라 alignment(방향)도 함께 측정. 아래 분석에서 drift는 정확도를 설명하지 못하고, alignment의 worst-case가 설명력이 있음을 보임.

→ 이 가설을 Experiment A 메트릭으로 검증하려 했으나, **3번과 5번이 데이터에 의해 부정됨** (상세 분석은 Section 7 참조).

---

## 3. 비교 기법별 핵심 알고리즘

### 3.1 SFL (Split Federated Learning) — Baseline

- 모델을 client(bottom layers)와 server(top layers)로 분할
- Client: forward pass → activation을 server에 전송
- Server: activation으로 forward → loss 계산 → backward → gradient를 client에 반환
- Client: gradient로 backward pass
- 매 라운드 FedAvg로 client/server 모델 각각 집계
- **Non-IID 대응 없음**

### 3.2 USFL (Unified Split Federated Learning) — 우리 연구

- SFL 기반 + 2가지 핵심 Non-IID 최적화:
  1. **Activation Balancing with Concatenation**: 선택된 ~10 clients의 activation을 server에서 concat하여 하나의 balanced batch(~500)로 처리. 클래스 불균형을 강제로 over/undersampling을 해서 강제로 균형을 맞춰서,  Server가 class-균형 배치로 학습 (즉, 전역과 일치하는 분포로 만듬)
  2. **Gradient Shuffle**: Server가 계산한 gradient를 client에 돌려주기 전에 shuffle (random). 각 Client가 받는 gradient가 기대값 기준 전역 분포를 반영하게 함.

### 3.3 MIX2SFL (Mix-to Split Federated Learning)

- SFL 기반 + 2가지 mixing 기법:
  1. **SmashMix (Activation Mixup)**: 각 client의 activation(smashed data)에 대해 다른 client를 partner로 샘플링하여 `mixed_act = λ·act_i + (1-λ)·act_j` 생성. λ는 Beta(α,α)에서 샘플링. Server는 mixed activation으로 forward → soft cross-entropy loss (mixed label 기반). 즉, **server의 forward 자체가 augmented된 activation으로 수행됨**.
  2. **GradMix (Gradient Mixing)**: 참여 client 중 subset C'를 선택, 이들의 gradient를 집계(aggregation)하여 C'에 속한 client들에게 동일 gradient 반환 (mixing ratio 적용).
- **USFL과의 차이**: USFL은 activation을 concat하여 하나의 balanced batch를 만드는 반면, MIX2SFL은 pair-wise interpolation(SmashMix)으로 개별 client activation을 augment. GradMix는 USFL의 gradient shuffle과 유사하지만 subset 기반.

### 3.4 GAS (Generative Activation-aided SFL)

- SFL 기반 + **완전 비동기 학습** (client를 1명씩 순차 처리, WRTT 기반 선택)
- 3가지 핵심 메커니즘:
  1. **Generative Activation**: Server가 각 label의 activation 통계(평균, 분산)를 IncrementalStats로 추적. 부족한 class의 synthetic activation을 Gaussian 분포(N(μ_label, σ_label))에서 생성하여 학습 보강. 즉, **activation 수준에서 Non-IID 완화**.
  2. **V-value (Gradient Dissimilarity)**: 각 client에 대해 real activation과 synthetic activation으로 각각 server gradient를 계산, 두 gradient의 dissimilarity로 V-value 측정. V-value가 높은 client = 전역 분포와 차이가 큰 client.
  3. **Local Adjustment**: Client별 label 분포(p_i)에 기반한 logit bias를 server 예측에 적용. V-value에 비례하여 adjustment 강도 조절. Class imbalance에 의한 편향을 보정.
- **USFL과의 차이**: USFL은 여러 client를 병렬로 처리하고 activation concat, GAS는 1명씩 비동기 처리하고 synthetic activation으로 보강. Gradient shuffle은 사용하지 않음.

### 3.5 MultiSFL (Multi-Branch Split Federated Learning)

- 여러 branch server가 각각 다른 model split을 유지
- MainServer: 전체 집계, FedServer: branch별 독립 학습
- Knowledge replay: 과거 client data를 replay하여 branch 간 균형
- SamplingProportionScheduler: branch별 client 배분 동적 조정
- soft_pull_to_master: branch가 main model 방향으로 주기적 pull (α=0.1)
- **구조적 차이**: 다른 기법은 단일 server model, MultiSFL은 multi-branch

### 3.6 SFL IID — Upper Bound Baseline

- SFL과 동일한 알고리즘
- 차이점: data distribution이 IID (uniform)
- Non-IID의 영향을 측정하기 위한 reference

---

## 4. Experiment A 메트릭 정의

**Experiment A의 핵심 아이디어**: 고정된 probe set(test data 5000장)으로 "전역 최적 업데이트 방향(c)"을 매 라운드 정의하고, 각 기법의 client/server 업데이트가 그 방향과 얼마나 정렬되는지 측정. 정렬이 잘 되면 학습이 올바른 방향으로 진행, 어긋나면 학습이 비효율적이거나 역행.

### 4.1 핵심 메트릭 (Experiment A Probe 기반)

| 메트릭 | 수식 | 의미 | 방향 |
|--------|------|------|------|
| **A_c_ratio** | var_c / (m2_c + ε) | Client update 중 variance 비율 (noise/signal ratio) | **↓ 낮을수록 좋음** |
| **B_c** | 1 - cos(μ_c, c_c) | Client 합의 업데이트와 전역 최적 방향 간 정렬 오차 | **↓ 낮을수록 좋음** |
| **B_s** | 1 - cos(Δ_s, c_s) | Server 업데이트의 정렬 오차 | **↓ 낮을수록 좋음** |
| **C_c** | 1 - cos(μ_c, c_c per-client) | 개별 client probe 방향 기준 합의 업데이트 정렬 오차 | **↓ 낮을수록 좋음** |

B_c 값의 의미:
- B_c = 0: cos = 1, 완벽 정렬 (optimal 방향과 같은 방향)
- B_c = 1: cos = 0, 직교 (무관한 방향)
- **B_c > 1: cos < 0, 반대 방향 (optimal 반대로 이동 = reversal)**

### 4.2 보조 메트릭

| 메트릭 | 의미 |
|--------|------|
| m2_c | Client update 전체 에너지 E[‖Δ_i‖²] |
| u2_c | Client 합의 신호 크기 ‖μ_c‖² |
| var_c | Client update 분산 E[‖Δ_i - μ_c‖²] |
| server_mag_per_step | Server update 크기 / step 수 |

### 4.3 Drift 메트릭 (G_drift 계열)

| 메트릭 | 의미 |
|--------|------|
| G_drift | Client trajectory energy (1/\|P_t\|) Σ(S_n/B_n) |
| G_drift_norm | G_drift를 global update norm으로 정규화 |
| delta_client_norm_sq | 라운드 후 client model 변화량 ‖Δ_client‖² |

---

## 5. 측정 방법론

### 5.1 Probe 방향 (c) 계산

모든 기법에서 동일한 방식으로 **전역 최적 방향**을 정의:

1. 고정된 probe set Q (test set에서 class-balanced로 5000장 샘플, seed=42)
2. 매 라운드 시작 시, 현재 global model을 eval 모드로 두고 Q에 대해 gradient 계산
3. `c = -∇L_Q` (probe set loss의 negative gradient)를 flatten하여 방향 벡터로 사용
4. Client model → `c_c`, Server model → `c_s`로 각각 계산

→ c는 "현재 모델에서 Q에 대해 loss를 줄이는 방향" = **전역 optimal 업데이트 방향**의 근사.

### 5.2 Client 업데이트 (Δ_i) 수집

| 프레임워크 | 방식 |
|-----------|------|
| SFL/USFL/MIX2SFL | 라운드 시작 시 global client model 스냅샷 → 라운드 종료 후 각 client model과의 차이 `Δ_i = x_i^{end} - x^{start}` |
| GAS | 비동기: 각 client의 **자기 시작점** 기준. `record_client_start_state()` → 학습 후 delta 계산 |
| MultiSFL | 각 branch의 시작점 기준. `record_branch_start_state()` → 학습 후 delta 계산 |

### 5.3 Experiment A 메트릭 계산 (매 라운드)

모든 기법이 동일한 공통 함수 `compute_experiment_a_metrics()` (`shared/experiment_a_metrics.py`)를 호출:

```
입력: client_deltas [Δ_1, ..., Δ_n], client_weights, c_c, c_s, server_delta Δ_s, server_steps K_s
```

1. **μ_c** = Σ(w_i · Δ_i) — 가중 합의 업데이트
2. **m2_c** = Σ(w_i · ‖Δ_i‖²) — 전체 에너지
3. **u2_c** = ‖μ_c‖² — 합의 신호 크기
4. **var_c** = m2_c - u2_c — 분산
5. **A_c_ratio** = var_c / (m2_c + ε)
6. **B_c** = 1 - cos(μ_c, c_c) — client 합의 vs probe 방향
7. **B_s** = 1 - cos(Δ_s, c_s) — server update vs probe 방향

### 5.4 G_drift 계산

| 프레임워크 | 측정 시점 | 방식 |
|-----------|----------|------|
| SFL/USFL/MIX2SFL | 매 client optimizer step 후 | `‖x^{t,b} - x^{t,0}‖²` 누적, 라운드 종료 시 `(1/\|P_t\|) Σ(S_n/B_n)` |
| GAS | 매 client optimizer step 후 (비동기 루프 내) | 동일 수식, 단 client를 1명씩 순차 처리 |
| MultiSFL | 매 branch client step 후 | 동일 수식, 단 기준점이 master model (branch 자체의 시작점이 아님) |

### 5.5 프레임워크 간 측정 차이 요약

| 항목 | SFL 프레임워크 (SFL/USFL/MIX2SFL) | GAS | MultiSFL |
|------|-----------------------------------|-----|----------|
| client 학습 | 병렬 (동시) | **비동기 (1명씩 순차)** | branch별 병렬 |
| Δ_i 기준점 | global model (모든 client 동일) | **각 client 자기 시작점** | **각 branch 자기 시작점** |
| G_drift 기준점 | global model | global model | **master model (≠ branch start)** |
| probe 방향 | 공통 c_c, c_s | 공통 + per-client c_{c,i} | 공통 + per-branch c_{c,i} |
| B_c/B_s 계산 | 공통 함수 | 공통 함수 | 공통 함수 |

→ **B_c, B_s는 cosine 기반이므로 scale-invariant하게 비교 가능**. 그러나 G_drift, G_drift_norm은 기준점과 학습 구조 차이로 **프레임워크 간 직접 비교 부적절**.

---

## 6. 전체 실험 결과

### 6.1 정확도 및 Experiment A 핵심 메트릭 (평균)

| Method | accuracy | A_c_ratio ↓ | B_c ↓ | B_s ↓ | C_c ↓ |
|--------|----------|-------------|-------|-------|-------|
| SFL IID | **0.599** | 0.793 | 0.992 | 0.987 | 0.997 |
| USFL | **0.534** | 0.819 | 1.001 | 0.981 | 1.000 |
| GAS | 0.376 | **0.550** | 0.976 | 0.965 | 0.988 |
| MultiSFL | 0.367 | 0.718 | 0.984 | **0.917** | 0.998 |
| MIX2SFL | 0.362 | 0.669 | **0.912** | 0.940 | **0.945** |
| SFL Non-IID | 0.324 | 0.791 | 1.023 | 0.982 | 1.009 |

### 6.2 Experiment A 보조 메트릭 (평균)

| Method | m2_c | u2_c (신호) | var_c (노이즈) | server_mag_per_step |
|--------|------|-------------|----------------|---------------------|
| SFL IID | 0.00729 | 0.000725 | 0.00656 | 0.000256 |
| USFL | 0.02042 | 0.00288 | 0.01753 | **0.00739** |
| GAS | 0.00327 | 0.00137 | 0.00191 | 0.00216 |
| MultiSFL | 0.01182 | 0.00332 | 0.00851 | 0.000153 |
| MIX2SFL | 0.00609 | 0.00204 | 0.00405 | 0.00205 |
| SFL Non-IID | 0.02250 | 0.00625 | 0.01625 | 0.000435 |

### 6.3 Drift 메트릭 (평균)

| Method | G_drift | G_drift_norm | G_drift_server | delta_client_norm_sq | delta_server_norm_sq |
|--------|---------|-------------|----------------|---------------------|---------------------|
| SFL IID | 0.00536 | 3.592 | 0.0211 | 0.000725 | 0.0312 |
| USFL | 0.01200 | 3.309 | 0.1196 | 0.00288 | 0.1743 |
| GAS | 0.00173 | 147.08 | 0.000904 | 0.000304 | 0.00209 |
| MultiSFL | 0.02167 | 38.54 | 0.0538 | 0.000743 | 0.00490 |
| MIX2SFL | 0.00188 | 1.197 | 0.0706 | 0.00204 | 0.0871 |
| SFL Non-IID | 0.01076 | 4.099 | 0.0407 | 0.00625 | 0.0715 |

### 6.4 기타 메트릭 (평균)

| Method | A_cos | M_norm | D_dir_client_weighted | D_rel_client_weighted | n_valid_alignment |
|--------|-------|--------|----------------------|----------------------|-------------------|
| SFL IID | 0.00354 | 0.0628 | 0.00656 | 4.870 | 10.43 |
| USFL | -0.00393 | 0.1306 | 0.01753 | 4.809 | 10.43 |
| GAS | NaN | 0.0256 | 0.00140 | 5.713 | — |
| MultiSFL | client: 0.0025, server: -0.034 | — | — | — | — |
| MIX2SFL | 0.01532 | 0.0693 | 0.00405 | 2.312 | 10.0 |
| SFL Non-IID | -0.05043 | 0.1239 | 0.01625 | 4.517 | 10.43 |

### 6.5 Worst-case Alignment (max B_c, max B_s)

| Method | accuracy | avg(B_c) | **max(B_c)** | avg(B_s) | **max(B_s)** |
|--------|----------|----------|-------------|----------|-------------|
| SFL IID | **0.599** | 0.992 | **1.047** | 0.987 | **1.019** |
| USFL | **0.534** | 1.001 | **1.119** | 0.981 | **1.014** |
| GAS | 0.376 | 0.976 | 1.187 | 0.965 | 1.178 |
| MultiSFL | 0.367 | 0.984 | 1.253 | 0.917 | 1.181 |
| MIX2SFL | 0.362 | **0.912** | **1.550** | 0.940 | 1.126 |
| SFL Non-IID | 0.324 | 1.023 | **1.483** | 0.982 | **1.359** |

---

## 7. 분석: 기존 가설 검증 결과

### 7.1 기존 가설 6단계 검증 (평균 메트릭 기준)

| # | 가설 | 검증 결과 | 근거 |
|---|------|-----------|------|
| 1 | 기존 연구 Non-IID 정확도 하락 | **✅ 성립** | SFL IID 0.599 vs Non-IID 전부 0.32~0.38 |
| 2 | Non-IID에서 drift가 나쁘다 | **△ 부분 성립** | SFL Non-IID G_drift(0.011) > SFL IID(0.005)이지만, MIX2SFL(0.002)은 IID보다 오히려 낮음 |
| 3 | Drift를 줄이면 정확도 올라간다 | **❌ 불성립** | MIX2SFL이 drift 최저(0.002)이나 정확도 하위권(0.362). USFL이 drift 최고(0.012)이나 정확도 1위(0.534). **역상관** |
| 4 | USFL이 gradient를 전역과 align | **❌ 불성립** | USFL A_c_ratio=0.819 (6위, 최하위), avg(B_c)=1.001 (5위) → 오히려 가장 나쁜 편 |
| 5 | USFL이 drift를 해결했다 | **❌ 불성립** | USFL G_drift(0.012) > SFL Non-IID(0.011). 오히려 더 높음 |
| 6 | USFL 정확도가 IID 수준 | **✅ 성립** | USFL 0.534, SFL IID 0.599로 Non-IID 중 압도적 1위 |

→ **가설 1, 6은 데이터로 확인됨. 그러나 핵심 인과 논리인 3, 4, 5가 불성립.**

### 7.2 평균 메트릭이 정확도를 설명하지 못하는 근거

**평균 A_c_ratio 순위** (↓ better):
```
GAS(0.550) < MIX2SFL(0.669) < MultiSFL(0.718) < SFL(0.791) ≈ SFL_IID(0.793) < USFL(0.819)
```
→ USFL이 **최하위**. 정확도와 상관관계 없음.

**평균 B_c 순위** (↓ better):
```
MIX2SFL(0.912) < GAS(0.976) < MultiSFL(0.984) < SFL_IID(0.992) < USFL(1.001) < SFL(1.023)
```
→ MIX2SFL이 최고이나 정확도 하위권. 정확도와 상관관계 없음.

**G_drift 순위** (↓ better):
```
GAS(0.002) < MIX2SFL(0.002) < SFL_IID(0.005) < SFL(0.011) < USFL(0.012) < MultiSFL(0.022)
```
→ USFL drift가 두 번째로 높음. 상관관계 없음.

**결론: 평균 기반 drift/alignment 메트릭 중 정확도를 설명하는 것이 없음.**

### 7.3 전환점: max(B_c) ↔ 정확도 대응 (이거로 논리를 만들 수도 있지 않을까?)

**max(B_c) 순위와 정확도 순위 비교**:
```
max(B_c):  1.047 < 1.119 < 1.187 < 1.253 < 1.483 < 1.550
accuracy:  0.599   0.534   0.376   0.367   0.324   0.362
method:    IID     USFL    GAS     Multi   SFL     MIX2SFL
```
→ 6개 중 5개 완벽 대응 (MIX2SFL/SFL만 소폭 역전)

**max(B_s) 순위**:
```
max(B_s):  1.014 < 1.019 < 1.126 < 1.178 < 1.181 < 1.359
method:    USFL    IID     MIX2SFL  GAS    Multi   SFL
```
→ USFL의 max(B_s)가 SFL IID보다도 낮음

### 7.4 MIX2SFL 역설

MIX2SFL은 평균 메트릭에서 최고이나, worst-case에서 최악:
- avg(B_c) = **0.912** (6개 중 최저 = 최고)
- max(B_c) = **1.550** (6개 중 최고 = 최악)

→ SmashMix(activation mixup) + GradMix가 평균적으로는 정렬을 개선하지만, 간헐적으로 극단적 misalignment 발생

### 7.5 왜 max가 avg보다 중요한가

- B_c > 1인 라운드 = **"alignment reversal"** (합의 업데이트가 optimal **반대 방향**으로 이동)
- 이 reversal이 여러 라운드의 학습 진행을 소거할 수 있음
- 평균이 좋아도 극단적 reversal이 있으면 학습이 불안정
- Optimization 이론에서 worst-case bound가 수렴 속도를 결정하는 것과 일맥상통

### 7.6 기존 가설 재검증 (max 기준)

| # | 가설 | max 기준 검증 | 근거 |
|---|------|--------------|------|
| 1 | 기존 연구 Non-IID 정확도 하락 | **✅ 성립** | (동일) |
| 2 | Non-IID에서 alignment가 나쁘다 | **✅ 성립** | max(B_c): SFL IID(1.05) vs Non-IID 전부 1.19~1.55 |
| 3 | Worst-case alignment를 낮추면 정확도 올라간다 | **✅ 거의 성립** | max(B_c) 순위 ↔ 정확도 순위 5/6 대응 |
| 4 | USFL이 alignment를 안정화했다 | **✅ 성립** | max(B_c)=1.12, max(B_s)=1.01 → IID 수준 |
| 5 | USFL이 worst-case reversal을 억제했다 | **✅ 성립** | max(B_c) IID 대비 gap 0.07로 6개 중 최소 |
| 6 | USFL 정확도가 IID 수준 | **✅ 성립** | (동일) |

→ **평균 기준으로는 3,4,5가 불성립했으나, worst-case(max) 기준으로 전환하면 6개 가설 모두 성립.**

---

## 8. 현재 결과에서 가능한 논문 논리 (안)

max(B_c) 기반으로 재구성하면:

1. **문제 정의**: Non-IID + partial participation 환경의 SFL에서, client 합의 업데이트가 간헐적으로 전역 최적 방향의 반대로 이동하는 **alignment reversal** 현상이 발생한다.

2. **기존 연구의 한계**: MIX2SFL 등 기존 기법은 activation mixup(SmashMix) + gradient mixing(GradMix)으로 평균 alignment를 개선하지만, worst-case reversal은 오히려 악화 (max B_c = 1.55). GAS(generative activation + local adjustment), MultiSFL(multi-branch + knowledge replay)도 max B_c가 1.19~1.25로 불안정.

3. **USFL의 접근**: Activation balancing (concat) + gradient shuffle로 split-layer에서의 gradient batch를 전역 분포에 맞추어, worst-case alignment를 IID 수준으로 억제 (max B_c = 1.12 vs IID 1.05, max B_s = 1.01 vs IID 1.02).

4. **결과**: Alignment stability가 보장됨으로써, Non-IID에서도 SFL IID에 근접하는 정확도 달성 (0.534 vs 0.599).

---

## 9. 보충 정보

### 9.1 max(B_c) 발생 시점

| Method | max(B_c) | 발생 라운드 | 비고 |
|--------|----------|------------|------|
| SFL IID | 1.047 | R253 | 후기 |
| USFL | 1.119 | R187 | 중기 |
| GAS | 1.187 | R232 | 후기 |
| MultiSFL | 1.253 | R72 | 초기 |
| MIX2SFL | 1.550 | R43 | **초기에 극단적 reversal** |
| SFL Non-IID | 1.483 | R207 | 후기 |

### 9.2 GAS/MultiSFL G_drift_norm 이상치 설명

| Method | G_drift_norm | 이유 |
|--------|-------------|------|
| GAS | **147.08** | 비동기 학습 구조: client를 1명씩 순차 처리하여 라운드당 파라미터 이동(G_drift)은 작으나, global update norm(분모)이 극소 → 비율 폭등 |
| MultiSFL | **38.54** | drift 기준점이 master model: branch는 soft_pull(α=0.1)로 master와 상시 offset 유지. drift에 학습 drift + branch-master divergence 포함 → 과대 측정 |

→ 두 값 모두 구조적 차이에서 기인. Experiment A 메트릭(B_c, B_s)은 cosine 기반 scale-invariant이므로 기법 간 비교 가능.

---

## 10. 핵심 고민 사항 (교수님께 상의)

### 고민 1: 실험의 신뢰성 — 측정이 제대로 된 것인가?

현재 모티베이션 실험(Experiment A)의 결과를 신뢰할 수 있는지 우려됨:

- **평균 메트릭 전면 실패**: Experiment A의 메인 메트릭(A_c_ratio, B_c, B_s)의 평균값이 정확도와 **전혀 상관관계가 없음**. 논문 리포팅 가이드에서 권장하는 `mean_t(A_c_ratio)`, `mean_t(B_c)`, `mean_t(B_s)`가 모두 기법 간 변별력이 없음.
- **모든 B_c ≈ 1.0**: 6개 기법 전부 avg(B_c)가 0.91~1.02 범위. cos(μ_c, c_c) ≈ 0, 즉 모든 기법의 합의 업데이트가 probe 방향과 거의 직교. **probe 방향 자체가 의미있는 기준인지** 의문.
- **GAS/MultiSFL 구조적 차이**: G_drift_norm이 147, 38.5 등 비정상적 수치. 비동기/multi-branch 구조에서의 drift 측정이 SFL 프레임워크와 동일 기준으로 비교 가능한 것이 맞는지.
- **Single seed**: 모든 결과가 1회 실험. max(B_c) 상관관계가 우연일 가능성 배제 불가.

→ **질문**: 이 실험 결과 자체를 신뢰하고 논문에 사용해도 되는 것인지, 아니면 측정 방법론을 재검토해야 하는지?

### 고민 2: 현재 결과로 논리를 만들 수 있는가?

평균 메트릭은 실패했으나, **max(B_c)에서 정확도와의 상관관계를 발견**함 (Section 7.3, 7.6 참조). 이를 기반으로 "alignment reversal" 프레이밍이 가능할 수 있음 (Section 8 참조).

**핵심 질문: 300라운드 중 max 1개 값에 논문 모티베이션을 거는 것이 학술적으로 허용 가능한 논리인가?**

추가 우려:
- max만으로는 약하다면, B_c > 1 빈도, 시계열 패턴, 상위 percentile 등 보강 분석이 필요할 수 있음.
- **인과관계 부재**: USFL의 어떤 구성요소(activation balancing? grad shuffle?)가 max(B_c)를 낮추는지 ablation이 없음.
- **5/6 대응**: MIX2SFL(max 1.55, acc 0.362)과 SFL Non-IID(max 1.48, acc 0.324)의 순서가 역전. 완벽한 상관이 아님.

→ **질문**: max(B_c) 기반 "alignment reversal" 논리가 충분히 강한 모티베이션이 될 수 있는지? 추가로 어떤 실험/분석이 있으면 보강 가능한지?

### 고민 3: Model drift 프레이밍 자체를 포기해야 하는가?

만약 현재 Experiment A 결과가 논문 모티베이션으로 부족하다면, 근본적으로 다른 방향을 고려해야 할 수 있음:

**대안 A: Split-layer activation imbalance 관점**
- SFL에서 Non-IID의 진짜 병목은 parameter drift가 아니라, server가 받는 activation batch의 class imbalance
- USFL의 activation concatenation이 이를 직접 해결
- 새로운 메트릭 필요: split-layer에서의 class distribution entropy 등

**대안 B: Server-side gradient quality 관점**
- Non-IID에서 server가 편향된 activation batch로 학습 → server gradient가 biased
- USFL server_mag_per_step이 타 기법 대비 3~17배 큰 것이 근거 (0.00739 vs GAS 0.00216, MIX2SFL 0.00205, SFL 0.000435)
- 새로운 메트릭 필요: server gradient의 class별 분해 등

**대안 C: 모티베이션 실험 없이 성능 중심 논문**
- drift/alignment 분석을 부록으로 빼고, 정확도 개선 자체를 핵심 contribution으로
- 단, 이 경우 "왜 USFL이 잘 되는지" 설명력이 약해짐

→ **질문**: model drift/alignment 분석을 계속 밀고 나갈지, 아니면 다른 관점의 모티베이션 실험으로 전환할지?

### 다음 단계 제안 (우선순위)

| 우선순위 | 항목 | 목적 |
|----------|------|------|
| 1 | **교수님 방향 확정** | 위 3가지 고민에 대한 판단 |
| 2 | B_c > 1 빈도 + 시계열 시각화 | max만이 아닌 전체 reversal 패턴 정량화 |
| 3 | Multiple seeds (3~5회) | 상관관계 통계적 검증 |
| 4 | Ablation (USFL w/o grad shuffle) | 인과관계 확인 |
| 5 | 다른 α (0.1, 1.0) | 일반화 검증 |
