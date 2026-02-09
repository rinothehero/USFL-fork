# USFL vs Baseline SFL류: Drift 정량화 실험 스펙 (Experiment A / B)

목표 : **부분참여(10/100) + non-IID + 기법별 동역학 차이(gradient scale/초기상태/server step 수)**가 있어도, **기법 간 drift 개선 정도를 정합적으로 비교**하는 것.

---

## 0. 핵심 목표와 문제 정의

### 목표

USFL이 SFL iid / SFL non-iid / GAS / Mix2SFL / MultiSFL 대비:

1. **Client drift(참여 클라 업데이트 불일치)**를 줄이는지
2. **참여자 합의 업데이트가 모집단(전역) 방향과 더 잘 정렬**되는지
3. (SFL 특성상 중요한) **Server drift**도 악화시키지 않는지
   를 **정량 지표로** 제시.

### 핵심 난제(이미 겪는 문제)

* norm² 기반 지표는 **gradient scale이 작은 기법**에서 자동으로 작아져 “좋아 보이는 착시” 발생
* 기법마다

  * client 전달/적용 gradient scale: `1/B_i` vs `1/sumB` vs `1/(B_i+R_i)`
  * round-start client 상태: 동기(global 동일) vs 비동기(slot/branch persistent)
  * server update 횟수: `n`회 vs `1`회 vs `n*steps_to_run`
    가 달라 **라운드 단위 Δ나 norm²를 그대로 비교하면 동역학 차이를 섞어버림**

따라서 **Experiment A(실제 학습 경로)**와 **Experiment B(고정 체크포인트에서 정합 비교)**를 분리하고, 지표는 **스케일 불변(scale-invariant)** 형태로 설계한다.

---

## 1. 공통 정의(Notation) 및 설계 원칙

### 참여자 집합/가중치

* 전체 클라이언트 (N=100)
* 라운드 (t) 참여자 (P_t), (|P_t|=n=10)

가중치 (w_i)는 “기법 간 정합성”을 위해 기본은 **per-sample 가중치** 권장:
[
w_i = \frac{B_i}{\sum_{j\in P_t} B_j}
]

* (B_i): 라운드 (t)에서 클라이언트 i가 사용한 **main 배치 샘플 수**(또는 main 처리 샘플 수)
* MultiSFL에서 replay (R_i)가 있어도 **client drift(클라 모델 업데이트)** 중심이면 (B_i) 기반이 기본값으로 더 안전함
  (단, replay가 “서버 업데이트”에 미치는 영향 분석을 별도로 하려면 (B_i+R_i) 기반 가중치도 보조로 같이 기록)

> **필수 로그**: ({B_i}, \sum B, {R_i}) (해당 시)

### 파라미터 블록

* 클라이언트 모델 파라미터: (x_c)
* 서버 모델 파라미터: (x_s)
* 풀 모델: (x = (x_c, x_s))

### 업데이트(상태 변화)

클라이언트 i:
[
\Delta_{c,i}^t = x_{c,i}^{t,\text{end}} - x_{c,i}^{t,0}
]
서버:
[
\Delta_s^t = x_s^{t,\text{end}} - x_s^{t,0}
]

합의 업데이트(참여자 평균):
[
\mu_c^t = \sum_{i\in P_t} w_i \Delta_{c,i}^t
]

> **주의(비동기/persistent 방법)**: GAS/MultiSFL처럼 (x_{c,i}^{t,0})가 클라별로 서로 다를 수 있음.
> 이 경우에도 (\Delta_{c,i}^t)는 “같은 파라미터 공간에서의 변위 벡터”이므로 계산은 가능하지만, 해석이 (동기 시작 대비) 약해질 수 있어 **Experiment B로 보완**한다.

---

## 2. “중앙/모집단 방향” 정의: probe set 기반 (c)

“전역 방향”을 매 라운드 **전체 데이터 full gradient**로 정의하면 오라클 비판/비용 문제가 생길 수 있으므로, **고정 probe set (Q)**로 중앙 방향을 정의한다.

라운드 (t)의 기준 상태(권장: 라운드 시작 글로벌 상태):
[
(x_c^{t,0}, x_s^{t,0})
]

probe set 손실 (L_Q)에 대해:
[
g_{c,\text{probe}}^t = \nabla_{x_c} L_Q(x_c^{t,0}, x_s^{t,0})
]
[
g_{s,\text{probe}}^t = \nabla_{x_s} L_Q(x_c^{t,0}, x_s^{t,0})
]

내려가는 방향(central direction):
[
c_c^t = -g_{c,\text{probe}}^t,\qquad c_s^t = -g_{s,\text{probe}}^t
]

> **probe set 설계 권장**

* 모든 기법에서 동일한 (Q) 사용(고정)
* 매 라운드 전체 (Q)를 쓰기 부담이면 “고정된 mini-probe 배치”들을 사용하되 시드 고정
* dropout/augmentation 등 랜덤 요소는 probe 계산 시 **비활성화**(또는 시드 고정)해서 방향 노이즈를 줄임

---

## 3. Drift 메트릭 설계: 스케일 문제를 피하는 “정석 세트”

norm² 자체는 스케일에 취약하므로, 실험 A에서는 **크기(스케일) vs 정렬(드리프트)**를 분리해 보고해야 한다.

### 3.1 Client-side: 참여자 내부 drift (Metric A)

두 번째 모멘트:
[
m_{2,c}^t = \sum_{i\in P_t} w_i |\Delta_{c,i}^t|^2
]
합의 크기:
[
u_{2,c}^t = |\mu_c^t|^2
]
분산(불일치):
[
\mathrm{Var}*{c}^t
= \sum*{i\in P_t} w_i|\Delta_{c,i}^t - \mu_c^t|^2
= m_{2,c}^t - u_{2,c}^t
]

**(A-메인) 스케일 불변 drift ratio (강력 권장)**
[
A_{c,\text{ratio}}^t
= \frac{\mathrm{Var}*c^t}{m*{2,c}^t+\epsilon}
= 1 - \frac{u_{2,c}^t}{m_{2,c}^t+\epsilon}
]

* 범위가 대체로 ([0,1])에 가깝고 안정적
* 업데이트가 전체적으로 (s)배 작아져도(gradient scale 차이) 값이 거의 불변

**(A-보조) 합의 대비 상대 drift**
[
A_{c,\text{rel}}^t=\frac{\mathrm{Var}*c^t}{u*{2,c}^t+\epsilon}
]

* 해석은 직관적이지만 (\mu\approx 0)에서 튈 수 있음 → ratio를 메인으로, rel은 보조로 권장

> **함께 로깅 필수**: (m_{2,c}^t), (u_{2,c}^t)
> 그래야 “그냥 업데이트 억제해서 drift가 작아진 것”을 반박/분리 가능.

---

### 3.2 Client-side: 합의 업데이트가 중앙 방향과 정렬되는가 (Metric B)

스케일 불변으로 **코사인 미스얼라인** 사용:
[
B_c^t
= 1 - \frac{\langle \mu_c^t,; c_c^t\rangle}{|\mu_c^t|;|c_c^t|+\epsilon}
]

* 0에 가까울수록 “중앙 방향과 정렬”
* 2에 가까울수록 반대 방향(매우 나쁨)

---

### 3.3 Client-side: 개별 업데이트가 중앙 방향에서 얼마나 벗어나는가 (Metric C)

개별 정렬 오차(코사인 기반):
[
C_c^t
= \sum_{i\in P_t} w_i\left(
1 - \frac{\langle \Delta_{c,i}^t,; c_c^t\rangle}{|\Delta_{c,i}^t|;|c_c^t|+\epsilon}
\right)
]

* “outlier 업데이트/개별 불일치”를 직접 보여줌
* B는 합의(평균) 수준, C는 개별 수준

---

### 3.4 Server-side: 서버 업데이트의 중앙 정렬(B 변형) + step 정규화

서버 업데이트 정렬:
[
B_s^t
= 1 - \frac{\langle \Delta_s^t,; c_s^t\rangle}{|\Delta_s^t|;|c_s^t|+\epsilon}
]

서버 step 수(optimizer.step count) (K_s^t)를 반드시 로깅하고, 규모 비교는 정규화해서:
[
\text{ServerMagPerStep}^t = \frac{|\Delta_s^t|^2}{K_s^t+\epsilon}
\quad(\text{또는 } |\Delta_s^t|/(K_s^t+\epsilon))
]

> 서버는 기법별 step budget이 크게 다르므로,
> **방향 비교는 cosine(B_s)**, **규모 비교는 per-step(또는 per-sample) 정규화**가 기본.

---

### 3.5 Full-model(클라+서버 합친) drift (선택/확장)

풀 업데이트/중앙 방향:
[
\mu_{\text{full}}^t = \begin{bmatrix}\mu_c^t\ \Delta_s^t\end{bmatrix},\qquad
c_{\text{full}}^t = \begin{bmatrix}c_c^t\ c_s^t\end{bmatrix}
]
정렬:
[
B_{\text{full}}^t
= 1 - \frac{\langle \mu_{\text{full}}^t,; c_{\text{full}}^t\rangle}{|\mu_{\text{full}}^t|;|c_{\text{full}}^t|+\epsilon}
]

> 더 정교하게 하려면 SFL처럼 서버가 client별로 업데이트될 때 **클라 i에 대응하는 서버 업데이트 조각 (\Delta_{s,i})**를 기록해서
> (\Delta_{\text{full},i}=[\Delta_{c,i};\Delta_{s,i}])로 A/C까지 확장 가능.
> (USFL/Mix2SFL은 서버 업데이트 1회이므로 (\Delta_{s,i}\approx w_i\Delta_s) 같은 분배 가정도 가능하지만, 이는 “보조 분석” 권장)

---

## 4. Experiment A 스펙: “각 기법을 독립적으로 학습”하면서 drift 측정

### 4.1 목적

* 실제 운영(학습) 경로에서 **USFL이 baseline 대비 drift를 줄이는지**
* 단, gradient scale/step count 차이가 있으므로 **스케일 불변 지표(A_ratio/B/C)** 중심으로 평가

---

### 4.2 실험 조건(정합성 확보)

* **동일한 클라이언트 샘플링 스케줄 (P_t)** 을 모든 기법에 강제(가능하면 시드 고정)
* 데이터로더 시드/셔플 정책도 최대한 동일화
* 평가(accuracy/loss)도 동일 주기로 기록
* dropout/augmentation은 학습에서는 각 기법 동일 조건 유지(측정 시에는 필요시 고정/비활성화 옵션 별도)

---

### 4.3 라운드 (t)에서 수집해야 하는 것(정확히)

#### (1) Round-start 상태

* 글로벌(또는 master) 기준:

  * (x_c^{t,0}), (x_s^{t,0})

* **비동기/persistent 방법(GAS, MultiSFL)**은 필수로 추가:

  * 각 참여자 i의 시작 상태 (x_{c,i}^{t,0}) (slot/branch 포함)

#### (2) 각 참여 클라 i의 업데이트(클라 모델)

* 라운드 종료 시 상태 (x_{c,i}^{t,\text{end}})
* 업데이트 벡터 (\Delta_{c,i}^t = x_{c,i}^{t,\text{end}} - x_{c,i}^{t,0})
* 샘플 수 (B_i) (+ MultiSFL이면 (R_i))

#### (3) 서버 업데이트

* (x_s^{t,\text{end}}), (\Delta_s^t)
* 서버 optimizer.step 횟수 (K_s^t)
* (선택/진단) 서버 trajectory drift(현재 구현한 S/B 형태)

#### (4) 중앙 방향(Probe)

* probe set (Q)로부터 (c_c^t, c_s^t) 계산

  * 권장: 라운드 시작 상태 ((x_c^{t,0}, x_s^{t,0}))에서 계산
  * 비동기 방법에서 “시작 상태가 서로 다른데 중앙 방향이 하나여도 되냐?” 문제가 있으면:

    * **A-동기 기법(SFL/USFL/Mix2SFL)**: 그대로 (c^t) 사용
    * **A-비동기 기법(GAS/MultiSFL)**: 두 버전 모두 로깅 권장

      1. anchor 중앙: (c^t)를 master/global에서 계산
      2. per-client 중앙: (c_{c,i}^t = -\nabla_{x_c}L_Q(x_{c,i}^{t,0},x_s^{t,0}))를 계산하고, 개별 정렬(C)을 per-client 기준으로도 기록
         (논문에는 보통 anchor 기준을 main으로, per-client 기준을 보조로 제시하면 깔끔함)

---

### 4.4 Experiment A에서 계산할 메트릭(요약)

클라이언트:

* (m_{2,c}^t), (u_{2,c}^t), (\mathrm{Var}_c^t)
* **Metric A(main):** (A_{c,\text{ratio}}^t=\mathrm{Var}*c^t/(m*{2,c}^t+\epsilon))
* **Metric B:** (B_c^t=1-\cos(\mu_c^t,c_c^t))
* **Metric C:** (C_c^t=\sum w_i(1-\cos(\Delta_{c,i}^t,c_c^t)))

서버:

* **Metric B_s:** (B_s^t=1-\cos(\Delta_s^t,c_s^t))
* (|\Delta_s^t|^2/(K_s^t+\epsilon)) (또는 (|\Delta_s^t|/(K_s^t+\epsilon)))
* (진단) 서버 trajectory drift

풀 모델(선택):

* (B_{\text{full}}^t=1-\cos(\mu_{\text{full}}^t,c_{\text{full}}^t))

---

### 4.5 결과 집계/리포팅

* 라운드별 시계열 곡선 + 평균/표준편차/신뢰구간
* **seed 여러 개 평균**(부분참여 샘플링 분산이 크므로 권장)
* “드리프트 감소” 주장에는 반드시:

  * 성능(accuracy/loss) 유지/개선
  * 스케일 지표((m_{2,c}^t), (u_{2,c}^t)) 함께 제시
    를 같이 묶어서 보고

---

## 5. Experiment B 스펙: “체크포인트 고정 + 동일 조건에서 정합 비교”

### 5.1 목적

* 기법마다 학습 경로/상태가 달라서 A에서 drift가 달라 보이는 문제를 제거
* **동일한 모델 상태 (x^{(k)})** 에서 기법의 메커니즘이 만들어내는 drift를 비교
* 동역학 차이(서버 step 수, optimizer 상태 등)의 영향 최소화

---

### 5.2 체크포인트(고정 상태) 구성

* 체크포인트 집합 ({x^{(1)},\dots,x^{(M)}})
* 생성 방식(권장 우선순위):

  1. 특정 기준 학습(run)에서 일정 간격으로 저장한 체크포인트(가장 단순/재현성 좋음)
  2. 여러 기법 체크포인트를 섞되 동일 set을 모든 방법에 공유

---

### 5.3 “무엇을 수집할 것인가?” — 실험 B의 추천은 **1-step gradient 기반**

실험 B는 “정합 비교”가 목표이므로, **optimizer/LR 차이를 빼기 위해 gradient를 직접 캡처**하는 게 가장 깔끔함.

#### 고정 체크포인트 (x^{(k)})에서

* 동일 참여자 집합 (P) (10명) 고정
* 각 클라 i에 대해 동일 배치(또는 동일 샘플 인덱스/시드) 고정
* dropout/augmentation 비활성화 또는 시드 고정(측정 노이즈 최소화)

#### 수집 항목(클라이언트)

기법이 실제로 “클라이언트 모델 파라미터에 적용하게 만드는” gradient:
[
g_{c,i}^{(k)} \equiv \text{(method에 의해 유도된)};\nabla_{x_c}\ell_i(x^{(k)})
]

* USFL/Mix2SFL: concat CE backward → slice → client backward 후 client param grad가 형성됨
  → **“client optimizer.step에 들어가기 직전의 client param gradient”**를 캡처
* SFL: 클라 배치별 server backward 후 cut-grad 전달 → client backward
  → 마찬가지로 client param grad 캡처
* GradMix broadcast 등은 “실제 전달된 grad 형태”가 다르므로, 그 **적용 결과 grad**를 캡처

합의 gradient:
[
\mu_{g,c}^{(k)} = \sum_{i\in P} w_i g_{c,i}^{(k)}
]

#### 수집 항목(서버)

* 서버 파라미터 gradient (g_s^{(k)}) (method가 사용하는 손실 기준)
* 가능하면 “main loss grad” vs “추가 loss(예: SmashMix) 기여”를 분리 로깅(설득력↑)

#### 중앙 방향(Probe)

* 동일 체크포인트에서 probe set (Q)로:
  [
  c_c^{(k)}=-\nabla_{x_c}L_Q(x^{(k)}),\qquad c_s^{(k)}=-\nabla_{x_s}L_Q(x^{(k)})
  ]

> **왜 B는 gradient가 적절한가?**
>
> * LR/optimizer state/server update count 차이를 최소화
> * “기법이 만들어낸 전달/스케일/결합 구조”의 효과를 같은 (x)에서 비교 가능

---

### 5.4 Experiment B 메트릭(gradient 버전)

Experiment A의 수식을 그대로 쓰되 (\Delta\rightarrow g).

* 두 번째 모멘트:
  [
  m_{2,g}^{(k)} = \sum_{i\in P} w_i |g_{c,i}^{(k)}|^2
  ]
* 합의 크기:
  [
  u_{2,g}^{(k)}=|\mu_{g,c}^{(k)}|^2
  ]
* 분산:
  [
  \mathrm{Var}*g^{(k)}=m*{2,g}^{(k)}-u_{2,g}^{(k)}
  ]
* **Metric A(main):**
  [
  A_{g,\text{ratio}}^{(k)}=\frac{\mathrm{Var}*g^{(k)}}{m*{2,g}^{(k)}+\epsilon}
  ]
* **Metric B:**
  [
  B_{g,c}^{(k)}=1-\cos(\mu_{g,c}^{(k)},c_c^{(k)})
  ]
* **Metric C:**
  [
  C_{g,c}^{(k)}=\sum_i w_i\left(1-\cos(g_{c,i}^{(k)},c_c^{(k)})\right)
  ]

서버도 동일하게:
[
B_{g,s}^{(k)}=1-\cos(g_s^{(k)},c_s^{(k)})
]

---

### 5.5 결과 집계

* 체크포인트별 값의 평균/분산(또는 체크포인트 순서에 따른 곡선)
* 필요하면 “학습 진행도(accuracy/loss 구간)”별로 체크포인트를 그룹화해서 비교

---

## 6. 스케일/동역학 차이를 “보완”하는 scaling 실험을 어떻게 넣을까?

### 결론(논문 전략)

* grad scaling / server LR scaling은 **메인 비교(기법 자체 비교)**로 쓰기보다
  **통제/ablation 실험**으로 넣는 게 가장 안전하고 설득력이 높다.
* 이유: baseline을 “다른 알고리즘”으로 바꿔버렸다는 공격을 피하면서도,
  “차이가 단순 스케일/스텝 budget 때문이 아니다”를 증명할 수 있음.

---

### 6.1 grad scaling 통제(예시)

목표: `1/B_i` vs `1/sumB` 스케일 차이가 drift 지표를 왜곡하는지 분해.

* per-client grad가 기본 `~1/B_i`로 들어오는 방법(SFL/GAS류)에서, per-sample 평균 스케일(`~1/sumB`)에 맞추려면:
  [
  g_{i,\text{scaled}} = \frac{B_i}{\sum_{j\in P}B_j}; g_i
  ]
  (또는 동일 효과로 client LR에 (\frac{B_i}{\sum B})를 곱함)

이 scaling을 켜고/끄고 Experiment B(고정 체크포인트)에서 A/B/C를 비교하면:

* “스케일이 작아서 drift가 좋아 보였는지”
* “구조적으로 정렬이 좋아진 건지”
  를 분해 가능.

---

### 6.2 server LR scaling 통제(예시)

목표: server step 횟수 차이(`n`회 vs `1`회)가 per-round 업데이트 규모를 달리 만드는 문제 분해.

소규모 스텝 근사에서, 동일한 “라운드 총 업데이트 규모”를 맞추려면 대략:

* SFL처럼 라운드당 서버 step이 (n)회면:
  [
  \eta_{s,\text{SFL}} \approx \frac{1}{n}\eta_{s,\text{USFL}}
  ]
* MultiSFL처럼 (n\cdot steps_to_run)회면:
  [
  \eta_{s,\text{Multi}} \approx \frac{1}{n\cdot steps_to_run}\eta_{s,\text{USFL}}
  ]

이 역시 **통제 실험**으로만 제시하는 걸 권장.
(“서버가 더 많이 step을 밟는 게 그 방법의 설계”일 수도 있으니까)

---

## 7. 구현/로깅 체크리스트(실수 방지)

### 7.1 파라미터 공간 일치

* (\Delta), (c) 계산에 쓰는 파라미터 집합이 반드시 동일해야 함

  * trainable만? 전체 named_parameters?
    → 한 번 결정하면 끝까지 통일
* mixed precision이면 norm/내적 누적은 float64로(가능하면) 안정화

### 7.2 “벡터 저장” 없이 계산하기(권장)

m=10이라도 파라미터가 수백만이면 벡터 저장이 부담일 수 있어.
필요한 것은 대부분 **norm²와 dot**이므로 스트리밍으로 계산 가능:

* (|\Delta_i|^2), (|\mu|^2), (\langle \Delta_i,\mu\rangle), (\langle \mu,c\rangle) 등만 누적
* cosine을 위해 (|\cdot|)과 dot만 저장하면 됨

### 7.3 부분참여 분산 제어

* Experiment A: 동일 (P_t) 스케줄 강제 + seed 여러 개 평균 권장
* Experiment B: 참여자 (P), 배치 샘플까지 고정

### 7.4 (\epsilon) 설정

* (\mu)가 매우 작을 때 분모가 폭발하지 않도록 사용
* early rounds의 (|\mu|^2) 중앙값 기반으로 adaptive epsilon 쓰는 것도 OK(너희 코드 방향과 일치)

---

## 8. 이 스펙이 “기법 간 drift 해결 정도”를 좋은 지표로 비교하는가?

**Yes — 다만 아래 조건을 충족하면 강해진다.**

* A에서 **스케일 지표((m_2,u_2)) + 스케일 불변 drift(A_ratio/B/C)**를 함께 제시
  → “업데이트 억제” 반박 방지
* B에서 **동일 체크포인트/동일 배치**로 gradient 기반 A/B/C 제시
  → “모델 상태가 달라서 생긴 착시” 반박 방지
* 최종적으로 성능(accuracy/loss)도 함께 제시
  → “드리프트만 줄고 학습이 느려진 것” 반박 방지

이 조합이면 “USFL이 drift를 개선한다”는 모티베이션으로 쓰기에 상당히 탄탄해진다.

---

## 9. (권장) 최종 리포팅 구성 예시

### 메인(Experiment A)

* 라운드별:

  * (A_{c,\text{ratio}}^t), (B_c^t), (C_c^t)
  * (m_{2,c}^t), (u_{2,c}^t)
  * 서버: (B_s^t), (|\Delta_s^t|^2/(K_s^t+\epsilon))
  * 성능 곡선

### 정합(Experiment B)

* 체크포인트 평균:

  * (A_{g,\text{ratio}}^{(k)}), (B_{g,c}^{(k)}), (C_{g,c}^{(k)})
  * 서버 (B_{g,s}^{(k)})
* (선택) grad scaling / server LR scaling 통제 실험 결과 1–2개

---

실제 코드에서 **각 방법별로 “어느 텐서를 어디에서 캡처해야 (g_{c,i})가 ‘실제로 적용되는 gradient’인지”**가 제일 중요해(특히 Mix2SFL의 GradMix reduce=sum/mean, MultiSFL replay 포함).
방법별( SFL / USFL / GAS / Mix2SFL / MultiSFL )로 **캡처 지점 체크리스트(forward/backward/step 전후)**을 주의해서 봐야해.
