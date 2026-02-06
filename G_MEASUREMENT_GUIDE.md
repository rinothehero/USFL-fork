# G-Measurement Complete Guide

## Overview

G-Measurement는 **현재 학습 상태(g̃)와 이상적인 Oracle 상태(g*)의 gradient 거리**를 측정하는 시스템입니다. 세 가지 구현체에서 일관된 프로토콜로 사용됩니다:

- **GAS_implementation**: G-score 기반 클라이언트 선택
- **sfl_framework-fork**: USFL 및 SFL 연구용 측정
- **multisfl_implementation**: Multi-branch SFL 성능 평가

## Core Metrics

### 1. G (Absolute Gradient Distance)

```
G = ||g̃ - g*||² = Σ(g̃ᵢ - g*ᵢ)²
```

**의미**: 현재 gradient와 Oracle gradient의 L2 노름 제곱

### 2. G_rel (Relative Gradient Distance)

```
G_rel = G / (||g*||² + ε)
```

**의미**: Oracle gradient의 크기로 정규화한 상대적 거리

### 3. D_cosine (Cosine Distance)

```
D_cosine = 1 - cos(g̃, g*) = 1 - (g̃ · g*) / (||g̃|| × ||g*||)
```

**의미**: Gradient 방향의 차이 (0 = 동일 방향, 2 = 반대 방향)

### 4. Variance_G (Client/Server Variance)

**핵심 질문**: 여러 클라이언트/서버의 gradient가 Oracle로부터 얼마나 분산되어 있는가?

#### Variance_Client_G 계산

```
Vc = Σᵢ wᵢ × ||g̃ᵢ - g*||²

where:
  i = client index
  wᵢ = weight (normalized by Σwᵢ)
  g̃ᵢ = client i의 gradient (현재 측정값)
  g* = Oracle client gradient (이상적인 값)
```

**직관**: 각 클라이언트의 gradient가 Oracle로부터 얼마나 떨어져 있는지의 가중 평균

#### Variance_Client_G_Rel 계산

```
Vc_rel = Vc / (||g*||² + ε)
```

**의미**: 상대적으로 정규화된 Client variance

#### Variance_Server_G 계산

```
Vs = Σⱼ wⱼ × ||g̃ⱼ - g*||²

where:
  j = server batch/branch index
  wⱼ = weight (normalized by Σwⱼ)
  g̃ⱼ = server의 j번째 gradient
  g* = Oracle server gradient
```

#### Variance_Server_G_Rel 계산

```
Vs_rel = Vs / (||g*||² + ε)
```

---

## Complete Measurement Workflow

### Phase 1: Oracle Gradient 계산 (이상적인 참조값)

**목표**: 전체 데이터셋을 사용해 "이상적인" gradient를 계산

**프로토콜**:
```python
# 1. BN stats 백업
bn_backup = backup_bn_stats(model)

# 2. Train 모드 유지 (학습과 동일한 조건)
model.train()

# 3. 전체 데이터셋 순회
total_samples = 0
grad_accum = {}

for batch in full_dataloader:
    data, labels = batch
    model.zero_grad()

    outputs = model(data)
    loss = F.cross_entropy(outputs, labels, reduction='sum')
    loss.backward()

    # Gradient 누적
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_accum[name] += param.grad.cpu()

    total_samples += len(labels)

# 4. 전체 샘플 수로 나눔 (평균)
oracle_grad = {name: grad / total_samples for name, grad in grad_accum.items()}

# 5. BN stats 복구
restore_bn_stats(model, bn_backup)
```

**핵심 포인트**:
- `reduction='sum'` 사용 후 total_samples로 나눔
- Train 모드 유지 (Dropout, BN 등이 학습 시와 동일하게 작동)
- BN running stats는 백업/복구 (θ_ref 보존)

### Phase 2: Client/Server Gradient 분리

**목표**: Full model의 Oracle gradient를 client/server 파트로 분리

```python
# 방법 1: Parameter name 기준 분리 (Unified Framework, MultiSFL)
client_param_names = set(name for name, _ in client_model.named_parameters())
server_param_names = set(name for name, _ in server_model.named_parameters())

oracle_client_grad = {n: g for n, g in oracle_grad.items() if n in client_param_names}
oracle_server_grad = {n: g for n, g in oracle_grad.items() if n in server_param_names}

# 방법 2: Split Hook 사용 (Split Layer Gradient 필요 시)
def forward_hook(module, input, output):
    out_clone = output.clone().requires_grad_(True)
    out_clone.retain_grad()
    return out_clone

# Hook 등록 후 forward/backward 실행
hook_handle = split_layer.register_forward_hook(forward_hook)
# ... forward + backward ...
split_grad = activation.grad  # Split layer에서의 gradient
hook_handle.remove()
```

### Phase 3: Current Gradient 수집 (1-Step Measurement)

**목표**: 실제 학습 중 각 클라이언트와 서버의 gradient 수집

#### Client Gradient 수집

```python
# CRITICAL: 각 클라이언트의 FIRST BATCH만 측정
measured_clients = set()

for idx, (x, y, client_id) in enumerate(zip(x_list, y_list, client_ids)):
    if client_id in measured_clients:
        continue  # 이미 측정한 클라이언트는 스킵

    measured_clients.add(client_id)

    client_model.zero_grad()
    server_model.zero_grad()

    # SFL forward/backward
    activation = client_model(x)

    # Detach and require grad for server
    if isinstance(activation, tuple):
        act_detached = activation[0].detach().requires_grad_(True)
        id_detached = activation[1].detach().requires_grad_(True)
        logits = server_model((act_detached, id_detached))
    else:
        activation_detached = activation.detach().requires_grad_(True)
        logits = server_model(activation_detached)

    loss = F.cross_entropy(logits, y, reduction='mean')
    loss.backward()

    # Backward to client
    if isinstance(activation, tuple):
        torch.autograd.backward([activation[0], activation[1]],
                                [act_detached.grad, id_detached.grad])
    else:
        activation.backward(activation_detached.grad)

    # Collect client gradient
    current_client_grad[client_id] = {
        name: param.grad.clone().cpu()
        for name, param in client_model.named_parameters()
        if param.grad is not None
    }
```

**핵심 포인트**:
- **1-Step Only**: 각 클라이언트의 첫 번째 배치만 측정
- `reduction='mean'` 사용 (배치 내 평균)
- Detach 사용으로 client/server gradient 분리

#### Server Gradient 수집

```python
# Server는 각 배치마다 측정 가능
server_g_tildes = []
server_weights = []

for f_batch, y_batch in zip(f_list, y_list):
    server_model.zero_grad()

    if isinstance(f_batch, tuple):
        f_act, f_id = f_batch
        logits = server_model((f_act, f_id))
    else:
        logits = server_model(f_batch)

    loss = F.cross_entropy(logits, y_batch, reduction='mean')
    loss.backward()

    # Collect server gradient
    current_server_grad = {
        name: param.grad.clone().cpu()
        for name, param in server_model.named_parameters()
        if param.grad is not None
    }

    server_g_tildes.append(current_server_grad)
    server_weights.append(len(y_batch))  # batch size as weight
```

### Phase 4: Gradient Flattening (일관된 순서 보장)

**목표**: Dictionary 형태의 gradient를 1D vector로 변환 (비교 가능하도록)

```python
def gradient_to_vector(
    grad_dict: Dict[str, torch.Tensor],
    reference_names: Optional[List[str]] = None
) -> torch.Tensor:
    """
    CRITICAL: Parameter name을 정렬하여 일관된 순서 보장
    """
    if reference_names is None:
        reference_names = sorted(grad_dict.keys())  # ← SORTED!

    vectors = []
    for name in reference_names:
        if name in grad_dict:
            vectors.append(grad_dict[name].flatten().float())

    return torch.cat(vectors) if vectors else torch.zeros(1)
```

**왜 중요한가?**
- Dictionary는 순서가 보장되지 않을 수 있음
- Gradient 비교 시 동일한 parameter가 동일한 위치에 있어야 함
- `sorted()`를 사용해 알파벳 순으로 정렬

### Phase 5: G Metrics 계산

```python
def compute_g_metrics(
    g_tilde: Dict[str, torch.Tensor],
    g_oracle: Dict[str, torch.Tensor],
    epsilon: float = 1e-8
) -> GMetrics:
    # Key normalization (optional)
    g_tilde_norm = {k.replace('-', '.'): v for k, v in g_tilde.items()}
    g_oracle_norm = {k.replace('-', '.'): v for k, v in g_oracle.items()}

    # 공통 키 추출 및 정렬
    common_keys = sorted(set(g_tilde_norm.keys()) & set(g_oracle_norm.keys()))

    # Flatten with consistent ordering
    g_tilde_flat = gradient_to_vector(g_tilde_norm, common_keys)
    g_oracle_flat = gradient_to_vector(g_oracle_norm, common_keys)

    # 1. G (Absolute Distance)
    diff = g_tilde_flat - g_oracle_flat
    G = torch.dot(diff, diff).item()

    # 2. G_rel (Relative Distance)
    oracle_norm_sq = torch.dot(g_oracle_flat, g_oracle_flat).item()
    G_rel = G / (oracle_norm_sq + epsilon)

    # 3. D_cosine (Cosine Distance)
    tilde_norm = torch.norm(g_tilde_flat).item()
    oracle_norm = torch.norm(g_oracle_flat).item()

    if tilde_norm > epsilon and oracle_norm > epsilon:
        dot_product = torch.dot(g_tilde_flat, g_oracle_flat).item()
        cos_sim = dot_product / (tilde_norm * oracle_norm)
        D_cosine = 1.0 - cos_sim
    else:
        D_cosine = 1.0

    D_cosine = max(0.0, min(2.0, D_cosine))  # Clamp to [0, 2]

    return GMetrics(G=G, G_rel=G_rel, D_cosine=D_cosine)
```

### Phase 6: Variance_G 계산 (핵심!)

#### Variance_Client_G 계산

```python
# 준비: Oracle 및 per-client gradients
oracle_client_grad = {...}  # Oracle gradient (dict)
client_g_tildes = {
    client_id_0: {...},  # Client 0의 gradient
    client_id_1: {...},  # Client 1의 gradient
    ...
}
client_weights = {
    client_id_0: batch_size_0,
    client_id_1: batch_size_1,
    ...
}

# Step 1: Oracle gradient를 vector로 변환
oracle_grad_norm = {k.replace('-', '.'): v for k, v in oracle_client_grad.items()}
oracle_keys = sorted(oracle_grad_norm.keys())
oracle_vec = gradient_to_vector(oracle_grad_norm, oracle_keys)

# Step 2: Total weight 계산
total_weight = sum(client_weights.values())

# Step 3: 각 클라이언트의 variance 누적
Vc = 0.0
for client_id, g_tilde in client_g_tildes.items():
    # Normalize keys
    g_tilde_norm = {k.replace('-', '.'): v for k, v in g_tilde.items()}

    # Flatten with same key ordering as oracle
    vec = gradient_to_vector(g_tilde_norm, oracle_keys)

    # Weighted distance
    weight = client_weights[client_id] / total_weight
    diff = vec - oracle_vec
    Vc += weight * torch.dot(diff, diff).item()

variance_client_g = Vc

# Step 4: Relative variance
oracle_norm_sq = torch.dot(oracle_vec, oracle_vec).item()
variance_client_g_rel = Vc / oracle_norm_sq if oracle_norm_sq > 0 else float('nan')
```

**핵심 포인트**:
1. **가중치**: 각 클라이언트의 배치 크기를 가중치로 사용
2. **정규화**: Total weight로 나누어 정규화된 가중 평균
3. **Distance**: 각 클라이언트의 gradient와 Oracle의 L2 거리 제곱
4. **누적**: 가중치 × 거리를 모두 합산

#### Variance_Server_G 계산

```python
# 준비: Oracle 및 per-batch server gradients
oracle_server_grad = {...}
server_g_tildes = [grad_batch_0, grad_batch_1, ...]
server_weights = [batch_size_0, batch_size_1, ...]

# Step 1: Oracle vector
oracle_grad_norm = {k.replace('-', '.'): v for k, v in oracle_server_grad.items()}
server_keys = sorted(oracle_grad_norm.keys())
oracle_vec = gradient_to_vector(oracle_grad_norm, server_keys)

# Step 2: Total weight
total_weight = sum(server_weights)

# Step 3: 각 배치의 variance 누적
Vs = 0.0
for server_grad, weight in zip(server_g_tildes, server_weights):
    server_grad_norm = {k.replace('-', '.'): v for k, v in server_grad.items()}
    server_vec = gradient_to_vector(server_grad_norm, server_keys)

    scaled_weight = weight / total_weight
    diff = server_vec - oracle_vec
    Vs += scaled_weight * torch.dot(diff, diff).item()

variance_server_g = Vs

# Step 4: Relative variance
oracle_norm_sq = torch.dot(oracle_vec, oracle_vec).item()
variance_server_g_rel = Vs / oracle_norm_sq if oracle_norm_sq > 0 else float('nan')
```

---

## Implementation Differences

### GAS Implementation

**파일**: `GAS_implementation/utils/g_measurement.py`

**특징**:
- `GMeasurementManager` 클래스 사용
- Diagnostic round 패턴 (매 N 에폭마다 측정)
- Oracle 계산 시 `compute_oracle_with_split_hook` 지원
- Variance 계산은 history에 추적되지만 구현 미완성

**사용 패턴**:
```python
g_manager = GMeasurementManager(device='cuda', measure_frequency=10)

# Diagnostic round에서만 oracle 계산
if g_manager.should_measure(epoch):
    g_manager.compute_oracle(user_model, server_model, train_loader, criterion)

    # 학습 후 측정
    g_scores = g_manager.measure_and_record(
        user_model, server_model, split_output, loss, epoch
    )
```

### Unified Framework Implementation

**파일**: `sfl_framework-fork-feature-training-tracker/server/utils/g_measurement.py`

**특징**:
- `GMeasurementSystem` 클래스 사용
- 가장 완전한 구현 (1264 lines)
- Split layer gradient 측정 지원 (tuple output 포함)
- Variance_G 완전 구현
- Propagator 사용으로 다양한 모델 아키텍처 지원

**사용 패턴**:
```python
g_system = GMeasurementSystem(
    diagnostic_rounds=[1, 3, 5, 10],
    device='cuda',
    use_variance_g=True
)

# 초기화
g_system.initialize(full_dataloader)
g_system.set_param_names(client_model, server_model)

# Diagnostic round에서
if g_system.is_diagnostic_round(round_num):
    # Oracle 계산
    g_system.compute_oracle_split_for_round(
        client_model, server_model, full_model, split_layer_name, config
    )

    # 1-step measurement 후 gradient 저장
    g_system.store_measurement_gradient(server_grad, client_grads)

    # G 계산
    result = g_system.compute_g(round_num, client_weights)
```

### MultiSFL Implementation

**파일**: `multisfl_implementation/multisfl/g_measurement.py`

**특징**:
- `GMeasurementSystem` 클래스 사용 (Unified와 유사)
- Multi-branch 지원 (branch별 oracle 가능)
- Branch ID 기반 gradient 추적
- Variance_G 구현 (branch-aware)

**사용 패턴**:
```python
g_system = GMeasurementSystem(
    full_dataloader=full_dataloader,
    device='cuda',
    diagnostic_frequency=10,
    use_variance_g=True
)

# Oracle 계산
g_system.compute_oracle(client_model, server_model)

# Branch별 oracle (optional)
g_system.set_branch_oracle_grads(
    oracle_client_grads_by_branch={0: {...}, 1: {...}},
    oracle_server_grads_by_branch={0: {...}, 1: {...}}
)

# Round 측정
result = g_system.measure_round(
    round_idx, client_model, server_model,
    client_ids, x_all, y_all_client, f_all, y_all_server,
    client_weights, server_weights,
    client_branch_ids, server_branch_ids
)
```

---

## Key Implementation Details

### 1. BatchNorm Stats 관리

**문제**: Oracle 계산 시 BN running stats가 변경되면 θ_ref가 오염됨

**해결책**:
```python
def backup_bn_stats(model):
    backup = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            backup[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
                'num_batches_tracked': module.num_batches_tracked.clone()
            }
    return backup

def restore_bn_stats(model, backup):
    for name, module in model.named_modules():
        if name in backup:
            module.running_mean.copy_(backup[name]['running_mean'])
            module.running_var.copy_(backup[name]['running_var'])
            module.num_batches_tracked.copy_(backup[name]['num_batches_tracked'])
```

### 2. Tuple Output 처리 (Fine-grained Split)

일부 split 구성에서 activation이 tuple 형태:
```python
# Forward
activation = client_model(x)
if isinstance(activation, tuple):
    act, identity = activation  # (transformed_features, identity_shortcut)
    act_detached = act.detach().requires_grad_(True)
    id_detached = identity.detach().requires_grad_(True)
    logits = server_model((act_detached, id_detached))
else:
    activation_detached = activation.detach().requires_grad_(True)
    logits = server_model(activation_detached)

# Backward
loss.backward()

if isinstance(activation, tuple):
    torch.autograd.backward([act, identity], [act_detached.grad, id_detached.grad])
else:
    activation.backward(activation_detached.grad)
```

### 3. Key Normalization

Parameter name에 `-`와 `.`이 혼재할 수 있음:
```python
def normalize_grad_keys(grad_dict):
    return {k.replace('-', '.'): v for k, v in grad_dict.items()}
```

### 4. Memory Management

```python
# 1. CPU offloading
grad_cpu = param.grad.clone().detach().cpu()

# 2. Immediate cleanup
del images, labels, outputs, loss

# 3. CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 4. Garbage collection
import gc
gc.collect()
```

---

## Interpretation Guide

### G Values

| G Value | Interpretation | Action |
|---------|---------------|---------|
| **< 0.01** | Excellent alignment with Oracle | Current training is optimal |
| **0.01 - 0.1** | Good alignment | Minor drift, acceptable |
| **0.1 - 1.0** | Moderate drift | Consider adjusting learning rate or client selection |
| **> 1.0** | Significant drift | Investigate: gradient explosion, poor client selection, or Non-IID issues |

### G_rel Values

| G_rel | Interpretation |
|-------|---------------|
| **< 0.1** | Relative error < 10% |
| **0.1 - 0.5** | Moderate relative error |
| **> 0.5** | High relative error (gradient > 50% off) |

### D_cosine Values

| D_cosine | cos(similarity) | Interpretation |
|----------|----------------|----------------|
| **0.0** | 1.0 | Perfect alignment (same direction) |
| **0.5** | 0.5 | 60° angle |
| **1.0** | 0.0 | Orthogonal (90° angle) |
| **2.0** | -1.0 | Opposite direction (180°) |

### Variance_G Interpretation

**Variance_Client_G**:
- **Low (< 0.1)**: 모든 클라이언트가 Oracle에 가까움 → 균등한 학습
- **Medium (0.1 - 1.0)**: 일부 클라이언트가 drift → 선택 전략 개선 필요
- **High (> 1.0)**: 클라이언트 간 심각한 불균형 → Non-IID 문제 또는 데이터 분포 재검토

**Variance_Server_G**:
- **Low (< 0.05)**: 배치 간 일관된 gradient → 안정적인 학습
- **Medium (0.05 - 0.5)**: 배치 간 변동성 → 배치 크기 조정 고려
- **High (> 0.5)**: 배치 간 큰 변동 → Gradient shuffle 또는 배치 구성 검토

**Variance_G_Rel**:
- 절대값이 아닌 Oracle norm 대비 상대적 분산
- Non-IID 환경에서 클라이언트 선택 품질 평가에 유용

---

## Usage Examples

### Example 1: Basic G Measurement (GAS)

```python
from utils.g_measurement import GMeasurementManager, compute_g_score

# Setup
device = torch.device('cuda')
g_manager = GMeasurementManager(device, measure_frequency=10)

# Training loop
for epoch in range(300):
    # Oracle 계산 (diagnostic round)
    if g_manager.should_measure(epoch):
        g_manager.compute_oracle(
            user_model, server_model, train_loader, criterion
        )

    # Normal training
    for batch_idx, (data, target) in enumerate(trainloader):
        # Forward
        split_output = user_model(data)
        split_output.retain_grad()  # IMPORTANT!

        server_output = server_model(split_output)
        loss = criterion(server_output, target)

        # Backward
        loss.backward()

        # Measure G (if diagnostic round)
        if g_manager.should_measure(epoch):
            g_scores = g_manager.measure_and_record(
                user_model, server_model, split_output, loss, epoch
            )
            print(f"Client G: {g_scores['client_g']:.6f}")
            print(f"Server G: {g_scores['server_g']:.6f}")

        # Optimizer step
        client_optimizer.step()
        server_optimizer.step()

        user_model.zero_grad()
        server_model.zero_grad()

# Get history
history = g_manager.get_history()
```

### Example 2: Variance_G Measurement (Unified Framework)

```python
from server.utils.g_measurement import GMeasurementSystem

# Setup with variance enabled
g_system = GMeasurementSystem(
    diagnostic_rounds=[1, 5, 10, 20, 50],
    device='cuda',
    use_variance_g=True  # ← Enable variance calculation
)

# Initialize with full dataset
g_system.initialize(full_dataloader)
g_system.set_param_names(client_model, server_model)

# In diagnostic round
if g_system.is_diagnostic_round(round_num):
    # Compute oracle
    g_system.compute_oracle_split_for_round(
        client_model, server_model, full_model,
        split_layer_name='layer1.1.bn2', config=config
    )

    # 1-step measurement (collect gradients)
    # ... perform one training step and collect gradients ...
    server_grad = {name: param.grad for name, param in server_model.named_parameters()}
    client_grads = {
        client_id: {name: param.grad for name, param in client_model.named_parameters()}
        for client_id in selected_clients
    }

    # Store gradients
    g_system.store_measurement_gradient(server_grad, client_grads)

    # Compute G metrics including variance
    result = g_system.compute_g(round_num, client_weights={0: 32, 1: 32, 2: 28})

    # Print results
    print(f"Server G: {result.server.G:.6f}, G_rel: {result.server.G_rel:.4f}")
    print(f"Client G mean: {result.client_G_mean:.6f}")
    print(f"Variance Client G: {result.variance_client_g:.6f}")
    print(f"Variance Client G_rel: {result.variance_client_g_rel:.6f}")
    print(f"Variance Server G: {result.variance_server_g:.6f}")
    print(f"Variance Server G_rel: {result.variance_server_g_rel:.6f}")

    # Clear memory
    g_system.clear_round_data()
```

### Example 3: Multi-Branch Measurement (MultiSFL)

```python
from multisfl.g_measurement import GMeasurementSystem, GMetrics

# Setup for multi-branch
g_system = GMeasurementSystem(
    full_dataloader=full_dataloader,
    device='cuda',
    diagnostic_frequency=10,
    use_variance_g=True
)

# Compute branch-specific oracles
oracle_client_grads_by_branch = {}
oracle_server_grads_by_branch = {}

for branch_id in range(num_branches):
    client_model = branch_clients[branch_id]
    server_model = branch_servers[branch_id]

    oracle_client, oracle_server = g_system.oracle_calculator.compute_oracle_gradient(
        client_model, server_model
    )

    oracle_client_grads_by_branch[branch_id] = oracle_client
    oracle_server_grads_by_branch[branch_id] = oracle_server

g_system.set_branch_oracle_grads(
    oracle_client_grads_by_branch,
    oracle_server_grads_by_branch
)

# Measure round with branch IDs
result = g_system.measure_round(
    round_idx=round_num,
    client_model=main_client_model,
    server_model=main_server_model,
    client_ids=[0, 1, 2, 3, 4],
    x_all=[x0, x1, x2, x3, x4],
    y_all_client=[y0, y1, y2, y3, y4],
    f_all=[f0, f1, f2, f3, f4],
    y_all_server=[y0, y1, y2, y3, y4],
    client_branch_ids=[0, 0, 1, 1, 2],  # Client to branch mapping
    server_branch_ids=[0, 1, 2, 0, 1]   # Server batch to branch mapping
)

# Branch-specific results
for branch_id, metrics in result.per_branch_server_g.items():
    print(f"Branch {branch_id} Server G: {metrics.G:.6f}")
```

---

## Common Pitfalls

### 1. ❌ Forgetting retain_grad()

```python
# WRONG
activation = client_model(data)
logits = server_model(activation)
loss.backward()
# activation.grad is None!

# CORRECT
activation = client_model(data)
activation.retain_grad()  # ← This!
logits = server_model(activation)
loss.backward()
# Now activation.grad is available
```

### 2. ❌ Inconsistent Parameter Ordering

```python
# WRONG
vec1 = torch.cat([grad_dict[k].flatten() for k in grad_dict.keys()])
vec2 = torch.cat([grad_dict[k].flatten() for k in grad_dict.keys()])
# Dictionary iteration order might differ!

# CORRECT
keys = sorted(grad_dict.keys())
vec1 = torch.cat([grad_dict[k].flatten() for k in keys])
vec2 = torch.cat([grad_dict[k].flatten() for k in keys])
```

### 3. ❌ Not Restoring BN Stats

```python
# WRONG
for batch in full_loader:
    outputs = model(data)
    loss.backward()
# BN stats are now polluted!

# CORRECT
bn_backup = backup_bn_stats(model)
for batch in full_loader:
    outputs = model(data)
    loss.backward()
restore_bn_stats(model, bn_backup)
```

### 4. ❌ Using reduction='mean' for Oracle

```python
# SUBOPTIMAL (but used by some implementations)
loss = F.cross_entropy(outputs, labels, reduction='mean')
loss.backward()
# Gradient is already averaged by batch size

# RECOMMENDED (for accurate total gradient)
loss = F.cross_entropy(outputs, labels, reduction='sum')
loss.backward()
# Then divide by total_samples at the end
```

### 5. ❌ Measuring Multiple Steps per Client

```python
# WRONG - Violates 1-step measurement protocol
for client_id in clients:
    for batch in client_loader:  # Multiple batches!
        activation = client_model(batch)
        # ...

# CORRECT - Only first batch
measured_clients = set()
for batch, client_id in data:
    if client_id in measured_clients:
        continue  # Skip already measured clients
    measured_clients.add(client_id)
    # Measure this batch only
```

---

## Summary

**G-Measurement의 핵심**:
1. **Oracle = 이상적인 참조값** (전체 데이터셋 기준)
2. **Current = 실제 측정값** (1-step gradient)
3. **G = 거리**, **G_rel = 상대 거리**, **D_cosine = 방향 차이**
4. **Variance_G = 클라이언트/서버 간 분산** (가중 평균)

**사용 시나리오**:
- **GAS**: G-score 기반 클라이언트 선택 알고리즘
- **USFL**: Non-IID 환경에서 gradient quality 모니터링
- **MultiSFL**: Branch별 학습 품질 비교 및 최적화

**핵심 프로토콜**:
1. BN stats 백업/복구
2. Train 모드 유지
3. 1-step measurement
4. Consistent parameter ordering
5. Weighted variance calculation
