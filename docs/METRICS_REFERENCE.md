# Metrics Reference

Complete reference for every metric computed across the three frameworks (SFL, GAS, MultiSFL).

**Last updated:** 2026-02-08
**Scope:** All computed, logged, and saved metrics

---

## Table of Contents

1. [Overview & Metric Categories](#1-overview--metric-categories)
2. [G Measurement (Oracle Gradient Distance)](#2-g-measurement-oracle-gradient-distance)
3. [Drift Measurement (SCAFFOLD-style)](#3-drift-measurement-scaffold-style)
4. [Update Alignment (A_cos + M_norm)](#4-update-alignment-a_cos--m_norm)
5. [Training Metrics (Accuracy & Loss)](#5-training-metrics-accuracy--loss)
6. [GAS-Specific Metrics](#6-gas-specific-metrics)
7. [USFL-Specific Metrics](#7-usfl-specific-metrics)
8. [MultiSFL-Specific Metrics](#8-multisfl-specific-metrics)
9. [Configuration Reference](#9-configuration-reference)
10. [Result JSON Schema](#10-result-json-schema)

---

## 1. Overview & Metric Categories

The metrics fall into four conceptual layers:

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 4: Framework-Specific Metrics                         │
│  V-value (GAS), Freshness Score (USFL), FGN (MultiSFL)      │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Update Alignment   (A_cos, M_norm)                 │
│  Scale-invariant, comparable across all frameworks           │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Drift Measurement  (G_drift, G_end, D_dir, D_rel) │
│  Per-round trajectory tracking during local optimization     │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: G Measurement      (G, G_rel, D_cosine)            │
│  Oracle-based gradient quality (computed on diagnostic rounds)│
└──────────────────────────────────────────────────────────────┘
```

| Category | Frequency | Frameworks | Key Question Answered |
|----------|-----------|------------|----------------------|
| G Measurement | Every N rounds | All | "How far are training gradients from the true gradient?" |
| Drift Measurement | Every round | All | "How much do local updates deviate from the global model during a round?" |
| Update Alignment | Every round | All | "Do clients agree on which direction to update?" |
| Training Metrics | Every round | All | "How well is the model learning?" |

---

## 2. G Measurement (Oracle Gradient Distance)

Compares actual training gradients against the **oracle gradient** (the gradient you would get by training on the *entire* dataset at once). Measures how well the federated/split training approximates centralized training.

### 2.1 Oracle Gradient (g\*)

The oracle is the reference "perfect" gradient. Computed as:

```
g* = (1 / |D|) * Σ_{(x,y) ∈ D} ∇L(x, y)
```

Where `D` is the full training dataset and `L` is cross-entropy loss with `reduction='sum'`.

**Implementation details (shared across all frameworks):**
- Forward pass through the **entire** training dataset in batches
- Gradients accumulated with `reduction='sum'`, then divided by `total_samples`
- Model kept in `train()` mode (same conditions as actual training)
- BatchNorm statistics backed up before oracle computation and restored after
- Split into client-side oracle (`g*_client`) and server-side oracle (`g*_server`)

**Oracle modes:**

| Mode | Config Value | SFL/GAS | MultiSFL | Description |
|------|-------------|---------|----------|-------------|
| Global | `"global"` | `"strict"` | `"master"` | Single oracle using the current aggregated global model |
| Individual | `"individual"` | `"realistic"` | `"branch"` | Per-model oracle (per-client in SFL/GAS, per-branch in MultiSFL) |

### 2.2 Core G Metrics (the triplet)

Every G measurement produces three values. These are computed identically in all frameworks via `compute_g_metrics()`:

#### G (Squared L2 Distance)

```
G = ‖g̃ - g*‖² = (g̃ - g*)ᵀ(g̃ - g*)
```

| | |
|---|---|
| **What it measures** | Absolute squared distance between the training gradient (`g̃`) and the oracle gradient (`g*`) |
| **Range** | [0, +∞) — lower is better |
| **Unit** | Squared gradient norm (scale-dependent) |
| **Limitation** | Sensitive to learning rate, batch size, and model scale. Not directly comparable across different hyperparameter settings |

#### G_rel (Relative Distance)

```
G_rel = G / (‖g*‖² + ε)       where ε = 1e-8
```

| | |
|---|---|
| **What it measures** | G normalized by the oracle's magnitude. "How large is the error relative to the signal?" |
| **Range** | [0, +∞) — lower is better |
| **Interpretation** | G_rel = 0.5 means the error is half the oracle magnitude. G_rel > 1 means the error exceeds the signal |
| **Advantage** | Comparable across models of different sizes and learning rates |

#### D_cosine (Cosine Distance)

```
D_cosine = 1 - cos(g̃, g*) = 1 - (g̃ · g*) / (‖g̃‖ · ‖g*‖)
```

Clamped to [0, 2] for numerical stability.

| | |
|---|---|
| **What it measures** | Directional disagreement between training and oracle gradients |
| **Range** | [0, 2] — 0 = same direction, 1 = orthogonal, 2 = opposite |
| **Advantage** | Purely directional — immune to gradient magnitude/scale |

### 2.3 Perspectives (Client, Server, Split Layer)

Each G measurement is computed from three perspectives corresponding to the split learning architecture:

```
Input → [Client Model] → activations → [Server Model] → loss
         g̃_client              g̃_split         g̃_server
```

| Perspective | What gradient is measured | Granularity |
|-------------|-------------------------|-------------|
| **Client G** | `∇_client_params L` — gradients of the bottom model | Per-client (individual), then averaged |
| **Server G** | `∇_server_params L` — gradients of the top model | Per-batch or per-branch |
| **Split G** | `∂L/∂activations` — gradient at the cut point | Averaged across clients |

### 2.4 Measurement Modes

How training gradients (`g̃`) are collected during a diagnostic round:

| Mode | Config | Description | Use Case |
|------|--------|-------------|----------|
| `single` | `g_measurement_mode: "single"` | First batch per client only | Fast, approximate |
| `k_batch` | `g_measurement_mode: "k_batch"` | First K batches, weighted average | Balanced accuracy/speed |
| `accumulated` | `g_measurement_mode: "accumulated"` | All batches in the round, weighted average | Most accurate, slowest |

Weighted averaging formula (for k_batch and accumulated):
```
g̃_avg = (Σ_b  g̃_b × batch_size_b) / (Σ_b  batch_size_b)
```

### 2.5 Variance G (Optional)

When `use_variance_g = true`, computes a weighted-variance decomposition:

```
V_c = Σ_i (w_i / W) × ‖g̃_i - g*‖²
```

Where `w_i = batch_size_i` and `W = Σ w_i`.

| Metric | Formula | Purpose |
|--------|---------|---------|
| `variance_client_g` | V_c | Weighted sum of per-client G values (accounts for data volume) |
| `variance_client_g_rel` | V_c / ‖g\*‖² | Scale-invariant variant |
| `variance_server_g` | Same formula over server gradients | Server-side variance |
| `variance_server_g_rel` | V_s / ‖g\*‖² | Scale-invariant server variant |

**Difference from mean G:** Mean G treats each client equally (1/N weighting). Variance G weights by data contribution (batch_size_i / total), giving more influence to clients that process more data.

### 2.6 Source Locations

| Framework | File | Class/Function |
|-----------|------|----------------|
| SFL | `sfl_framework-.../server/utils/g_measurement.py` | `GMeasurementSystem` (line ~764) |
| GAS | `GAS_implementation/utils/g_measurement.py` | `GMeasurementManager` (line ~496) |
| MultiSFL | `multisfl_implementation/multisfl/g_measurement.py` | `GMeasurementSystem` (line ~440) |

---

## 3. Drift Measurement (SCAFFOLD-style)

Tracks how much local model parameters deviate from the round-start global model during a training round. Inspired by the SCAFFOLD paper's client drift analysis.

### 3.1 Core Concepts

At the start of each round, the global model parameters are `x^{t,0}`. During local training, each client takes B optimizer steps, producing `x^{t,1}, x^{t,2}, ..., x^{t,B}`. Drift metrics capture how this trajectory deviates.

```
x^{t,0} ──step 1──> x^{t,1} ──step 2──> x^{t,2} ──...──> x^{t,B}
  │                    │                    │                  │
  └─── drift₁ ────────┘                    │                  │
  └─── drift₂ ─────────────────────────────┘                  │
  └─── drift_B (endpoint) ────────────────────────────────────┘
```

Only **trainable parameters** are included (BatchNorm running statistics are excluded).

### 3.2 Per-Client State

For each participating client `n` in round `t`:

| Symbol | Name | Formula | Description |
|--------|------|---------|-------------|
| S_n | `trajectory_sum` | `Σ_{b=1}^{B_n} ‖x_n^{t,b} - x_n^{t,0}‖²` | Accumulated squared distance from round start after each optimizer step |
| B_n | `batch_steps` | Step counter | Number of optimizer steps taken |
| E_n | `endpoint_drift` | `‖x_n^{t,B_n} - x_n^{t,0}‖²` | Final squared distance from round start |

Where `‖·‖²` is the sum of squared differences across all trainable parameters:
```
‖a - b‖² = Σ_{param ∈ trainable} Σ_{element} (a[param][element] - b[param][element])²
```

### 3.3 Client Drift Metrics (aggregated over clients)

#### G_drift_client (Average Trajectory Drift Energy)

```
G_drift_client = (1 / |P_t|) × Σ_n (S_n / B_n)
```

| | |
|---|---|
| **What it measures** | Average per-step drift energy across participating clients |
| **Interpretation** | Higher = clients are drifting further from the global model during each step |
| **Use case** | Primary drift indicator. High G_drift_client suggests local training is pushing clients away from consensus |

#### G_drift_client_stepweighted (Step-Weighted Variant)

```
G_drift_client_stepweighted = ΣS / ΣB = (Σ_n S_n) / (Σ_n B_n)
```

| | |
|---|---|
| **Difference from G_drift_client** | Avoids "1 client = 1 vote" distortion when clients have different step counts |
| **Example** | If client A takes 10 steps and client B takes 1 step, G_drift_client weights them equally, but stepweighted gives client A 10x more influence |

#### G_end_client (Average Endpoint Drift)

```
G_end_client = (1 / |P_t|) × Σ_n E_n
```

| | |
|---|---|
| **What it measures** | Average squared distance between each client's final model and the round-start global model |
| **Difference from G_drift** | G_drift captures the *journey* (trajectory), G_end captures only the *destination* |

#### G_end_client_weighted (Aggregation-Weight-Weighted Endpoint)

```
G_end_client_weighted = Σ_i (w_i / W) × E_i
```

Where `w_i` are the FedAvg aggregation weights (proportional to dataset size). In GAS, aggregation is uniform, so this equals `G_end_client`.

### 3.4 Update Disagreement Metrics

These use a **variance decomposition identity** to measure how much individual client updates disagree with each other.

#### D_dir (Directional Disagreement)

```
D_dir = E_w[‖Δ_i‖²] - ‖E_w[Δ_i]‖²
      = G_end_client_weighted - delta_client_norm_sq
```

Where `Δ_i = x_i^{t,B} - x^{t,0}` is client i's update vector.

| | |
|---|---|
| **Mathematical identity** | This is exactly `Var_w(Δ_i)` — the weighted variance of client updates |
| **Interpretation** | D_dir = 0 means all clients agree perfectly. Higher = more disagreement |
| **Why "directional"** | It captures disagreement in both magnitude AND direction |

#### D_rel (Relative Disagreement)

```
D_rel = D_dir / (‖Δ_global‖² + ε)
```

| | |
|---|---|
| **What it measures** | Disagreement normalized by the aggregated update magnitude |
| **Interpretation** | D_rel = 1 means client disagreement is as large as the global update itself. D_rel >> 1 signals that clients are pulling in very different directions |

### 3.5 Server Drift Metrics

Same formulas as client drift but applied to the server model (top portion of the split):

| Metric | Formula |
|--------|---------|
| `G_drift_server` | `S_server / B_server` |
| `G_end_server` | `E_server = ‖x_s^{t,B} - x_s^{t,0}‖²` |
| `delta_server_norm_sq` | `‖x_s^{t+1,0} - x_s^{t,0}‖²` (aggregated server update magnitude) |
| `G_drift_norm_server` | `G_drift_server / (delta_server_norm_sq + ε)` |

### 3.6 Global Update Magnitude

```
delta_client_norm_sq = ‖x_c^{t+1,0} - x_c^{t,0}‖²
```

| | |
|---|---|
| **What it measures** | How much the aggregated global client model changed this round |
| **Purpose** | Used as denominator in normalized metrics (G_drift_norm, D_rel) |
| **Note** | This is the **aggregated** update — the weighted average of all client updates — not individual client drift |

### 3.7 Normalized Drift

```
G_drift_norm_client = G_drift_client / (delta_client_norm_sq + ε)
```

| | |
|---|---|
| **Purpose** | Prevents the "update suppression" criticism: if the model barely moves (small delta), raw drift might look small even though it's proportionally large |
| **Adaptive ε** | After 10 rounds, ε is set to `1e-3 × median(delta_norms of first 10 rounds)`. This prevents division-by-near-zero instability in early training |

### 3.8 Combined Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `G_drift_total` | `G_drift_client + G_drift_server` | Total system drift (both halves of the split model) |
| `G_end_total` | `G_end_client + G_end_server` | Total endpoint drift |

### 3.9 Legacy Aliases

For backward compatibility, the result JSON also includes:

| Legacy Name | Actual Metric |
|-------------|---------------|
| `G_drift` | `G_drift_client` |
| `G_end` | `G_end_client` |
| `G_drift_norm` | `G_drift_norm_client` |
| `delta_global_norm_sq` | `delta_client_norm_sq` |
| `num_clients` | `num_clients` (GAS/SFL) or `num_branches` (MultiSFL) |

### 3.10 Source Locations

| Framework | File | Class |
|-----------|------|-------|
| SFL | `sfl_framework-.../server/utils/drift_measurement.py` | `DriftMeasurementTracker` |
| GAS | `GAS_implementation/utils/drift_measurement.py` | `DriftMeasurementTracker` |
| MultiSFL | `multisfl_implementation/multisfl/drift_measurement.py` | `MultiSFLDriftTracker` |

### 3.11 MultiSFL Drift Differences

In MultiSFL, drift is measured **per-branch** (not per-client):
- Each branch trains a copy of the model
- `S_b`, `B_b`, `E_b` are tracked per branch server
- `G_drift_client` averages over branches, not individual clients
- **Both** client-side and server-side `A_cos` are computed (unique to MultiSFL)

---

## 4. Update Alignment (A_cos + M_norm)

Scale-invariant metrics that measure how aligned client (or branch) updates are in their optimization direction. Designed to be **directly comparable across all frameworks** regardless of learning rate, batch size, or step count.

### 4.1 A_cos (Cosine Alignment)

```
A_cos = (Σ_{i<j} w_ij × cos(Δ_i, Δ_j)) / (Σ_{i<j} w_ij)
```

Where:
- `Δ_i = flatten(θ_end_i - θ_start)` — client i's flattened parameter update vector
- `cos(a, b) = (a · b) / (‖a‖ × ‖b‖)` — cosine similarity
- `w_ij = w_i × w_j` — pair weight (product of individual aggregation weights)
- Only trainable parameters are included (BatchNorm buffers excluded)

| | |
|---|---|
| **Range** | [-1, 1] (NaN if fewer than 2 valid clients) |
| **A_cos = 1** | All clients update in exactly the same direction (perfect alignment) |
| **A_cos ≈ 0** | Client updates are roughly orthogonal (no systematic agreement) |
| **A_cos < 0** | Clients actively disagree (updates point in opposing directions) |
| **Threshold** | Clients with `‖Δ_i‖ ≤ τ` (default τ = 1e-7) are excluded from A_cos |
| **Key advantage** | Immune to LR, batch size, and step count differences across frameworks |

**Intuition:** Think of each client's update as an arrow in high-dimensional space. A_cos measures whether these arrows point in the same direction, regardless of their length.

### 4.2 M_norm (Mean Update Magnitude)

```
M_norm = (Σ_i w_i × ‖Δ_i‖) / (Σ_i w_i)
```

| | |
|---|---|
| **Range** | [0, +∞) |
| **What it measures** | Average L2 norm of client update vectors |
| **Purpose** | Complements A_cos — tells you HOW MUCH clients are moving, while A_cos tells you WHETHER they agree on direction |
| **Includes all clients** | Unlike A_cos, clients below the τ threshold are still included |

### 4.3 Relationship Between A_cos and D_rel

Both measure client agreement but from different angles:

| Aspect | A_cos | D_rel |
|--------|-------|-------|
| **What it compares** | Pairwise client directions | Clients vs. aggregated mean |
| **Scale invariance** | Yes (cosine-based) | No (uses squared norms) |
| **Sensitivity** | Direction only | Direction + magnitude |
| **Cross-framework comparable** | Yes | Only within same hyperparameters |

### 4.4 MultiSFL Dual A_cos

MultiSFL computes **separate** alignment metrics for client and server sides:

| Metric | What it measures |
|--------|-----------------|
| `A_cos_client` | Alignment of branch client model updates |
| `A_cos_server` | Alignment of branch server model updates |
| `M_norm_client` | Mean branch client update magnitude |
| `M_norm_server` | Mean branch server update magnitude |

This is unique to MultiSFL because it has per-branch server models. SFL and GAS only compute a single A_cos over client updates.

### 4.5 Source Location

| File | Line |
|------|------|
| `shared/update_alignment.py` | `compute_update_alignment()` at line 82 |
| `shared/update_alignment.py` | `flatten_delta()` at line 34 |

---

## 5. Training Metrics (Accuracy & Loss)

### 5.1 Test Accuracy

```
accuracy = correct / total
```

Where `correct = Σ 𝟙[argmax(model(x)) == y]` over the test set.

| Framework | Location | Frequency |
|-----------|----------|-----------|
| SFL | `server/modules/model/flexible_resnet.py` + stage organizer `_post_round` | Every round |
| GAS | `GAS_main.py:1286-1296` | Every `Accu_Test_Frequency` rounds (default: 1) |
| MultiSFL | `multisfl/trainer.py:147-195` | Every round |

**MultiSFL note:** Accuracy is evaluated using the **master model** (averaged across all branches), not individual branch models.

### 5.2 Training Loss

```
loss = CrossEntropyLoss(server_model(activations), labels)
```

| Framework | Granularity | Saved? |
|-----------|-------------|--------|
| SFL | Averaged per round: `epoch_loss = Σ loss / num_iterations` | Logged to TrainingTracker |
| GAS | Per-step (client + server) | Not saved to results |
| MultiSFL | Per-step per-branch | Not saved to results |

### 5.3 NLP Metrics (SFL only)

For NLP tasks in the SFL framework, additional metrics are computed during in-round evaluation:

| Task | Metrics | Location |
|------|---------|----------|
| MRPC, QQP | F1 score + accuracy | `sfl_stage_organizer.py:581-585` |
| CoLA | Matthews correlation coefficient | `sfl_stage_organizer.py:587-589` |
| STS-B | Pearson + Spearman correlation | `sfl_stage_organizer.py:591-593` |
| Others | Accuracy | Default |

---

## 6. GAS-Specific Metrics

### 6.1 V-Value (Gradient Dissimilarity)

Measures how well the cached split-layer activations represent the true data distribution.

```
g_real    = (1/M) × Σ_{m=1}^{M} ∇L(server_model, test_batch_m)
g_sampled = ∇L(server_model, concat_features)

V = (1/|params|) × Σ_p ‖g_sampled_p - g_real_p‖²
```

Where `concat_features` are the split-layer activations from participating clients in the most recent round, and `M` = number of test minibatches (default: 10).

| | |
|---|---|
| **Range** | [0, +∞) — lower is better |
| **Interpretation** | Measures how different the server gradient from cached activations is compared to the "true" gradient on test data |
| **Purpose** | Evaluates the quality of GAS's feature sampling/generation mechanism |
| **When computed** | Every `V_Test_Frequency` rounds (default: 1), only if `V_Test=True` |
| **Location** | `utils/utils.py:57-110` (`calculate_v_value`) |

### 6.2 Split-Layer G (Split G)

GAS computes an additional G metric at the split point (activation gradients):

```
g̃_split_avg = mean(g̃_split_i for each participating client i)
split_G = ‖g̃_split_avg - g*_split‖²
```

| | |
|---|---|
| **What it measures** | How far the average activation-gradient at the cut point is from the oracle |
| **Location** | `GAS_main.py:734-741` |
| **Note** | GAS captures only the first batch for split-layer oracle (memory optimization), while SFL accumulates across all batches |

### 6.3 Logit Local Adjustment

Per-client logit adjustment for label imbalance:

```
label_freq[k] = count of label k in client's data
p[k] = label_freq[k] / Σ label_freq
adjustment[k] = log(p[k]^τ + 1e-12)     where τ = 1 (default)
```

Applied during training as: `loss = CE(output + adjustment, labels)`

| | |
|---|---|
| **Purpose** | Compensates for label frequency imbalance in Non-IID settings |
| **Location** | `utils/utils.py:221-239` (`compute_local_adjustment`) |
| **Saved?** | No — computed once at initialization, used internally |

### 6.4 Activation Statistics (IncrementalStats)

Running statistics of split-layer activations for synthetic feature generation:

```
decay = old_weight / (old_weight + new_weight)
new_mean = decay × old_mean + (1 - decay) × batch_mean

# ResNet (diagonal variance):
new_var = decay × (old_var + (new_mean - old_mean)²)
        + (1 - decay) × (batch_var + (new_mean - batch_mean)²) + 1e-5

# AlexNet (full covariance):
new_cov = decay × (old_cov + outer(diff_old))
        + (1 - decay) × (batch_cov + outer(diff_new)) + 1e-5 × I
```

| | |
|---|---|
| **Granularity** | Per-label, global |
| **Purpose** | Enables generation of synthetic activations for replay when real clients are unavailable |
| **Location** | `GAS_main.py:330-403` (`IncrementalStats`) |
| **Saved?** | No — ephemeral training state |

### 6.5 Communication Time Simulation

Simulates wireless channel latency for heterogeneous clients:

```
path_loss = 128.1 + 37.6 × log10(distance_km)
h = 10^(-path_loss / 10)
rate = W × log2(1 + (P × h) / (W × N₀))

model_process_time     = FLOPs / computing_capacity
transmit_activation_time = (activation_bits × batch_size) / rate
transmit_model_time    = model_bits / rate
```

Where W = 10 MHz, P = 0.2 W, N₀ = 3.981e-21 W/Hz.

| | |
|---|---|
| **Saved as** | `time_record[round] = max(local_models_time)` |
| **Purpose** | Simulates asynchronous SFL communication overhead |
| **Location** | `GAS_main.py:227-327` (Client class) |
| **Enabled when** | `WRTT=True` |

---

## 7. USFL-Specific Metrics

### 7.1 KL Divergence (Class Imbalance)

```
KL_scaled = KL(empirical ‖ uniform) / log(C)
         = (Σ_k p_k × log(p_k × C)) / log(C)
```

Where `p_k` is the empirical class proportion and `C` is the number of classes.

| | |
|---|---|
| **Range** | [0, 1] — 0 = perfectly balanced, 1 = maximally imbalanced |
| **Purpose** | Measures class imbalance of a batch or client's data |
| **Location** | `usfl_stage_organizer.py:291-316` |
| **Saved?** | No — used internally for diagnostics |

### 7.2 Freshness Score (Client Selection)

```
freshness_score = Σ_{label} Σ_{bin} min(amount_to_use, available) × decay_rate^avg_usage
```

Where:
- `avg_usage = (bin_min + bin_max) / 2.0`
- Bins are exponential: [0], [1], [2-3], [4-7], [8-15], ...
- `decay_rate` is configurable (default: 0.95)

| | |
|---|---|
| **Higher = better** | More fresh (less frequently used) data |
| **Purpose** | USFL selector Phase 3 — prioritizes clients with underused data |
| **Location** | `usfl_selector.py:247-289` |
| **Saved?** | Logged to USFLLogger file |

### 7.3 Data Balancing Metrics

| Metric | Description | Location |
|--------|-------------|----------|
| `added_count` | Samples added via replication/target | `usfl_stage_organizer.py:586-697` |
| `removed_count` | Samples removed via trimming/target | Same |
| `augmented_dataset_sizes` | Per-client per-label adjusted data sizes | `usfl_stage_organizer.py:746` |

Saved as `CLIENT_DATA_USAGE_PER_ROUND` event in the SFL result JSON.

### 7.4 Gradient Shuffle Metrics (Adaptive Alpha)

For the `average_adaptive_alpha` gradient shuffle strategy:

```
cos_sim_i = (grad_i · mean_grad) / (‖grad_i‖ × ‖mean_grad‖)
alpha_i   = sigmoid(β × cos_sim_i)
shuffled_i = alpha_i × grad_i + (1 - alpha_i) × mean_grad
```

| | |
|---|---|
| **Purpose** | Per-sample adaptive mixing weight based on gradient alignment |
| **β** | Configurable `adaptive_alpha_beta` parameter |
| **Location** | `usfl_stage_organizer.py:1179-1190` |
| **Saved?** | Logged via print (not in result JSON) |

### 7.5 Aggregation Weights (Label-Capped)

USFL uses a label-aware aggregation instead of standard FedAvg:

```
max_weight[l] = n_l / N          (per-label cap from global distribution)
weight_j = Σ_l max_weight[l] × (n_{l,j} / n_l)
weights = normalize(weights)     (so Σ w_j = 1)
```

| | |
|---|---|
| **Purpose** | Prevents clients with many samples of rare classes from dominating the aggregation |
| **Comparison** | FedAvg uses `w_j = dataset_size_j / total_size` (label-blind) |
| **Location** | `usfl_aggregator.py:55-70` |

---

## 8. MultiSFL-Specific Metrics

### 8.1 FGN (Functional Gradient Norm)

```
FGN_r = mean([-lr_server × grad_norm_sq_per_branch])
```

Where `grad_norm_sq = Σ_param ‖param.grad‖²` computed per branch.

| | |
|---|---|
| **Purpose** | Drives the sampling proportion scheduler — tracks how fast the model is learning |
| **Location** | `multisfl/trainer.py:535-538` |

### 8.2 Sampling Proportion (p_r)

Controls what fraction of training data comes from replay vs. new client data. Updated each round using FGN:

| Mode | Formula |
|------|---------|
| `paper` | `p_{r+1} = p_r × (1 + (FGN_r - FGN_{r-1}) / FGN_{r-1})` |
| `abs_ratio` | `p_{r+1} = p_r × \|FGN_r\| / (\|FGN_{r-1}\| + ε)` |
| `one_plus_delta` | `p_{r+1} = p_r × (1 + clip(δ, -δ_clip, +δ_clip))` where `δ = (FGN_r - FGN_{r-1}) / (\|FGN_{r-1}\| + ε)` |

All clipped to `[p_min, p_max]`.

| | |
|---|---|
| **Location** | `multisfl/scheduler.py:37-79` |
| **Saved?** | Yes — `p_r` per round in result JSON |

### 8.3 Score Vector & Knowledge Replay

**Score Vector (per-branch):**
```
sv = Σ_j γ^{r-j} × L_j / Σ_j γ^{r-j}
```

Where `L_j` is the label distribution at round j and `γ` is a decay factor. This is an exponentially-weighted moving average of historical label distributions.

**Replay Prior (per-branch, per-class):**
```
prior[c] = max(0, mean(sv) - sv[c])
```

Identifies underrepresented classes by comparing each class's score to the mean.

**Quota (replay samples requested):**
```
total = round(base_count × p_r)
q[c]  = round(total × prior[c] / Σ prior)
```

| | |
|---|---|
| **Purpose** | Plans replay of past client data to fill knowledge gaps in each branch |
| **Location** | `multisfl/replay.py:23-81` |
| **Saved?** | `requested`, `collected`, `trials` per round in result JSON |

### 8.4 Server Training Metrics (per-branch)

| Metric | Formula | Purpose |
|--------|---------|---------|
| `grad_norm_sq` | `Σ_param ‖param.grad‖²` | Server gradient magnitude |
| `grad_f_main_norm` | `‖∂L/∂activations‖` | Split-point gradient magnitude |
| `server_param_update_norm` | `√(Σ_param ‖p_after - p_before‖²)` | How much server params changed per step |

Aggregated to per-round means:
- `mean_grad_f_main_norm` — averaged over branches and steps
- `mean_server_update_norm` — same

| **Location** | `multisfl/servers.py:193-215` |
|---|---|

### 8.5 Client Training Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `param_update_norm` | `√(Σ_param ‖p_after - p_before‖²)` | Client model change per step |
| `label_dist` | `count[c] / total_count` | Normalized label distribution of batch |

Aggregated: `mean_client_update_norm` per round.

### 8.6 Soft-Pull Blending

After each round, branch models are blended toward the master:

```
branch_new = (branch + α × master) / (1 + α)
```

Where `α = alpha_master_pull`. This is applied independently to both client-side and server-side branch models.

| | |
|---|---|
| **Purpose** | Prevents branch divergence while allowing specialization |
| **Location** | `multisfl/utils.py:54-64`, `servers.py:66-73, 230-237` |

---

## 9. Configuration Reference

All metric-related configuration lives in `experiment_configs/common.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `enable_g_measurement` | `false` | Enable G measurement system |
| `g_measurement_mode` | `"single"` | Gradient collection mode: `"single"`, `"k_batch"`, `"accumulated"` |
| `g_measurement_k` | `5` | Number of batches for k_batch mode |
| `use_variance_g` | `false` | Enable weighted-variance G decomposition |
| `g_measure_frequency` | `10` | Measure G every N rounds (modulo-based: `(round+1) % N == 0`) |
| `g_oracle_mode` | `"global"` | Oracle mode: `"global"` (single oracle) or `"individual"` (per-model) |
| `enable_drift_measurement` | `true` | Enable drift tracking + A_cos + M_norm |
| `drift_sample_interval` | `1` | Accumulate drift every N local steps (1 = every step) |

### Oracle Mode Mapping

| Common Config | SFL Adapter | GAS Adapter | MultiSFL Adapter |
|---------------|-------------|-------------|------------------|
| `"global"` | (direct) | `"strict"` | `"master"` |
| `"individual"` | (direct) | `"realistic"` | `"branch"` |

---

## 10. Result JSON Schema

### SFL Framework

```jsonc
{
  "config": { /* all Config fields */ },
  "metric": {
    "0": [
      {"timestamp": "...", "event": "PRE_ROUND_START", "params": {}},
      {"event": "CLIENTS_SELECTED", "params": {"client_ids": [...]}},
      {"event": "MODEL_EVALUATED", "params": {"accuracy": 0.85}},
      {"event": "G_MEASUREMENT", "params": {/* RoundGMeasurement */}},
      {"event": "DRIFT_MEASUREMENT", "params": {/* DriftMetrics */}},
      {"event": "CLIENT_DATA_USAGE_PER_ROUND", "params": {/* USFL only */}}
    ]
  },
  "g_measurements": [
    {"round": 9, "params": {
      "server": {"G": ..., "G_rel": ..., "D_cosine": ...},
      "client_G_mean": ..., "client_G_max": ..., "client_D_mean": ...,
      "per_client": {/* client_id -> {G, G_rel, D_cosine} */},
      "split_layer": {"G": ..., "G_rel": ..., "D_cosine": ...},
      "variance_client_g": ..., "variance_client_g_rel": ...,
      "variance_server_g": ..., "variance_server_g_rel": ...
    }}
  ],
  "drift_history": {
    "G_drift": [...],                      // per-round (legacy alias)
    "G_drift_client": [...],
    "G_drift_client_stepweighted": [...],
    "G_end": [...],                        // legacy alias
    "G_end_client": [...],
    "G_end_client_weighted": [...],
    "G_drift_norm": [...],                 // legacy alias
    "G_drift_norm_client": [...],
    "delta_global_norm_sq": [...],         // legacy alias
    "delta_client_norm_sq": [...],
    "D_dir_client_weighted": [...],
    "D_rel_client_weighted": [...],
    "G_drift_server": [...],
    "G_end_server": [...],
    "G_drift_norm_server": [...],
    "delta_server_norm_sq": [...],
    "G_drift_total": [...],
    "G_end_total": [...],
    "A_cos": [...],
    "M_norm": [...],
    "n_valid_alignment": [...],
    "per_round": [/* full DriftMetrics per round */]
  }
}
```

### GAS Framework

```jsonc
{
  "config": { /* experiment settings */ },
  "accuracy": [...],                       // per-round
  "v_value": [...],                        // per-round (if V_Test=True)
  "time_record": [...],                    // per-round (if WRTT=True)
  "g_history": {                           // NOTE: different key from SFL!
    "client_g": [...],
    "client_g_rel": [...],
    "client_d": [...],
    "server_g": [...],
    "server_g_rel": [...],
    "server_d": [...],
    "split_g": [...],
    "variance_client_g": [...],
    "variance_client_g_rel": [...],
    "variance_server_g": [...],
    "variance_server_g_rel": [...],
    "per_client_g": [{/* client_id -> {G, G_rel, D} */}],
    "per_server_g": [{/* {G, G_rel, D} */}]
  },
  "drift_history": { /* same structure as SFL */ }
}
```

### MultiSFL Framework

```jsonc
{
  "config": { /* CLI args */ },
  "rounds": [
    {
      "round": 0,
      "accuracy": 0.10,
      "p_r": 0.5,
      "fgn_r": -0.001,
      "requested": 100,
      "collected": 95,
      "trials": 120,
      "mean_grad_f_main_norm": 0.05,
      "mean_client_update_norm": 0.02,
      "mean_server_update_norm": 0.03,
      // If G measurement enabled:
      "client_g": ..., "client_g_rel": ...,
      "server_g": ..., "server_g_rel": ...,
      "per_client_g": {...}, "per_branch_server_g": {...}
    }
  ],
  "summary": {
    "final_accuracy": 0.85,
    "best_accuracy": 0.87,
    "total_requested": 10000,
    "total_collected": 9500
  },
  "g_measurements": [/* detailed G data per diagnostic round */],
  "drift_history": {
    // Same as SFL/GAS but with additional:
    "A_cos_client": [...],
    "M_norm_client": [...],
    "A_cos_server": [...],       // unique to MultiSFL
    "M_norm_server": [...]       // unique to MultiSFL
  }
}
```

### Known Inconsistencies

| Issue | Details |
|-------|---------|
| G measurement key name | GAS uses `g_history`, SFL/MultiSFL use `g_measurements` |
| Accuracy location | SFL: inside `metric` events, GAS: top-level `accuracy` array, MultiSFL: inside `rounds` array |
| A_cos scope | SFL/GAS: single `A_cos`, MultiSFL: separate `A_cos_client` + `A_cos_server` |
