# Unified Simulation Framework — Implementation Design Document

> **Purpose**: This document is a self-contained specification for implementing a lightweight, unified Split Federated Learning (SFL) simulation framework. An agent with **no prior context** about this repository should be able to read this document and implement the framework end-to-end.
>
> **Date**: 2026-03-12
> **Location**: `unified-simulation-framework/` at repo root (`USFL-fork/`)

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Current State: Three Separate Frameworks](#2-current-state-three-separate-frameworks)
3. [Target Architecture Overview](#3-target-architecture-overview)
4. [Component Reuse Map](#4-component-reuse-map)
5. [Directory Structure](#5-directory-structure)
6. [Core Interfaces & Data Structures](#6-core-interfaces--data-structures)
7. [File-by-File Specifications](#7-file-by-file-specifications)
8. [Method Hook Specifications](#8-method-hook-specifications)
9. [Implementation Phases](#9-implementation-phases)
10. [Validation Strategy](#10-validation-strategy)
11. [Integration with Experiment Infrastructure](#11-integration-with-experiment-infrastructure)
12. [**ERRATA: Critical Design Corrections (Post-Review)**](#12-errata-critical-design-corrections-post-review)
13. [Appendix: SFL Training Flow Explained](#appendix-a-sfl-training-flow-explained)
14. [Appendix: Existing Code Reference](#appendix-b-existing-code-reference)

> **IMPORTANT**: Section 12 (ERRATA) contains critical corrections discovered during code review.
> These corrections OVERRIDE the earlier sections. Read Section 12 before implementing.

---

## 1. Background & Motivation

### What is Split Federated Learning (SFL)?

In SFL, a neural network is **split** between client and server:

```
Full Model:  [Input] → [Layer1, Layer2] → [Layer3, Layer4] → [Output]
                        ╰── client_model ──╯  ╰── server_model ──╯

Training flow (one client, one iteration):
  1. Client forward:  x → client_model(x) → activation
  2. Server forward:  activation → server_model(activation) → logits
  3. Server backward: loss(logits, labels).backward() → activation_grad
  4. Client backward: activation.backward(activation_grad) → update client_model
  5. Server update:   optimizer.step() → update server_model
```

Multiple clients participate per round. After all clients train, their client models are **aggregated** (e.g., FedAvg) into a new global model for the next round.

### Why a new framework?

The repository currently has **three separate implementations** of SFL variants:

| Framework | Location | Methods | Lines | Problem |
|-----------|----------|---------|-------|---------|
| SFL Framework | `sfl_framework-fork-feature-training-tracker/` | SFL, USFL, SCAFFOLD, Mix2SFL, FedCBS | ~31,500 | Designed for real-device emulation; simulation mode carries massive communication protocol overhead (async queues, polling loops, message wrapping) |
| GAS | `GAS_implementation/` | GAS | ~4,500 | Standalone monolithic file; duplicates model/data code |
| MultiSFL | `multisfl_implementation/` | MultiSFL | ~3,500 | Standalone package; duplicates model/data code |

**Problems:**
1. **60% of SFL framework's stage organizer code is communication protocol**, not ML logic
2. **Polling anti-pattern**: `while True: await asyncio.sleep(0)` causes millions of unnecessary context switches in simulation
3. **Unnecessary model copies**: `copy.deepcopy(model)` per client per round
4. **Three separate codebases** for the same fundamental SFL operation
5. **Adding a new method requires understanding 5,000+ lines** of communication-entangled code

**Goal**: One lightweight framework, ~1,500 lines of new code, pure synchronous Python, all 7 methods supported.

---

## 2. Current State: Three Separate Frameworks

### 2.1 SFL Framework (`sfl_framework-fork-feature-training-tracker/`)

**Entry point**: `simulation.py` (665 lines)

**Architecture**:
```
simulation.py
  → run_simulation(server_cmd, client_cmds)
    → asyncio.gather(server.run(), client0.run(), client1.run(), ...)
      → Server: Trainer → StageOrganizer._pre_round/_in_round/_post_round
        → PreRound: poll for clients → split model → send via queue
        → InRound: poll for activations → forward/backward → send grad via queue
        → PostRound: aggregate → evaluate
      → Client: poll for model → forward → send activation → poll for grad → backward
```

**Methods supported**: SFL, USFL, SCAFFOLD_SFL, Mix2SFL, FedCBS (+ FL, FedProx, etc.)

**Key files for ML logic extraction**:
- `server/modules/trainer/stage/sfl_stage_organizer.py` (694 lines) — Base SFL
- `server/modules/trainer/stage/usfl_stage_organizer.py` (1,642 lines) — USFL with 6 optimizations
- `server/modules/trainer/stage/scaffold_stage_organizer.py` — SCAFFOLD control variates
- `server/modules/trainer/stage/mix2sfl_stage_organizer.py` — Mix2SFL SmashMix+GradMix
- `client/modules/trainer/model_trainer/sfl_model_trainer.py` — Client-side SFL training
- `client/modules/trainer/model_trainer/usfl_model_trainer.py` — Client-side USFL training

**Reusable components** (zero communication dependency):
- `server/modules/model/` — All model classes (ResNet, VGG, AlexNet, DeiT, etc.)
- `server/modules/trainer/splitter/` — Model splitting strategies
- `server/modules/trainer/seletor/` — Client selection (uniform, usfl, fedcbs, missing_class)
- `server/modules/trainer/aggregator/` — Aggregation (fedavg, usfl, fitfl)
- `server/modules/trainer/distributer/` — Data distribution (uniform, dirichlet, shard_dirichlet, etc.)
- `server/modules/dataset/` — Dataset loading (CIFAR, MNIST, FMNIST, GLUE)
- `server/modules/trainer/scheduler/batch_scheduler.py` — USFL dynamic batch scheduling
- `client/modules/dataset/maskable_dataset.py` — USFL data balancing
- `server/modules/trainer/utils/training_tracker.py` — Metrics logging
- `server/server_args.py` — Config dataclass (223 fields)

### 2.2 GAS Framework (`GAS_implementation/`)

**Entry point**: `GAS_main.py` (1,697 lines — monolithic single file)

**Architecture**:
```
GAS_main.py
  → Main loop: for epoch in range(epochs):
    → Select client (time-based WRTT or fixed schedule)
    → Client forward → buffer activation
    → When buffer full: server forward/backward on concatenated buffer
    → Client backward
    → When all clients done: FedAvg aggregate
    → Optional: G measurement, drift measurement, feature generation
```

**GAS-specific features** (must be preserved):
1. **Logit local adjustment**: Per-client label frequency correction added to logits before loss
2. **Feature sampling/generation**: Synthesize activations from Gaussian statistics when real data is insufficient
3. **IncrementalStats**: Welford's algorithm for running mean/variance of activations per label
4. **V-value computation**: Gradient dissimilarity metric (diagnostic, not used for selection)
5. **Time-based client selection (WRTT)**: Simulates communication delay for client ordering

**Key files**:
- `GAS_main.py` — Everything (training loop, config, client selection)
- `utils/network.py` (1,672 lines) — Model definitions and splitting
- `utils/dataset.py` (557 lines) — Data partitioning
- `utils/g_measurement.py` (603 lines) — G measurement system
- `utils/utils.py` — V-value, feature generation, local adjustment
- `utils/drift_measurement.py` — Drift tracking

### 2.3 MultiSFL Framework (`multisfl_implementation/`)

**Entry point**: `run_multisfl.py`

**Architecture**:
```
run_multisfl.py
  → MultiSFLTrainer.run()
    → For each round:
      → Select n_main clients, map to B branches
      → For each branch b (sequentially):
        → For each local_step:
          → Client forward (branch's client model)
          → Knowledge replay: collect data from inactive clients
          → Server train (branch's server model) on main + replay data
          → Client backward
        → Track per-branch scores
      → Master aggregation: average across B branches
      → Soft pull: blend branches back toward master (α-blend)
      → Update replay proportion scheduler (p_r)
      → Evaluate master model
```

**MultiSFL-specific features** (must be preserved):
1. **B independent branches**: Each branch has its own client_model + server_model + optimizers
2. **FedServer + MainServer**: Hierarchical aggregation with `compute_master()` and `soft_pull_to_master()`
3. **Knowledge replay**: `ScoreVectorTracker` tracks per-branch label history; `KnowledgeRequestPlanner` computes per-class replay quotas; inactive clients provide replay data
4. **SamplingProportionScheduler**: Adjusts replay proportion (p_r) based on FGN (Falsified Gradient Norm)
5. **Soft pull**: `new_state = (old_state + α * master_state) / (1 + α)` — gentle blending, not hard replacement

**Key files**:
- `multisfl/trainer.py` — MultiSFLTrainer (main loop, ~770 lines)
- `multisfl/servers.py` — FedServer, MainServer (aggregation, server training)
- `multisfl/client.py` — Client (forward, backward, replay sampling)
- `multisfl/replay.py` — ScoreVectorTracker, KnowledgeRequestPlanner
- `multisfl/scheduler.py` — SamplingProportionScheduler
- `multisfl/models.py` — Model splitting
- `multisfl/data.py` — Data partitioning

---

## 3. Target Architecture Overview

### Design Principles

1. **No async, no queues, no polling** — Pure synchronous Python function calls
2. **No message wrapping** — Pass tensors directly, not dicts-in-queues
3. **Minimal copies** — `load_state_dict()` instead of `deepcopy()`; share model references when safe
4. **Strategy pattern** — Each method is a "hook" class with 3-5 small methods
5. **Reuse existing components** — Import models, splitters, selectors, etc. from SFL framework
6. **Coexist with existing frameworks** — Don't modify existing code; add new code only

### Conceptual Architecture

```
┌─ unified-simulation-framework/ ─────────────────────────────────────────┐
│                                                                          │
│  entry.py ──→ SimTrainer ──→ MethodHook (strategy pattern)               │
│                  │                │                                       │
│                  │                ├── SFLHook                             │
│                  │                ├── USFLHook                            │
│                  │                ├── ScaffoldHook                        │
│                  │                ├── Mix2SFLHook                         │
│                  │                ├── GASHook                             │
│                  │                └── MultiSFLHook                        │
│                  │                                                       │
│                  ├── _run_sfl_round()  ← shared by ALL methods           │
│                  │     │                                                  │
│                  │     ├── client_forward()   ← client_ops.py            │
│                  │     ├── server_forward()                               │
│                  │     ├── server_backward()                              │
│                  │     ├── hook.process_gradients()  ← method-specific    │
│                  │     └── client_backward()  ← client_ops.py            │
│                  │                                                       │
│                  └── imports from sfl_framework:                          │
│                        models, splitters, selectors, aggregators,         │
│                        distributers, datasets, maskable_dataset,          │
│                        batch_scheduler, training_tracker                  │
│                                                                          │
│  Additional imports:                                                     │
│    GAS: utils.py (feature_gen, v_value, local_adj), g_measurement.py     │
│    MultiSFL: replay.py, scheduler.py, servers.py (aggregation logic)     │
│    Shared: update_alignment.py, drift_measurement                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Two Execution Modes

```python
# Mode 1: Single-branch (SFL, USFL, SCAFFOLD, Mix2SFL, GAS)
for round in rounds:
    ctx = hook.pre_round(round)
    results = trainer._run_sfl_round(ctx)
    hook.post_round(round, ctx, results)

# Mode 2: Multi-branch (MultiSFL)
for round in rounds:
    hook.pre_round(round)           # select clients, setup branches
    for branch in range(B):
        ctx = hook.pre_branch(round, branch)  # map client, prepare replay
        results = trainer._run_sfl_round(ctx) # SAME core loop!
        hook.post_branch(round, branch, results)
    hook.post_round(round)          # master aggregation, soft pull, evaluate
```

The key insight: `_run_sfl_round()` is **identical** for all 7 methods. All differentiation happens in hooks.

---

## 4. Component Reuse Map

### What to IMPORT (not copy) from existing frameworks

All paths are relative to repo root (`USFL-fork/`).

#### From SFL Framework (`sfl_framework-fork-feature-training-tracker/`)

| Component | Import Path | Interface | Notes |
|-----------|-------------|-----------|-------|
| **Models** | `server.modules.model.model.get_model` | `get_model(config) → BaseModel` | All architectures: ResNet18, FlexibleResNet, VGG11, AlexNet, DeiT, DistilBERT, MobileNet, LeNet |
| **BaseModel** | `server.modules.model.base_model.BaseModel` | `.forward()`, `.evaluate()`, `.get_torch_model()`, `.set_torch_model()` | Pure nn.Module wrapper |
| **Splitters** | `server.modules.trainer.splitter.splitter.get_splitter` | `get_splitter(config) → BaseSplitter`; `.split(model, params) → List[ModuleDict]` | Strategies: layer_name, ratio_param, ratio_layer |
| **Selectors** | `server.modules.trainer.seletor.selector.get_selector` | `get_selector(config) → BaseSelector`; `.select(n, client_ids, data) → List[int]` | Types: uniform, usfl, missing_class, fedcbs |
| **Aggregators** | `server.modules.trainer.aggregator.aggregator.get_aggregator` | `get_aggregator(config) → BaseAggregator`; `.aggregate(models, params) → Model` | Types: fedavg, usfl, fitfl |
| **Distributers** | `server.modules.trainer.distributer.distributer.get_distributer` | `get_distributer(config) → BaseDistributer`; `.distribute(dataset, clients) → Dict[int, List[int]]` | Types: uniform, dirichlet, label, shard_dirichlet |
| **Datasets** | `server.modules.dataset.dataset.get_dataset` | `get_dataset(config) → BaseDataset`; `.get_trainset()`, `.get_testset()`, `.get_num_classes()` | CIFAR-10/100, MNIST, FMNIST, GLUE |
| **MaskableDataset** | `client.modules.dataset.maskable_dataset.MaskableDataset` | `MaskableDataset(dataset, indices)`; `.update_amount_per_label(dict)`, `.get_label_distribution()` | Data balancing (trimming/replication) |
| **BatchScheduler** | `server.modules.trainer.scheduler.batch_scheduler.create_schedule` | `create_schedule(B, C_list) → (k, schedule)` | USFL dynamic batch sizing |
| **TrainingTracker** | `server.modules.trainer.utils.training_tracker.TrainingTracker` | `.initialize()`, `.start_round()`, `.log_iteration_data()`, `.log_aggregation_weights()` | Metrics logging |
| **Config** | `server.server_args.Config`, `server.server_args.parse_args` | Dataclass with 223 fields; `parse_args(argv) → Config` | Argparse-based config |
| **G Measurement** | `server.utils.g_measurement` | `GMeasurementSystem` class (if exists), or raw functions | Gradient quality metrics |

#### From GAS Framework (`GAS_implementation/`)

| Component | Import Path | Interface | Notes |
|-----------|-------------|-----------|-------|
| **Feature generation** | `utils.utils.sample_or_generate_features` | `(concat_features, concat_labels, batchsize, num_labels, shape, device, stats, diagonal) → (features, labels)` | Gaussian synthesis from running statistics |
| **Local adjustment** | `utils.utils.compute_local_adjustment` | `(train_loader, device) → Tensor` | Log-frequency label correction |
| **V-value** | `utils.utils.calculate_v_value` | `(sampled_grads, oracle_grads) → float` | Gradient dissimilarity metric |
| **G measurement** | `utils.g_measurement` | `compute_oracle_gradients()`, `compute_g_score()` | 3-perspective gradient quality |
| **Drift tracker** | `utils.drift_measurement.DriftMeasurementTracker` | `.on_round_start()`, `.accumulate_client_drift()`, `.on_round_end()` | Per-round drift metrics |

#### From MultiSFL Framework (`multisfl_implementation/`)

| Component | Import Path | Interface | Notes |
|-----------|-------------|-----------|-------|
| **ScoreVectorTracker** | `multisfl.replay.ScoreVectorTracker` | `.append_label_dist(branch, dist)`, `.score_vector(branch) → array` | Per-branch label history |
| **KnowledgeRequestPlanner** | `multisfl.replay.KnowledgeRequestPlanner` | `.plan(sv, p_r, base_count, min_total) → ReplayRequest` | Per-class replay quotas |
| **SamplingProportionScheduler** | `multisfl.scheduler.SamplingProportionScheduler` | `.update(fgn) → float` | Adaptive p_r |

#### From Shared (`shared/`)

| Component | Import Path | Interface |
|-----------|-------------|-----------|
| **Update alignment** | `shared.update_alignment` | `flatten_delta()`, `compute_update_alignment()` → A_cos, M_norm |

### What to WRITE NEW

Only the training loop orchestration and method hooks. Everything else is imported.

---

## 5. Directory Structure

```
unified-simulation-framework/
├── DESIGN.md                       # This document
├── __init__.py                     # Package marker
├── trainer.py                      # SimTrainer: main training loop (~500 lines)
├── client_ops.py                   # Client forward/backward helpers (~200 lines)
├── config.py                       # Unified config (extends/wraps SFL Config) (~100 lines)
├── entry.py                        # CLI entry point + result saving (~150 lines)
├── methods/
│   ├── __init__.py                 # Method registry (~30 lines)
│   ├── base.py                     # BaseMethodHook ABC (~50 lines)
│   ├── sfl.py                      # SFL hook (~80 lines)
│   ├── usfl.py                     # USFL hook — 6 optimizations (~300 lines)
│   ├── scaffold.py                 # SCAFFOLD hook — control variates (~150 lines)
│   ├── mix2sfl.py                  # Mix2SFL hook — SmashMix + GradMix (~180 lines)
│   ├── gas.py                      # GAS hook — feature gen, logit adj (~250 lines)
│   └── multisfl.py                 # MultiSFL hook — multi-branch, replay (~350 lines)
└── tests/
    ├── test_sfl_match.py           # Validates SFL output matches original framework
    ├── test_usfl_match.py          # Validates USFL output matches original framework
    └── ...
```

**Estimated total new code: ~1,840 lines**

---

## 6. Core Interfaces & Data Structures

### 6.1 Data Structures

```python
# client_ops.py

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ClientState:
    """Per-client training state for one round."""
    client_id: int
    client_model: nn.Module
    optimizer: torch.optim.Optimizer
    dataloader: DataLoader
    data_iter: Any                          # iter(dataloader), reset per round
    label_distribution: Dict[int, int]      # {label: count}
    dataset_size: int
    # Optional measurement state
    g_accumulator: Optional[Any] = None     # For G measurement
    drift_snapshots: Optional[list] = None  # For drift measurement


@dataclass
class ClientResult:
    """Result of one client's training in a round."""
    client_id: int
    model_state_dict: dict                  # state_dict snapshot for aggregation
    dataset_size: int
    label_distribution: Dict[int, int]
    # Optional metrics
    g_grads: Optional[dict] = None          # Collected gradients for G measurement
    drift_metrics: Optional[dict] = None    # Drift trajectory data


@dataclass
class RoundContext:
    """Everything needed to execute one round (or one branch-round)."""
    round_number: int
    selected_client_ids: List[int]
    client_states: Dict[int, ClientState]
    server_model: nn.Module
    server_optimizer: torch.optim.Optimizer
    criterion: nn.Module
    iterations: int                         # Number of local iterations
    device: torch.device
    # Method-specific (set by hooks)
    batch_schedule: Optional[dict] = None   # USFL: per-client per-iteration batch sizes
    extra: dict = field(default_factory=dict)  # Arbitrary hook data


@dataclass
class RoundResult:
    """Aggregated result of one round."""
    round_number: int
    accuracy: float
    loss: float
    client_results: List[ClientResult]
    metrics: dict = field(default_factory=dict)  # G, drift, etc.
```

### 6.2 BaseMethodHook Interface

```python
# methods/base.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..trainer import SimTrainer

class BaseMethodHook(ABC):
    """
    Strategy interface for method-specific behavior.

    The SimTrainer calls these methods at specific points in the training loop.
    Each method (SFL, USFL, GAS, etc.) implements this interface.

    Lifecycle for single-branch methods (SFL, USFL, SCAFFOLD, Mix2SFL, GAS):
        for round in rounds:
            ctx = hook.pre_round(trainer, round)
            results = trainer._run_sfl_round(ctx)
            round_result = hook.post_round(trainer, round, ctx, results)

    Lifecycle for multi-branch methods (MultiSFL):
        for round in rounds:
            hook.pre_round(trainer, round)
            for branch in range(B):
                ctx = hook.pre_branch(trainer, round, branch)
                results = trainer._run_sfl_round(ctx)
                hook.post_branch(trainer, round, branch, ctx, results)
            round_result = hook.post_round_multi(trainer, round)
    """

    def __init__(self, config, trainer_resources: dict):
        """
        Args:
            config: Parsed Config object
            trainer_resources: Dict with shared resources:
                - 'model': BaseModel (global model)
                - 'dataset': BaseDataset
                - 'distributer': BaseDistributer
                - 'selector': BaseSelector
                - 'aggregator': BaseAggregator
                - 'splitter': BaseSplitter
                - 'client_data_masks': Dict[int, List[int]]
                - 'testloader': DataLoader
                - 'num_classes': int
        """
        self.config = config
        self.resources = trainer_resources

    @abstractmethod
    def pre_round(self, trainer: 'SimTrainer', round_number: int) -> 'RoundContext':
        """
        Prepare for a round of training.

        Responsibilities:
        - Select clients (using selector)
        - Split model into client_model and server_model
        - Create ClientState for each selected client
        - Apply data balancing (if applicable)
        - Create and return RoundContext

        Returns: RoundContext ready for _run_sfl_round()
        """
        ...

    def process_gradients(
        self,
        activation_grads: torch.Tensor,
        activations: List[torch.Tensor],
        labels: torch.Tensor,
        round_ctx: 'RoundContext',
    ) -> torch.Tensor:
        """
        Post-process server-computed activation gradients before sending to clients.

        Called after server backward, before client backward.
        Default: return gradients unchanged (pass-through).

        Override for:
        - USFL: gradient shuffle (random, inplace, average, adaptive_alpha)
        - Mix2SFL: GradMix
        - GAS: logit adjustment is applied at loss level, not here

        Args:
            activation_grads: Gradient w.r.t. concatenated activations [total_batch, ...]
            activations: List of per-client activation tensors
            labels: Concatenated labels
            round_ctx: Current round context

        Returns: Modified activation_grads (same shape)
        """
        return activation_grads

    @abstractmethod
    def post_round(
        self,
        trainer: 'SimTrainer',
        round_number: int,
        round_ctx: 'RoundContext',
        client_results: List['ClientResult'],
    ) -> 'RoundResult':
        """
        Finalize a round of training.

        Responsibilities:
        - Aggregate client models (using aggregator)
        - Update global model
        - Evaluate on test set
        - Update method-specific state (cumulative usage, control variates, etc.)
        - Collect and return metrics

        Returns: RoundResult with accuracy, loss, and metrics
        """
        ...

    # --- Multi-branch methods only (MultiSFL) ---

    @property
    def is_multi_branch(self) -> bool:
        """Return True if this method uses multi-branch training."""
        return False

    def pre_branch(
        self, trainer: 'SimTrainer', round_number: int, branch: int
    ) -> 'RoundContext':
        """Prepare one branch's training context. Only called if is_multi_branch."""
        raise NotImplementedError

    def post_branch(
        self, trainer: 'SimTrainer', round_number: int, branch: int,
        round_ctx: 'RoundContext', client_results: List['ClientResult'],
    ):
        """Finalize one branch's training. Only called if is_multi_branch."""
        raise NotImplementedError

    def post_round_multi(
        self, trainer: 'SimTrainer', round_number: int
    ) -> 'RoundResult':
        """Finalize a multi-branch round (master aggregation, evaluate). Only called if is_multi_branch."""
        raise NotImplementedError
```

### 6.3 SimTrainer Interface

```python
# trainer.py (public interface only; implementation in Section 7)

class SimTrainer:
    """
    Lightweight simulation-only SFL trainer.

    Eliminates ALL communication overhead:
    - No async/await (pure synchronous)
    - No queues or polling
    - No message wrapping/unwrapping
    - Minimal model copies (load_state_dict instead of deepcopy)
    """

    def __init__(self, config):
        """
        Initialize from Config dataclass.

        Sets up: dataset, model, distributer, selector, aggregator,
        splitter, method hook, G measurement, drift tracker.
        """

    def train(self) -> List[RoundResult]:
        """
        Main entry point. Runs all rounds and returns results.

        Dispatches to single-branch or multi-branch loop based on hook.is_multi_branch.
        """

    def _run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Execute one round of SFL training. SHARED BY ALL METHODS.

        For each iteration:
          1. For each client: client_forward() → activation
          2. Concatenate all activations
          3. Server forward → logits → loss
          4. Server backward → activation_grads
          5. hook.process_gradients() → modified grads
          6. For each client: client_backward(grad_slice)

        After all iterations:
          Snapshot each client's model state_dict → ClientResult
        """
```

---

## 7. File-by-File Specifications

### 7.1 `__init__.py`

```python
"""Unified Simulation Framework for Split Federated Learning."""
```

### 7.2 `config.py` (~100 lines)

**Purpose**: Unified configuration that works across all 7 methods.

**Approach**: Reuse the existing `Config` dataclass from the SFL framework's `server_args.py` as the base. Add GAS-specific and MultiSFL-specific fields.

```python
"""
Unified configuration for all SFL methods.

Strategy: Extend the existing SFL Config with fields for GAS and MultiSFL.
The SFL Config already covers SFL, USFL, SCAFFOLD, Mix2SFL via argparse.
"""
import sys
from dataclasses import dataclass, field
from typing import Optional

# Add SFL framework to path for imports
SFL_FRAMEWORK_PATH = ...  # resolved relative to this file
sys.path.insert(0, SFL_FRAMEWORK_PATH)

from server.server_args import Config as SFLConfig, parse_args as sfl_parse_args


@dataclass
class UnifiedConfig:
    """
    Wraps SFL Config and adds GAS + MultiSFL fields.

    For SFL/USFL/SCAFFOLD/Mix2SFL: all fields come from SFLConfig.
    For GAS: additional fields below.
    For MultiSFL: additional fields below.
    """
    # Base SFL config (contains all SFL/USFL/SCAFFOLD/Mix2SFL fields)
    sfl_config: SFLConfig = None

    # Method identifier (unified across all frameworks)
    method: str = "sfl"  # sfl, usfl, scaffold_sfl, mix2sfl, gas, multisfl

    # --- GAS-specific ---
    gas_generate_features: bool = False
    gas_sample_frequency: int = 5
    gas_logit_adjustment: bool = True
    gas_use_wrtt: bool = False
    gas_use_variance_g: bool = False
    gas_v_test: bool = False
    gas_diagonal_covariance: bool = True

    # --- MultiSFL-specific ---
    multisfl_branches: int = 3
    multisfl_alpha_master_pull: float = 0.1
    multisfl_p0: float = 0.3                    # Initial replay proportion
    multisfl_p_min: float = 0.05
    multisfl_p_max: float = 0.8
    multisfl_p_update: str = "abs_ratio"        # paper, abs_ratio, one_plus_delta
    multisfl_gamma: float = 0.5                 # Score decay
    multisfl_replay_min_total: int = 10
    multisfl_max_assistant_trials: int = 5
    multisfl_local_steps: int = 5

    # --- Common overrides (for fields shared across all methods) ---
    # These map to SFLConfig fields but provide a unified namespace
    @property
    def dataset(self): return self.sfl_config.dataset
    @property
    def model(self): return self.sfl_config.model
    @property
    def device(self): return self.sfl_config.device
    @property
    def global_round(self): return self.sfl_config.global_round
    @property
    def num_clients(self): return self.sfl_config.num_clients
    @property
    def num_clients_per_round(self): return self.sfl_config.num_clients_per_round
    @property
    def batch_size(self): return self.sfl_config.batch_size
    @property
    def local_epochs(self): return self.sfl_config.local_epochs
    # ... (delegate all common fields)


def parse_unified_config(method: str, sfl_args: list[str], extra_args: dict = None) -> UnifiedConfig:
    """
    Parse unified config from SFL-style CLI args + method-specific extras.

    Args:
        method: One of 'sfl', 'usfl', 'scaffold_sfl', 'mix2sfl', 'gas', 'multisfl'
        sfl_args: CLI args in SFL format (e.g., ['-d', 'cifar10', '-m', 'resnet18', ...])
        extra_args: Dict of method-specific overrides (e.g., {'gas_generate_features': True})

    Returns: UnifiedConfig
    """
    sfl_config = sfl_parse_args(sfl_args)
    config = UnifiedConfig(sfl_config=sfl_config, method=method)
    if extra_args:
        for k, v in extra_args.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return config
```

**Key decisions**:
- Don't duplicate the 223 SFL Config fields; wrap and delegate
- GAS/MultiSFL fields are flat attributes on UnifiedConfig
- `parse_unified_config()` is the single entry point for all methods

### 7.3 `client_ops.py` (~200 lines)

**Purpose**: Stateless helper functions for client-side SFL operations.

```python
"""
Client-side SFL operations: forward pass, backward pass, state management.

These are pure functions operating on ClientState — no communication, no async.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional, Dict
from .config import UnifiedConfig


def create_client_state(
    client_id: int,
    client_model: nn.Module,
    full_dataset,
    data_indices: list,
    config: UnifiedConfig,
    batch_size: int = None,
    augmented_sizes: dict = None,  # USFL: {label: target_count}
) -> 'ClientState':
    """
    Create a ClientState for one client in one round.

    Args:
        client_id: Client identifier
        client_model: The client-side split model (will be trained)
        full_dataset: Full training dataset
        data_indices: Indices assigned to this client (from distributer)
        config: Unified config
        batch_size: Override batch size (for USFL dynamic scheduling)
        augmented_sizes: If provided, apply data balancing via MaskableDataset

    Returns: ClientState ready for training

    Implementation notes:
        - Import MaskableDataset from SFL framework
        - If augmented_sizes provided, call dataset.update_amount_per_label(augmented_sizes)
        - Create DataLoader with shuffle=True
        - Create optimizer based on config (SGD, Adam, AdamW)
    """
    ...


def client_forward(
    state: 'ClientState',
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run client-side forward pass.

    Args:
        state: ClientState with client_model
        batch: (images, labels) tuple
        device: Target device (cuda/cpu)

    Returns: (activation, labels)
        - activation: Output of client_model, with grad enabled (requires_grad=True)
        - labels: Labels tensor on device

    Implementation notes:
        - images, labels = batch
        - images = images.to(device)
        - labels = labels.to(device)
        - activation = state.client_model(images)
        - If activation is tuple (e.g., FlexibleResNet returns (act, identity)):
            handle appropriately — store identity in state for backward
        - activation.requires_grad_(True)  # Enable gradient flow at split point
        - activation.retain_grad()         # Keep grad after backward
    """
    ...


def client_backward(
    state: 'ClientState',
    activation: torch.Tensor,
    activation_grad: torch.Tensor,
    config: UnifiedConfig,
):
    """
    Run client-side backward pass and optimizer step.

    Args:
        state: ClientState with client_model and optimizer
        activation: The activation tensor from client_forward (with grad_fn)
        activation_grad: Gradient from server's backward pass
        config: For clip_grad settings

    Implementation notes:
        - state.optimizer.zero_grad()
        - activation.backward(activation_grad)
        - Optional: torch.nn.utils.clip_grad_norm_(state.client_model.parameters(), max_norm)
        - state.optimizer.step()
    """
    ...


def get_next_batch(state: 'ClientState') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get next batch from client's data iterator. Reset if exhausted.

    Returns: (images, labels) tuple
    """
    ...


def snapshot_model(model: nn.Module) -> dict:
    """
    Create a detached copy of model's state_dict.

    Cheaper than deepcopy: only copies tensor data, not the module graph.
    """
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def restore_model(model: nn.Module, state_dict: dict):
    """Restore model parameters from snapshot."""
    model.load_state_dict(state_dict)
```

### 7.4 `trainer.py` (~500 lines)

**Purpose**: Main training loop. The heart of the framework.

```python
"""
SimTrainer: Lightweight simulation-only SFL trainer.

This is the core training loop that ALL methods share. Method-specific behavior
is injected via MethodHook instances (strategy pattern).
"""
import sys
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader

from .config import UnifiedConfig
from .client_ops import (
    ClientState, ClientResult, RoundContext, RoundResult,
    client_forward, client_backward, get_next_batch, snapshot_model, restore_model,
)
from .methods import get_method_hook

# Imports from SFL framework (path setup in config.py)
from server.modules.model.model import get_model
from server.modules.dataset.dataset import get_dataset
from server.modules.trainer.splitter.splitter import get_splitter
from server.modules.trainer.seletor.selector import get_selector
from server.modules.trainer.aggregator.aggregator import get_aggregator
from server.modules.trainer.distributer.distributer import get_distributer


class SimTrainer:

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: List[RoundResult] = []

        # Initialize components (reuse existing factory functions)
        self._init_components()

        # Initialize method hook
        resources = {
            'model': self.model,
            'dataset': self.dataset_obj,
            'trainset': self.trainset,
            'testloader': self.testloader,
            'distributer': self.distributer,
            'selector': self.selector,
            'aggregator': self.aggregator,
            'splitter': self.splitter,
            'client_data_masks': self.client_data_masks,
            'num_classes': self.num_classes,
        }
        self.hook = get_method_hook(config.method, config, resources)

    def _init_components(self):
        """Initialize dataset, model, distributer, selector, aggregator, splitter."""
        cfg = self.config.sfl_config  # SFL Config for factory functions

        # Dataset
        self.dataset_obj = get_dataset(cfg)
        self.dataset_obj.initialize()
        self.trainset = self.dataset_obj.get_trainset()
        self.testloader = self.dataset_obj.get_testloader(cfg.batch_size)
        self.num_classes = self.dataset_obj.get_num_classes()

        # Model
        self.model = get_model(cfg)
        self.model.get_torch_model().to(self.device)

        # Distributer: partition data among clients
        self.distributer = get_distributer(cfg)
        client_ids = list(range(cfg.num_clients))
        self.client_data_masks = self.distributer.distribute(self.trainset, client_ids)

        # Selector, Aggregator, Splitter
        self.selector = get_selector(cfg)
        self.aggregator = get_aggregator(cfg)
        self.splitter = get_splitter(cfg)

        # Seed
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

    def train(self) -> List[RoundResult]:
        """
        Main training loop. Returns list of RoundResult (one per round).
        """
        total_rounds = self.config.global_round

        if self.hook.is_multi_branch:
            return self._train_multi_branch(total_rounds)
        else:
            return self._train_single_branch(total_rounds)

    def _train_single_branch(self, total_rounds: int) -> List[RoundResult]:
        """Training loop for SFL, USFL, SCAFFOLD, Mix2SFL, GAS."""
        results = []
        for round_num in range(1, total_rounds + 1):
            t0 = time.time()

            # 1. Pre-round: client selection, model splitting, data prep
            ctx = self.hook.pre_round(self, round_num)

            # 2. Training: the core SFL loop (shared by ALL methods)
            client_results = self._run_sfl_round(ctx)

            # 3. Post-round: aggregation, evaluation, state updates
            round_result = self.hook.post_round(self, round_num, ctx, client_results)
            round_result.metrics['round_time'] = time.time() - t0

            results.append(round_result)
            self._log_round(round_result)

        return results

    def _train_multi_branch(self, total_rounds: int) -> List[RoundResult]:
        """Training loop for MultiSFL."""
        results = []
        for round_num in range(1, total_rounds + 1):
            t0 = time.time()

            # 1. Pre-round: select clients, setup branch mapping
            self.hook.pre_round(self, round_num)

            # 2. Per-branch training
            B = self.config.multisfl_branches
            for branch in range(B):
                ctx = self.hook.pre_branch(self, round_num, branch)
                client_results = self._run_sfl_round(ctx)
                self.hook.post_branch(self, round_num, branch, ctx, client_results)

            # 3. Post-round: master aggregation, soft pull, evaluate
            round_result = self.hook.post_round_multi(self, round_num)
            round_result.metrics['round_time'] = time.time() - t0

            results.append(round_result)
            self._log_round(round_result)

        return results

    def _run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
        """
        Execute one round of SFL training. THIS IS SHARED BY ALL 7 METHODS.

        The only method-specific behavior is hook.process_gradients(), called
        between server backward and client backward.
        """
        client_states = ctx.client_states
        server_model = ctx.server_model
        server_optimizer = ctx.server_optimizer
        criterion = ctx.criterion
        device = ctx.device

        # Save initial client model state (for restoring between clients)
        # All clients start from the same global model state
        client_model_snapshot = None
        sample_client = next(iter(client_states.values()))
        client_model_snapshot = snapshot_model(sample_client.client_model)

        # Training iterations
        for iteration in range(ctx.iterations):
            # Determine per-client batch sizes for this iteration
            # (USFL dynamic scheduler may vary this)

            # --- Phase 1: All clients run forward pass ---
            activations = []
            labels_list = []
            client_order = sorted(client_states.keys())  # Deterministic order

            for cid in client_order:
                state = client_states[cid]

                # Restore client model to global state at start of each client
                # (each client trains independently from the same starting point)
                if iteration == 0:
                    restore_model(state.client_model, client_model_snapshot)

                batch = get_next_batch(state)
                act, lbl = client_forward(state, batch, device)
                activations.append(act)
                labels_list.append(lbl)

            # --- Phase 2: Server forward + backward ---
            # Concatenate activations from all clients
            concat_act = torch.cat(activations, dim=0)
            concat_labels = torch.cat(labels_list, dim=0)

            # Handle BatchNorm edge case
            if concat_act.size(0) == 1:
                server_model.eval()
            else:
                server_model.train()

            concat_act.requires_grad_(True)
            concat_act.retain_grad()

            server_optimizer.zero_grad()
            logits = server_model(concat_act)
            loss = criterion(logits, concat_labels)
            loss.backward()
            server_optimizer.step()

            # Get activation gradients
            activation_grads = concat_act.grad.clone().detach()

            # --- Phase 3: Method-specific gradient processing ---
            activation_grads = self.hook.process_gradients(
                activation_grads, activations, concat_labels, ctx
            )

            # --- Phase 4: All clients run backward pass ---
            offset = 0
            for cid in client_order:
                state = client_states[cid]
                act = activations[client_order.index(cid)]
                batch_size = act.size(0)
                grad_slice = activation_grads[offset:offset + batch_size]
                offset += batch_size

                client_backward(state, act, grad_slice, self.config)

        # --- Collect results ---
        client_results = []
        for cid in client_order:
            state = client_states[cid]
            result = ClientResult(
                client_id=cid,
                model_state_dict=snapshot_model(state.client_model),
                dataset_size=state.dataset_size,
                label_distribution=state.label_distribution,
            )
            client_results.append(result)

        return client_results

    def evaluate(self, model: nn.Module, testloader: DataLoader) -> tuple:
        """Evaluate model on test set. Returns (accuracy, loss)."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / total
        return accuracy, avg_loss

    def _log_round(self, result: RoundResult):
        """Print round summary to stdout."""
        print(f"[Round {result.round_number:3d}] "
              f"Acc: {result.accuracy:.4f}  "
              f"Loss: {result.loss:.4f}  "
              f"Time: {result.metrics.get('round_time', 0):.1f}s")
```

**CRITICAL IMPLEMENTATION NOTES for `_run_sfl_round()`:**

1. **Client model lifecycle**: Each client starts from the SAME global model state (snapshot at round start). After training, each client has diverged — their state_dicts are collected for aggregation.

2. **Activation gradient flow**: `concat_act.requires_grad_(True)` and `.retain_grad()` are essential. Without these, `concat_act.grad` will be None after `loss.backward()`.

3. **The `process_gradients` hook** is the ONLY point where methods differ within a round. Everything else is identical.

4. **Client ordering**: Always process clients in sorted order by `client_id` for reproducibility.

5. **BatchNorm**: When total batch size is 1, switch server model to eval mode to avoid BatchNorm errors.

6. **Server model aggregation mode** (`server_model_aggregation=True`): Some experiments maintain separate server models per client. This requires `deepcopy(server_model)` per client — a rare case that must be supported. Add a flag in RoundContext.

### 7.5 `entry.py` (~150 lines)

**Purpose**: CLI entry point, seed setup, result saving.

```python
"""
CLI entry point for the unified simulation framework.

Usage:
    python -m unified_simulation_framework.entry \
        --method usfl \
        --sfl-args "-d cifar10 -m resnet18 -M usfl -gr 100 ..." \
        [--gas-generate-features] \
        [--multisfl-branches 3]

    Or from Python:
        from unified_simulation_framework.entry import run
        results = run(method="sfl", sfl_args=["-d", "cifar10", ...])
"""
import argparse
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Optional

from .config import UnifiedConfig, parse_unified_config
from .trainer import SimTrainer
from .client_ops import RoundResult


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(
    method: str,
    sfl_args: List[str],
    extra_args: dict = None,
    output_dir: str = None,
) -> List[RoundResult]:
    """
    Run a complete experiment.

    Args:
        method: 'sfl', 'usfl', 'scaffold_sfl', 'mix2sfl', 'gas', 'multisfl'
        sfl_args: CLI args in SFL format
        extra_args: Method-specific overrides
        output_dir: Where to save results (optional)

    Returns: List of RoundResult
    """
    config = parse_unified_config(method, sfl_args, extra_args)

    if config.sfl_config.seed is not None:
        set_seed(config.sfl_config.seed)

    trainer = SimTrainer(config)
    results = trainer.train()

    if output_dir:
        save_results(results, output_dir, method)

    return results


def save_results(results: List[RoundResult], output_dir: str, method: str):
    """Save results in the same JSON format as the existing frameworks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Main results file
    result_data = {
        "method": method,
        "rounds": [
            {
                "round": r.round_number,
                "accuracy": r.accuracy,
                "loss": r.loss,
                "metrics": r.metrics,
            }
            for r in results
        ],
        "final_accuracy": results[-1].accuracy if results else 0.0,
    }

    with open(output_path / f"{method}_results.json", "w") as f:
        json.dump(result_data, f, indent=2)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Unified SFL Simulation")
    parser.add_argument("--method", required=True,
                        choices=["sfl", "usfl", "scaffold_sfl", "mix2sfl", "gas", "multisfl"])
    parser.add_argument("--sfl-args", required=True, type=str,
                        help="SFL-format CLI args as a single quoted string")
    parser.add_argument("--output-dir", type=str, default=None)
    # GAS-specific
    parser.add_argument("--gas-generate-features", action="store_true")
    parser.add_argument("--gas-logit-adjustment", action="store_true", default=True)
    # MultiSFL-specific
    parser.add_argument("--multisfl-branches", type=int, default=3)
    parser.add_argument("--multisfl-alpha-master-pull", type=float, default=0.1)
    parser.add_argument("--multisfl-p-update", type=str, default="abs_ratio")

    args = parser.parse_args()

    sfl_args = args.sfl_args.split()
    extra_args = {
        k: v for k, v in vars(args).items()
        if k.startswith("gas_") or k.startswith("multisfl_")
    }

    run(
        method=args.method,
        sfl_args=sfl_args,
        extra_args=extra_args,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
```

### 7.6 `methods/__init__.py` (~30 lines)

```python
"""Method hook registry."""

from .base import BaseMethodHook
from .sfl import SFLHook
from .usfl import USFLHook
from .scaffold import ScaffoldHook
from .mix2sfl import Mix2SFLHook
from .gas import GASHook
from .multisfl import MultiSFLHook

HOOKS = {
    "sfl": SFLHook,
    "usfl": USFLHook,
    "scaffold_sfl": ScaffoldHook,
    "mix2sfl": Mix2SFLHook,
    "gas": GASHook,
    "multisfl": MultiSFLHook,
}

def get_method_hook(method: str, config, resources: dict) -> BaseMethodHook:
    """Factory: create the appropriate method hook."""
    if method not in HOOKS:
        raise ValueError(f"Unknown method: {method}. Available: {list(HOOKS.keys())}")
    return HOOKS[method](config, resources)
```

---

## 8. Method Hook Specifications

### 8.1 SFL Hook (`methods/sfl.py`, ~80 lines)

The simplest case. Baseline for all other methods.

```python
class SFLHook(BaseMethodHook):
    """Basic Split Federated Learning."""

    def pre_round(self, trainer, round_number):
        """
        1. Select clients: selector.select(num_clients_per_round, all_client_ids, client_info)
        2. Split model: splitter.split(model) → (client_model_dict, server_model_dict)
        3. Create server model + optimizer
        4. For each selected client:
           a. Create ClientState with create_client_state()
        5. Return RoundContext

        MODEL COPY STRATEGY:
        - Get the torch model: torch_model = model.get_torch_model()
        - Split once: client_module, server_module = splitter.split(torch_model)
        - Snapshot client state: base_state = snapshot_model(client_module)
        - For each client: same client_module reference, restored from base_state
        - Server model: single instance, shared across all clients
        """

    def post_round(self, trainer, round_number, ctx, client_results):
        """
        1. Build aggregation params from client_results
        2. Aggregate client models: aggregator.aggregate(models, params)
        3. Update global model with aggregated state
        4. Sync full model from split models (FlexibleResNet: sync_full_model_from_split())
        5. Evaluate: trainer.evaluate(full_model, testloader)
        6. Optional: G measurement, drift measurement
        7. Return RoundResult
        """
```

**Source for extraction**: `sfl_stage_organizer.py` lines 172-307 (pre_round), 500-600 (post_round)

### 8.2 USFL Hook (`methods/usfl.py`, ~300 lines)

The most complex hook. Implements 6 optimizations.

```python
class USFLHook(BaseMethodHook):
    """USFL: Unified Split Federated Learning with 6 Non-IID optimizations."""

    def __init__(self, config, resources):
        super().__init__(config, resources)
        # Persistent state across rounds
        self.cumulative_usage = {}   # {client_id: {label: {bin_key: count}}}
        self.selection_history = []  # Recent selection lists

    def pre_round(self, trainer, round_number):
        """
        1. Collect client_informations: {client_id: label_distribution}
           (In simulation, we have this from distributer — no need to "wait")

        2. USFL Selector: select clients with missing label + freshness scoring
           - Pass cumulative_usage to selector for fresh scoring
           - selector.select(n, client_ids, client_informations)

        3. Calculate augmented_dataset_sizes (DATA BALANCING):
           Strategy determined by config.balancing_strategy:
           a. 'trimming': All labels trimmed to min label count across selected clients
           b. 'replication': All labels replicated to max label count
           c. 'target': Hybrid using mean/median/custom target

           For each selected client, compute:
             augmented_sizes[client_id] = {label: target_count, ...}

        4. Dynamic Batch Scheduler (if enabled):
           - C_list = [sum(augmented_sizes[cid].values()) for cid in selected]
           - k, schedule = create_schedule(batch_size, C_list)
           - Store schedule in RoundContext

        5. Split model, create ClientStates with augmented_sizes
           - Each client's MaskableDataset gets update_amount_per_label(augmented_sizes[cid])

        6. Return RoundContext with batch_schedule
        """

    def process_gradients(self, activation_grads, activations, labels, ctx):
        """
        Apply gradient shuffle. Strategy from config.gradient_shuffle_strategy:

        a. 'random': Random permutation of gradient indices
           - perm = torch.randperm(activation_grads.size(0))
           - return activation_grads[perm]

        b. 'inplace': Class-balanced shuffle
           - Group gradients by label
           - Redistribute evenly across clients

        c. 'average': Mix with global mean gradient
           - mean_grad = activation_grads.mean(dim=0)
           - weight = config.gradient_average_weight
           - return weight * activation_grads + (1 - weight) * mean_grad

        d. 'average_adaptive_alpha': Cosine similarity-based adaptive mixing
           - For each client's grad slice:
             cos_sim = F.cosine_similarity(grad, mean_grad)
             alpha = 1 / (1 + exp(-config.adaptive_alpha_beta * cos_sim))
             mixed = alpha * grad + (1 - alpha) * mean_grad

        Source: usfl_stage_organizer.py _shuffle_gradients() (lines ~1200-1330)
        """

    def post_round(self, trainer, round_number, ctx, client_results):
        """
        1. Aggregate with USFL Aggregator (label-capped weighted)
        2. Update global model
        3. Evaluate
        4. Update cumulative_usage tracking:
           For each selected client, for each label used:
             usage_count = previous_usage + samples_used_this_round
             bin_key = _get_exponential_bin(usage_count)
             cumulative_usage[client_id][label][bin_key] += 1
        5. Return RoundResult

        Source: usfl_stage_organizer.py _post_round() (lines ~1550-1642)
        """

    @staticmethod
    def _get_exponential_bin(usage_count: int) -> int:
        """Map usage count to exponential bin key.
        0→0, 1→1, 2-3→2, 4-7→4, 8-15→8, ..."""
        if usage_count <= 1:
            return usage_count
        return 2 ** (int(usage_count).bit_length() - 1)
```

**Sources for extraction**:
- Data balancing: `usfl_stage_organizer.py` lines 660-830
- Gradient shuffle: `usfl_stage_organizer.py` lines 396-518, 1200-1330
- Cumulative usage: `usfl_stage_organizer.py` lines 528-544, 1565-1600
- Dynamic batch: `scheduler/batch_scheduler.py` (reuse directly)
- USFL selector: `seletor/usfl_selector.py` (reuse directly)
- USFL aggregator: `aggregator/usfl_aggregator.py` (reuse directly)

### 8.3 SCAFFOLD Hook (`methods/scaffold.py`, ~150 lines)

```python
class ScaffoldHook(BaseMethodHook):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning."""

    def __init__(self, config, resources):
        super().__init__(config, resources)
        self.c_global = None          # Global control variate
        self.c_clients = {}           # {client_id: control_variate_state_dict}

    def pre_round(self, trainer, round_number):
        """
        Same as SFL pre_round, plus:
        - Initialize c_global on first round (zeros like model params)
        - Initialize c_i for new clients (zeros)
        - Pass c_global and c_i to ClientState.extra for use in backward
        """

    def post_round(self, trainer, round_number, ctx, client_results):
        """
        1. Standard FedAvg aggregation
        2. Update control variates:
           For each client i:
             c_i_new = c_i - c_global + (1 / (K * lr)) * (model_global - model_i)
           c_global_new = c_global + (1 / N) * Σ(c_i_new - c_i)
        3. Evaluate

        Source: scaffold_stage_organizer.py _post_round()
        """
```

**Note on SCAFFOLD client backward**: The SCAFFOLD correction term `c_global - c_i` must be applied during client optimizer step. This can be done by modifying the gradient before optimizer.step() in `client_backward()`, or by storing the correction in ClientState and applying it in a custom step.

### 8.4 Mix2SFL Hook (`methods/mix2sfl.py`, ~180 lines)

```python
class Mix2SFLHook(BaseMethodHook):
    """Mix2SFL: SmashMix + GradMix for SFL."""

    def pre_round(self, trainer, round_number):
        """Same as SFL pre_round."""

    def process_gradients(self, activation_grads, activations, labels, ctx):
        """
        Apply GradMix: mix gradients between client pairs.

        1. Select C' subset of clients for GradMix
        2. For each pair (c1, c2) in C':
           mixed_grad = phi * grad_c1 + (1 - phi) * grad_c2
        3. Replace original grads with mixed grads

        Source: mix2sfl_stage_organizer.py GradMix logic
        """

    # SmashMix modifies activations before server forward, which requires
    # a hook at a different point. Options:
    # a. Add a process_activations() hook called before server forward
    # b. Override the entire round in a custom method
    # Recommended: add process_activations() to BaseMethodHook (default: pass-through)

    def process_activations(self, activations, labels, ctx):
        """
        Apply SmashMix: mix activations between client pairs.

        1. Select client pairs
        2. For each pair: mix_act = lambda * act_1 + (1 - lambda) * act_2
           where lambda ~ Beta(alpha, alpha)
        3. Create mixed labels accordingly

        Source: mix2sfl_stage_organizer.py SmashMix logic
        """
```

**Note**: SmashMix requires an additional hook point (between client forward and server forward). Add `process_activations()` to `BaseMethodHook` with a default pass-through, and call it in `_run_sfl_round()` after collecting all activations but before server forward.

### 8.5 GAS Hook (`methods/gas.py`, ~250 lines)

```python
class GASHook(BaseMethodHook):
    """GAS: Gradient Adjustment Scheme."""

    def __init__(self, config, resources):
        super().__init__(config, resources)
        self.feature_stats = {}       # IncrementalStats per label
        self.logit_adjustments = {}   # Per-client logit correction
        self.activation_buffer = []   # Buffer for server model update

    def pre_round(self, trainer, round_number):
        """
        1. Client selection:
           - If WRTT enabled: time-based selection (find_client_with_min_time)
           - Else: uniform or fixed schedule
        2. Compute logit local adjustment for each client:
           logit_adj[cid] = compute_local_adjustment(client_dataloader, device)
        3. Split model, create ClientStates
        4. Return RoundContext with logit_adjustments in ctx.extra

        Source: GAS_main.py lines 930-1100
        """

    def post_round(self, trainer, round_number, ctx, client_results):
        """
        1. FedAvg aggregation
        2. Update feature statistics (IncrementalStats):
           For each client's activation buffer:
             For each label:
               stats.update(mean, variance, weight, label)
        3. Optional: Feature generation
           If config.gas_generate_features and round % sample_frequency == 0:
             augmented = sample_or_generate_features(buffer, stats)
        4. V-value computation (diagnostic, if config.gas_v_test):
           v = calculate_v_value(sampled_grads, oracle_grads)
        5. Evaluate

        Source: GAS_main.py lines 1400-1580
        """

    # GAS logit adjustment is applied at loss computation, not at gradient level.
    # This requires modifying the server forward in _run_sfl_round.
    # Options:
    # a. Add a compute_loss() hook that receives logits and labels → loss
    # b. Apply adjustment by modifying logits before criterion
    # Recommended: add compute_loss() hook to BaseMethodHook

    def compute_loss(self, logits, labels, ctx, client_id):
        """
        Apply logit local adjustment before loss computation.

        adjusted_logits = logits + logit_adjustments[client_id]
        loss = criterion(adjusted_logits, labels)

        Source: GAS_main.py line 1183
        """
```

**Note**: GAS logit adjustment requires per-client loss modification. The simplest approach: add a `compute_loss()` hook to `BaseMethodHook` (default: `criterion(logits, labels)`), and override in GASHook to add the adjustment.

### 8.6 MultiSFL Hook (`methods/multisfl.py`, ~350 lines)

```python
class MultiSFLHook(BaseMethodHook):
    """MultiSFL: Multi-branch SFL with knowledge replay."""

    def __init__(self, config, resources):
        super().__init__(config, resources)
        B = config.multisfl_branches

        # Per-branch state
        self.branch_client_models = [None] * B    # nn.Module per branch
        self.branch_server_models = [None] * B    # nn.Module per branch
        self.branch_client_optimizers = [None] * B
        self.branch_server_optimizers = [None] * B

        # Master state (average across branches)
        self.master_client_state = None
        self.master_server_state = None

        # Replay components (import from multisfl_implementation)
        self.score_tracker = ScoreVectorTracker(config.multisfl_gamma, B)
        self.replay_planner = KnowledgeRequestPlanner(resources['num_classes'])
        self.scheduler = SamplingProportionScheduler(
            p0=config.multisfl_p0,
            p_min=config.multisfl_p_min,
            p_max=config.multisfl_p_max,
            mode=config.multisfl_p_update,
        )
        self.p_r = config.multisfl_p0  # Current replay proportion

        # Client objects (for replay sampling)
        self.clients = {}  # {client_id: Client object}

        # Initialize branch models (deep copy from global)
        self._init_branches()

    @property
    def is_multi_branch(self):
        return True

    def pre_round(self, trainer, round_number):
        """
        1. Select n_main clients (or load from fixed schedule)
        2. Create mapping: branch_index → client_id
        3. Identify inactive_clients (not selected)
        4. Store mapping for pre_branch/post_branch

        Source: trainer.py lines 269-300
        """

    def pre_branch(self, trainer, round_number, branch):
        """
        1. Get client_id for this branch from mapping
        2. Get branch's client_model and server_model
        3. Compute knowledge replay:
           a. sv = score_tracker.score_vector(branch)
           b. req = replay_planner.plan(sv, p_r, base_count, replay_min_total)
           c. For each inactive client, call sample_batch_by_quota(q_remaining)
           d. Forward replay data through branch's client_model (no grad)
           e. Store replay features/labels in ctx.extra
        4. Create ClientState for the main client
        5. Return RoundContext (server training will use main + replay data)

        Source: trainer.py lines 465-660
        """

    def post_branch(self, trainer, round_number, branch, ctx, client_results):
        """
        1. Update branch client model from training results
        2. Track score vector: score_tracker.append_label_dist(branch, dist)
        3. Collect FGN (Falsified Gradient Norm) for scheduler

        Source: trainer.py lines 660-700
        """

    def post_round_multi(self, trainer, round_number):
        """
        1. Master aggregation:
           master_client = average(branch_client_models)
           master_server = average(branch_server_models)
        2. Soft pull:
           For each branch b:
             branch_b = (branch_b + alpha * master) / (1 + alpha)
        3. Update p_r: scheduler.update(fgn_mean)
        4. Evaluate master model on test set
        5. Return RoundResult

        Source: trainer.py lines 700-774, servers.py compute_master/soft_pull_to_master
        """

    def _soft_pull(self, branch_state, master_state, alpha):
        """
        Blend branch toward master:
        new[k] = (branch[k] + alpha * master[k]) / (1 + alpha)

        Source: servers.py line 72
        """
```

**Note on MultiSFL's `_run_sfl_round()` compatibility**: MultiSFL's server training includes replay data (concatenated with main client's activations). This must be handled in the hook. The approach: in `pre_branch()`, prepare replay activations and store in `ctx.extra['replay_features']` and `ctx.extra['replay_labels']`. Then, modify `_run_sfl_round()` to check for replay data and concatenate it with main activations before server forward. Alternatively, add a `prepare_server_input()` hook.

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure + SFL (~3 files, ~700 lines)

**Goal**: Get the simplest case (basic SFL) working end-to-end.

**Files to create**:
1. `__init__.py` — Package marker
2. `config.py` — UnifiedConfig wrapping SFL Config
3. `client_ops.py` — Data structures + client forward/backward
4. `methods/base.py` — BaseMethodHook ABC
5. `methods/__init__.py` — Registry
6. `methods/sfl.py` — SFL hook (simplest method)
7. `trainer.py` — SimTrainer with `_run_sfl_round()`
8. `entry.py` — CLI entry point

**Validation**: Run SFL experiment:
```bash
python -m unified_simulation_framework.entry \
    --method sfl \
    --sfl-args "-d cifar10 -m resnet18_flex -M sfl -le 5 -gr 10 -bs 50 -nc 100 -ncpr 10 -ss layer_name -sl layer2 -sd 42 -distr shard_dirichlet -diri-alpha 0.3 -lpc 2 -o sgd -lr 0.001 -s uniform -aggr fedavg -c ce -de cuda -sma false -nnf"
```
Compare accuracy curve with existing framework output (same seed=42).

### Phase 2: USFL (~1 file, ~300 lines)

**Goal**: Support USFL with all 6 optimizations.

**Files to create**:
1. `methods/usfl.py` — USFLHook

**Key extraction targets** (from `usfl_stage_organizer.py`):
- `_calculate_augmented_sizes()` → `USFLHook.pre_round()` data balancing section
- `_shuffle_gradients()` → `USFLHook.process_gradients()`
- `_update_cumulative_usage()` → `USFLHook.post_round()` tracking section

**Validation**: Run USFL experiment with gradient_shuffle=True, dynamic_batch=True, cumulative_usage=True, fresh_scoring=True. Compare with existing framework.

### Phase 3: GAS (~1 file, ~250 lines)

**Goal**: Support GAS method.

**Files to create**:
1. `methods/gas.py` — GASHook

**Key**: Ensure imports from `GAS_implementation/utils/` work. May need to add `GAS_implementation/` to `sys.path`.

**Additional hooks needed**: Add `compute_loss()` hook to `BaseMethodHook`, update `_run_sfl_round()` to call it.

**Validation**: Run GAS experiment. Compare with `GAS_main.py` output.

### Phase 4: SCAFFOLD + Mix2SFL (~2 files, ~330 lines)

**Goal**: Support SCAFFOLD control variates and Mix2SFL SmashMix+GradMix.

**Files to create**:
1. `methods/scaffold.py` — ScaffoldHook
2. `methods/mix2sfl.py` — Mix2SFLHook

**Additional hooks needed**: Add `process_activations()` hook to `BaseMethodHook`, update `_run_sfl_round()` to call it between client forward and server forward.

### Phase 5: MultiSFL (~1 file, ~350 lines)

**Goal**: Support multi-branch training with knowledge replay.

**Files to create**:
1. `methods/multisfl.py` — MultiSFLHook

**Key**: Ensure imports from `multisfl_implementation/multisfl/` work.

**`_run_sfl_round()` modification**: Must support replay data in server forward. Add `prepare_server_input()` hook or check `ctx.extra['replay_features']`.

### Phase 6: Experiment Infrastructure Integration

**Goal**: Connect to `experiment_core/` for deploy.sh integration.

**Files to create**:
1. `experiment_core/unified_runner.py` — Bridge between batch_runner and unified framework

**Update** (minimal changes to existing files):
1. `experiment_core/adapters/sfl_adapter.py` — Add `use_unified` flag
2. `experiment_configs/` — Add `"use_unified": true` to method JSONs

---

## 10. Validation Strategy

### Numerical Reproducibility

For each method, run with **identical config + seed** on both old and new frameworks. Compare:

1. **Round-by-round accuracy** — Must match within ±0.001 (floating-point tolerance)
2. **Round-by-round loss** — Must match within ±0.01
3. **Final accuracy** — Must match exactly (same seed, same operations)

**If curves don't match**, debug by comparing:
- Client selection order per round
- Per-client batch composition (data indices)
- Activation tensors at split point (should be identical)
- Gradient tensors after server backward (should be identical)
- Aggregated model weights (should be identical)

### Test Matrix

| Method | Dataset | Distribution | Rounds | Expected |
|--------|---------|-------------|--------|----------|
| SFL | CIFAR-10 | shard_dirichlet(0.3) | 10 | Match old framework |
| SFL | CIFAR-10 | uniform (IID) | 10 | Match old framework |
| USFL | CIFAR-10 | shard_dirichlet(0.3) | 10 | Match old framework |
| USFL (all features) | CIFAR-10 | shard_dirichlet(0.3) | 10 | Match old framework |
| SCAFFOLD | CIFAR-10 | shard_dirichlet(0.3) | 10 | Match old framework |
| Mix2SFL | CIFAR-10 | shard_dirichlet(0.3) | 10 | Match old framework |
| GAS | CIFAR-10 | shard_dirichlet(0.3) | 10 | Match GAS_main.py |
| MultiSFL | CIFAR-10 | shard_dirichlet(0.3) | 10 | Match run_multisfl.py |

### Performance Benchmark

After validation, measure speedup:
```
Old framework (10 rounds, 100 clients, CIFAR-10, ResNet18): ____ seconds
New framework (same config): ____ seconds
Speedup: ____x
```

Expected: **2-5x speedup** from eliminating communication overhead.

---

## 11. Integration with Experiment Infrastructure

### How deploy.sh works (for context)

```
deploy.sh run usfl@xsailor4:0
  → generate_spec.py → batch_spec.json
  → scp to server
  → tmux: remote_run.sh → batch_runner.py → runner.py → sfl_runner.py → simulation.py
```

### Integration point

Replace the last step:
```
runner.py → unified_runner.py → unified_simulation_framework.entry.run()
```

**`experiment_core/unified_runner.py`**:
```python
"""Bridge between experiment_core batch_runner and unified simulation framework."""

def run_from_spec(spec_path: str, repo_root: str):
    """
    Read a spec JSON, convert to unified config, run experiment.

    Reuses sfl_runner.py's spec_to_workload() and workload_to_server_args()
    for compatibility with existing spec format.
    """
    spec = load_spec(spec_path)
    method = spec['method']
    sfl_args = spec_to_sfl_args(spec)  # Convert spec to CLI args
    extra_args = extract_method_extras(spec)

    from unified_simulation_framework.entry import run
    run(method=method, sfl_args=sfl_args, extra_args=extra_args,
        output_dir=spec.get('output_dir'))
```

### Adapter changes

In `experiment_core/adapters/sfl_adapter.py`, add:
```python
def build_command(self, spec, repo_root):
    if spec.get('use_unified'):
        return f"python -m experiment_core.unified_runner --spec {spec_path} --repo-root {repo_root}"
    else:
        return f"python {repo_root}/experiment_core/sfl_runner.py ..."  # existing
```

---

## 12. ERRATA: Critical Design Corrections (Post-Review)

> **READ THIS SECTION FIRST.** These corrections were discovered by reviewing the actual source code of all three frameworks after the initial design was written. They **override** the corresponding sections above. Failing to apply these corrections will produce a framework that is **numerically incorrect** compared to the original implementations.

---

### ERRATA 1: SFL Uses Per-Client Server Forward/Backward (NOT Concatenated)

**Affected sections:** Section 7 (`_run_sfl_round`), Section 3 (Architecture Overview)

**What the design says (WRONG):**
The `_run_sfl_round()` specification (Section 7) shows ALL clients' activations being concatenated into one batch (`torch.cat(activations, dim=0)`) before a single server forward + backward + `optimizer.step()`. This is presented as the universal SFL core shared by all methods.

**What the code actually does:**
There are **two distinct server-side training patterns** in the original framework:

#### Pattern A: Per-client sequential (SFL, SCAFFOLD)
From `sfl_stage_organizer.py:329-375`:
```python
async def __server_side_training():
    server_model = self.split_models[1].to(self.config.device)
    server_optimizer = self.in_round._get_optimizer(server_model, self.config)
    server_criterion = self.in_round._get_criterion(self.config)

    while True:
        # Receive ONE client's activation at a time (arrival order)
        activation = await self.in_round.wait_for_activations()
        client_id = activation["client_id"]

        # Server forward on THIS client's activation only
        output = await self.in_round.forward(server_model, activation)

        # Server backward + optimizer.step() for THIS client
        grad, loss, server_grad = await self.in_round.backward_from_label(
            server_model, output, activation,
            optimizer=server_optimizer, criterion=server_criterion,
        )

        # Send gradient back to THIS client immediately
        # (Next iteration: process next client)
```

Key characteristics:
- `optimizer.step()` is called **once per client per iteration** (inside `backward_from_label`)
- Server model is updated by each client sequentially — client 2 sees a server model already updated by client 1
- Gradients are returned immediately per-client (no accumulation)

#### Pattern B: Concatenated batch (USFL, Mix2SFL)
From `usfl_stage_organizer.py:1057-1131`:
```python
async def __server_side_training():
    while True:
        # Wait for ALL selected clients' activations
        activations = await self.in_round.wait_for_concatenated_activations(
            self.selected_clients
        )

        # Concatenate into single batch
        concatenated_activations = {}
        for activation in activations:
            concatenated_activations = self._concatenate_activations(
                concatenated_activations, activation
            )

        # Single server forward + backward on the concatenated batch
        logits = await self.in_round.forward(server_model, concatenated_activations)
        grad, loss, server_grad = await self.in_round.backward_from_label(
            server_model, logits, concatenated_activations,
            optimizer=server_optimizer, criterion=server_criterion,
        )

        # Gradient shuffle (USFL-specific), then split and return per-client
```

Key characteristics:
- `optimizer.step()` is called **once per iteration** on the concatenated batch
- All clients contribute to the same gradient before server update
- USFL applies gradient shuffle before splitting gradients back per-client

#### GAS Pattern: Sequential per-client (like Pattern A)
From `GAS_main.py`: The training loop processes **one client at a time** within each round:
```python
for user_index in participating_users:
    # Forward client model → activation
    # Server forward → logits (with logit adjustment)
    # Server backward → gradient
    # Server optimizer.step() per client
    # Client backward
```

#### MultiSFL Pattern: Sequential per-client per-branch
From `multisfl/trainer.py`: Each branch processes its assigned client sequentially, with a double-forward pattern (see ERRATA 5).

**CORRECTED DESIGN:**

The `_run_sfl_round()` must support **two modes**, selected by the hook:

```python
class BaseMethodHook:
    @property
    def server_training_mode(self) -> str:
        """Return 'per_client' or 'concatenated'."""
        return 'per_client'  # Default for SFL, SCAFFOLD, GAS, MultiSFL

class USFLHook(BaseMethodHook):
    @property
    def server_training_mode(self) -> str:
        return 'concatenated'  # USFL, Mix2SFL use concatenated mode

def _run_sfl_round(self, ctx: RoundContext) -> List[ClientResult]:
    if self.hook.server_training_mode == 'concatenated':
        return self._run_sfl_round_concatenated(ctx)
    else:
        return self._run_sfl_round_per_client(ctx)
```

**`_run_sfl_round_per_client` (Pattern A):**
```python
def _run_sfl_round_per_client(self, ctx):
    server_model = ctx.server_model
    server_optimizer = ctx.server_optimizer
    criterion = ctx.criterion

    for epoch in range(ctx.local_epochs):  # See ERRATA 4
        for cid in ctx.client_order:
            state = ctx.client_states[cid]
            for batch in state.dataloader:
                # Client forward
                act, labels = client_forward(state, batch, ctx.device)
                act.requires_grad_(True)
                act.retain_grad()

                # Server forward + backward (per-client)
                server_optimizer.zero_grad()
                logits = server_model(act)
                loss = self.hook.compute_loss(logits, labels, ctx)  # Hook for GAS logit adj
                loss.backward()
                server_optimizer.step()

                # Get activation gradient
                grad = act.grad.clone().detach()
                grad = self.hook.process_gradients(grad, [act], labels, ctx)

                # Client backward
                client_backward(state, act, grad, self.config)

    # Collect client model states
    return [ClientResult(cid, snapshot_model(state.client_model), ...) for ...]
```

**`_run_sfl_round_concatenated` (Pattern B):**
```python
def _run_sfl_round_concatenated(self, ctx):
    server_model = ctx.server_model
    server_optimizer = ctx.server_optimizer
    criterion = ctx.criterion

    for iteration in range(ctx.iterations):
        # ALL clients forward
        activations, labels_list = [], []
        for cid in ctx.client_order:
            act, lbl = client_forward(ctx.client_states[cid], ..., ctx.device)
            activations.append(act)
            labels_list.append(lbl)

        # Concatenate
        concat_act = torch.cat(activations, dim=0)
        concat_labels = torch.cat(labels_list, dim=0)
        concat_act.requires_grad_(True)
        concat_act.retain_grad()

        # Server forward + backward (single step for all clients)
        server_optimizer.zero_grad()
        logits = server_model(concat_act)
        loss = criterion(logits, concat_labels)
        loss.backward()
        server_optimizer.step()

        # Gradient processing (USFL shuffle)
        activation_grads = concat_act.grad.clone().detach()
        activation_grads = self.hook.process_gradients(
            activation_grads, activations, concat_labels, ctx
        )

        # Split gradients back to per-client, client backward
        offset = 0
        for i, cid in enumerate(ctx.client_order):
            batch_size = activations[i].size(0)
            grad_slice = activation_grads[offset:offset + batch_size]
            offset += batch_size
            client_backward(ctx.client_states[cid], activations[i], grad_slice, self.config)

    return [ClientResult(...) for ...]
```

**Impact**: This is the most critical correction. Without it, SFL and SCAFFOLD would use concatenated training (wrong gradient computation, wrong server model update frequency), producing different convergence behavior.

---

### ERRATA 2: Server `optimizer.step()` Timing Differs by Pattern

**Affected sections:** Section 7 (`_run_sfl_round` pseudocode)

**What the design says (WRONG):**
A single `server_optimizer.step()` per iteration for all methods.

**What actually happens:**
- **Pattern A (per-client)**: `optimizer.step()` is called inside `backward_from_label()` for **each client's activation** separately. This means the server model is updated K times per iteration (K = clients per round). Each subsequent client sees a slightly different server model.
- **Pattern B (concatenated)**: `optimizer.step()` is called **once** on the combined gradient from all clients. All clients contributed to the same gradient update.

**Corrected in ERRATA 1 above.** The two `_run_sfl_round_*` methods already encode this difference correctly.

---

### ERRATA 3: `FlexibleResNet.get_split_models()` Returns References, Not Copies

**Affected sections:** Section 7 (client model snapshot/restore logic)

**What the design says (WRONG):**
The design uses `snapshot_model()` and `restore_model()` to manage client model state between clients, assuming each client gets an independent copy. It also implies that calling `get_split_models()` creates fresh models.

**What the code actually does:**
From `flexible_resnet.py:432-434`:
```python
def get_split_models(self) -> Tuple[nn.Module, nn.Module]:
    """Return (client_model, server_model) tuple."""
    return self.client_model, self.server_model
```

This returns **references** to the same `nn.Module` objects. The SFL framework handles this correctly: at `sfl_stage_organizer.py:296`, the same `self.split_models[0]` reference is sent to all clients in a list:
```python
[self.split_models[0] for _ in range(len(self.selected_clients))]
```

This works because in the original framework, each client runs its own forward pass on its own data using this model, and the InMemoryConnection manages state via the async event loop.

**Impact on unified framework:**
In our synchronous design, we must be explicit about model state management:

1. **At round start**: `snapshot = {k: v.clone() for k, v in client_model.state_dict().items()}`
2. **Before each client trains**: `client_model.load_state_dict(snapshot)` (for Pattern A, where each client starts from the same global state)
3. **After each client trains**: `results[cid] = {k: v.clone() for k, v in client_model.state_dict().items()}`

**Key rule**: Always use `state_dict()` clone, never `copy.deepcopy(model)` — it's 10x slower and unnecessary when we only need the parameters.

**HOWEVER**, for Pattern A (per-client sequential), there's a subtlety in the original SFL framework: clients process activations **as they arrive** (async), and the server model is shared and updated per-client. This means client 2's server forward uses a server model already updated by client 1's backward. This sequential dependency is preserved naturally in Pattern A's `_run_sfl_round_per_client`.

For the **client model**, the original framework sends the same `split_models[0]` to all clients but each client loads it independently. In the unified framework, since we use one shared `nn.Module`, we must explicitly restore state between clients.

---

### ERRATA 4: `local_epochs` Means Full Dataset Passes, Not Fixed Iteration Count

**Affected sections:** Section 7 (`_run_sfl_round` iteration loop)

**What the design says (WRONG):**
```python
for iteration in range(ctx.iterations):
    batch = get_next_batch(state)
```
This implies a fixed number of iterations, with batches drawn sequentially.

**What the code actually does:**
From `sfl_model_trainer.py:169-172`:
```python
for epoch in range(self.training_params["local_epochs"]):
    for batch in tqdm(self.trainloader, desc="Training Batches", disable=TQDM_DISABLED):
        # ... forward → send activation → receive gradient → backward
```

`local_epochs` means **full passes through the client's entire dataset**. Each epoch iterates over the client's `DataLoader` completely.

**Corrected design for Pattern A (per-client):**
```python
for epoch in range(ctx.local_epochs):
    for cid in ctx.client_order:
        state = ctx.client_states[cid]
        for batch in state.dataloader:  # Full pass through client's data
            act, labels = client_forward(state, batch, ctx.device)
            # ... server forward/backward, client backward
```

**For Pattern B (concatenated/USFL):**
USFL's dynamic batch scheduler explicitly computes the number of iterations (`k`) to exhaust all client data. So Pattern B uses `ctx.iterations` as designed — but this value comes from `batch_scheduler.create_schedule()`, which **also** aims to fully exhaust data. The key difference is that USFL coordinates iteration count across all clients to ensure balanced consumption.

**For GAS:**
GAS also uses `localEpoch` full dataset passes:
```python
for local_epoch in range(localEpoch):
    for batch_idx, (data, label) in enumerate(trainloader):
```

**Summary**: Pattern A (SFL/SCAFFOLD/GAS) uses `local_epochs × len(dataloader)` iterations. Pattern B (USFL/Mix2SFL) uses `iterations` from the dynamic batch scheduler.

---

### ERRATA 5: MultiSFL Uses Double-Forward Pattern

**Affected sections:** Section 8.6 (MultiSFL Hook), Section 7 (`_run_sfl_round`)

**What the design says (WRONG):**
Standard single forward pass: `client_model(x) → activation → server_model(activation)`.

**What MultiSFL actually does:**
From `multisfl/client.py:76-105` (`forward_features`) and `client.py:107-145` (`apply_feature_grad`):

```python
# Step 1: Forward WITHOUT gradient (for collecting activations to send to server)
def forward_features(self, w_c, batch_size):
    w_c.eval()
    x, y = self._get_iter(batch_size)
    with torch.no_grad():
        f = w_c(x)                    # No grad → fast, memory-efficient
    f = f.detach()
    cache = ForwardCache(x=x, y=y)    # Save raw input for re-forward
    return f, y.detach(), label_dist, cache, base_count

# Step 2: Apply gradient by RE-FORWARDING with gradient
def apply_feature_grad(self, w_c, opt_c, cache, grad_f):
    w_c.train()
    opt_c.zero_grad(set_to_none=True)
    f = w_c(cache.x)                  # Re-forward WITH gradient graph
    f.backward(grad_f)                # Now autograd can compute client gradients
    opt_c.step()
```

**Why double-forward?** In the original MultiSFL code, the client model's activations are sent to the server for forward + backward. But since `f.detach()` was called in step 1, the gradient graph is broken. To actually update the client model, MultiSFL re-runs the forward pass (step 2) with the gradient enabled, then calls `.backward()` with the server-provided gradient.

**Impact on unified framework:**
The `_run_sfl_round_per_client()` for MultiSFL must use this two-phase pattern. Options:

1. **Hook-based approach (recommended):** Add `client_forward()` and `client_backward()` as overridable hooks:
   ```python
   class BaseMethodHook:
       def client_forward(self, state, batch, device):
           """Default: single forward with grad."""
           act = state.client_model(batch[0].to(device))
           return act, batch[1].to(device), None  # (activation, labels, cache)

       def client_backward(self, state, activation, grad, cache):
           """Default: backward on existing computation graph."""
           activation.backward(grad)
           state.client_optimizer.step()

   class MultiSFLHook(BaseMethodHook):
       def client_forward(self, state, batch, device):
           """Double-forward: no_grad forward for activation collection."""
           state.client_model.eval()
           x, y = batch[0].to(device), batch[1].to(device)
           with torch.no_grad():
               act = state.client_model(x)
           act = act.detach()
           cache = {'x': x, 'y': y}
           return act, y, cache

       def client_backward(self, state, activation, grad, cache):
           """Re-forward with grad, then backward."""
           state.client_model.train()
           state.client_optimizer.zero_grad()
           f = state.client_model(cache['x'])
           f.backward(grad)
           state.client_optimizer.step()
   ```

2. **In `_run_sfl_round_per_client`**, call `self.hook.client_forward()` and `self.hook.client_backward()` instead of the inline implementations.

---

### ERRATA 6: GAS Maintains Per-Client Persistent State Across Rounds

**Affected sections:** Section 8.5 (GAS Hook), Section 6 (Data Structures)

**What the design says (WRONG):**
GAS is treated as a simple hook with `pre_round()`/`post_round()` and no persistent state beyond what's in `RoundContext`.

**What GAS actually does:**
GAS maintains **`IncrementalStats`** — a persistent statistical model that accumulates across ALL rounds:

From `GAS_main.py:393-462`:
```python
class IncrementalStats:
    def __init__(self, device, diagonal=False):
        self.means = {}        # Per-label running mean of activations
        self.variances = {}    # Per-label running variance (or covariance matrix)
        self.weight = {}       # Accumulated weight per label
        self.counts = {}       # Update count per label

    def update(self, new_mean, new_var_or_cov, new_weight, label):
        """Weighted incremental update of per-label activation statistics."""
        decay_factor = old_weight / (old_weight + new_weight)
        self.means[label] = decay_factor * old_mean + (1 - decay_factor) * new_mean
        # ... variance update with Welford-like formula
```

This is used for **feature generation**: when a client doesn't have data for a certain class, GAS generates synthetic activations from the learned distribution `N(mean[label], variance[label])`.

**Key persistent state in GAS:**
1. **`IncrementalStats`**: Running mean/variance of activations per label (updated every round by every participating client)
2. **Feature sampling/generation**: `sample_or_generate_features()` uses these stats to create balanced activation batches for server training
3. **V-value history**: `total_v_value` accumulates across rounds
4. **Client rotation state**: `replace_user()` tracks which clients have been used

**Corrected design for GASHook:**
```python
class GASHook(BaseMethodHook):
    def __init__(self, config):
        super().__init__(config)
        # Persistent state across rounds
        self.feature_stats = IncrementalStats(
            device=config.device,
            diagonal=(config.model_name in ['resnet18', 'resnet18_flex'])
        )
        self.v_value_history = []
        self.used_clients = set()

    def pre_round(self, trainer, round_num):
        # Client selection with rotation (avoid reusing recent clients)
        ...

    def process_activations(self, activations, labels, ctx):
        """
        GAS-specific: Update feature statistics, then generate balanced features.
        Called AFTER client forward, BEFORE server forward.
        """
        # 1. Update IncrementalStats with this round's activations
        for act, lbl in zip(activations, labels):
            mean = act.mean(dim=0)
            var = act.var(dim=0)
            for label in lbl.unique():
                mask = lbl == label
                label_act = act[mask]
                self.feature_stats.update(
                    label_act.mean(0), label_act.var(0),
                    new_weight=mask.sum().item(), label=label.item()
                )

        # 2. Generate balanced features using stats
        balanced_act, balanced_labels = sample_or_generate_features(
            torch.cat(activations), torch.cat(labels),
            batchsize=..., num_labels=...,
            stats=self.feature_stats, ...
        )
        return balanced_act, balanced_labels

    def compute_loss(self, logits, labels, ctx):
        """GAS logit adjustment before loss computation."""
        # Apply label frequency-based logit adjustment
        adjusted_logits = logits + self.logit_adjustment
        return F.cross_entropy(adjusted_logits, labels)

    def post_round(self, trainer, round_num, ctx, results):
        # V-value computation
        v = calculate_v_value(...)
        self.v_value_history.append(v)
        ...
```

**Additional hook point needed**: `process_activations()` must be added to `BaseMethodHook` (default: pass-through). This is called in `_run_sfl_round_per_client` **after** all client forwards (or per-client for GAS) but **before** server forward. This is distinct from `process_gradients()` which happens after server backward.

---

### Summary of All Corrections

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 1 | SFL uses per-client server training, not concatenated | **Critical** — wrong gradient computation | Two `_run_sfl_round` modes: `per_client` vs `concatenated` |
| 2 | Server optimizer.step() frequency differs | **Critical** — K steps vs 1 step per iteration | Encoded in the two modes from ERRATA 1 |
| 3 | FlexibleResNet returns references | **Medium** — client model corruption if not handled | Use `state_dict()` clone, not deepcopy; restore before each client |
| 4 | local_epochs = full dataset passes | **High** — wrong training volume | Pattern A: nested `for epoch / for batch`; Pattern B: scheduler iterations |
| 5 | MultiSFL double-forward pattern | **High** — client model won't train without it | Hook-based `client_forward()` / `client_backward()` overrides |
| 6 | GAS persistent IncrementalStats | **High** — feature generation won't work | GASHook carries persistent state; add `process_activations()` hook |

### Updated Hook Interface (with ERRATA corrections applied)

The complete `BaseMethodHook` interface after all corrections:

```python
class BaseMethodHook(ABC):
    """Base class for all method-specific hooks."""

    @property
    def server_training_mode(self) -> str:
        """'per_client' or 'concatenated'. Default: per_client."""
        return 'per_client'

    @property
    def is_multi_branch(self) -> bool:
        """True only for MultiSFL. Default: False."""
        return False

    def pre_round(self, trainer, round_num) -> RoundContext: ...
    def post_round(self, trainer, round_num, ctx, results) -> RoundResult: ...

    # Multi-branch only (MultiSFL)
    def pre_branch(self, trainer, round_num, branch) -> RoundContext: ...
    def post_branch(self, trainer, round_num, branch, ctx, results): ...
    def post_round_multi(self, trainer, round_num) -> RoundResult: ...

    # Client-side hooks (overridden by MultiSFL for double-forward)
    def client_forward(self, state, batch, device):
        """Single forward with grad (default). Returns (activation, labels, cache)."""
        x, y = batch[0].to(device), batch[1].to(device)
        act = state.client_model(x)
        return act, y, None

    def client_backward(self, state, activation, grad, cache):
        """Backward on existing graph (default)."""
        activation.backward(grad)
        state.client_optimizer.step()

    # Activation processing (overridden by GAS for feature generation)
    def process_activations(self, activations, labels, ctx):
        """Process activations before server forward. Default: pass-through."""
        return torch.cat(activations, dim=0), torch.cat(labels, dim=0)

    # Gradient processing (overridden by USFL for gradient shuffle)
    def process_gradients(self, grads, activations, labels, ctx):
        """Process gradients before client backward. Default: pass-through."""
        return grads

    # Loss computation (overridden by GAS for logit adjustment)
    def compute_loss(self, logits, labels, ctx):
        """Compute loss from logits and labels. Default: CrossEntropyLoss."""
        return F.cross_entropy(logits, labels)
```

### Method → Mode Mapping

| Method | `server_training_mode` | `is_multi_branch` | Key Hook Overrides |
|--------|----------------------|-------------------|-------------------|
| SFL | `per_client` | `False` | (none — uses all defaults) |
| USFL | `concatenated` | `False` | `process_gradients` (shuffle), `pre_round` (balancing) |
| SCAFFOLD | `per_client` | `False` | `pre_round`/`post_round` (control variates) |
| Mix2SFL | `concatenated` | `False` | `process_activations` (SmashMix), `process_gradients` (GradMix) |
| GAS | `per_client` | `False` | `process_activations` (feature gen), `compute_loss` (logit adj) |
| MultiSFL | `per_client` | `True` | `client_forward`/`client_backward` (double-forward), branch hooks |
| FedCBS | `per_client` | `False` | `pre_round` (class-balanced selection) |

---

## Appendix A: SFL Training Flow Explained

For readers unfamiliar with SFL, here is the complete training flow for one round:

```
┌─────────── ONE ROUND OF SFL TRAINING ───────────┐
│                                                   │
│  1. SELECT CLIENTS                                │
│     Pick K out of N clients (e.g., 10 out of 100) │
│                                                   │
│  2. SPLIT MODEL                                   │
│     full_model = [layers 1-4]                     │
│     client_model = [layers 1-2]  (bottom)         │
│     server_model = [layers 3-4]  (top)            │
│                                                   │
│  3. FOR EACH LOCAL ITERATION:                     │
│     ┌─────────────────────────────────────────┐   │
│     │  FOR EACH CLIENT c:                     │   │
│     │    batch = c.next_batch()               │   │
│     │    activation = client_model(batch)     │   │
│     │    activations.append(activation)       │   │
│     │                                         │   │
│     │  SERVER SIDE:                           │   │
│     │    concat = cat(activations)            │   │
│     │    logits = server_model(concat)        │   │
│     │    loss = CrossEntropy(logits, labels)  │   │
│     │    loss.backward()                      │   │
│     │    act_grad = concat.grad               │   │
│     │    server_optimizer.step()              │   │
│     │                                         │   │
│     │  [HOOK: process_gradients(act_grad)]    │   │
│     │                                         │   │
│     │  FOR EACH CLIENT c:                     │   │
│     │    grad_c = act_grad[c's slice]         │   │
│     │    activation_c.backward(grad_c)        │   │
│     │    client_optimizer.step()              │   │
│     └─────────────────────────────────────────┘   │
│                                                   │
│  4. COLLECT client models (state_dicts)            │
│                                                   │
│  5. AGGREGATE (e.g., FedAvg weighted average)      │
│     global_model = weighted_avg(client_models)     │
│                                                   │
│  6. EVALUATE on test set → accuracy, loss          │
└───────────────────────────────────────────────────┘
```

---

## Appendix B: Existing Code Reference

### File locations (all relative to repo root `USFL-fork/`)

#### SFL Framework
```
sfl_framework-fork-feature-training-tracker/
├── simulation.py                                    # Current entry point (665 lines)
├── server/
│   ├── server_args.py                               # Config dataclass (223 fields)
│   ├── modules/
│   │   ├── model/
│   │   │   ├── base_model.py                        # BaseModel interface
│   │   │   ├── model.py                             # get_model() factory
│   │   │   ├── resnet.py, vgg11.py, alexnet.py      # Model implementations
│   │   │   ├── flexible_resnet.py                   # Fine-grained splitting
│   │   │   └── deit.py                              # Vision Transformer
│   │   ├── dataset/
│   │   │   ├── dataset.py                           # get_dataset() factory
│   │   │   ├── cifar.py, mnist.py, glue.py          # Dataset implementations
│   │   ├── trainer/
│   │   │   ├── trainer.py                           # State machine trainer
│   │   │   ├── splitter/
│   │   │   │   ├── splitter.py                      # get_splitter() factory + BaseSplitter
│   │   │   │   ├── resnet_splitter.py, flexible_resnet_splitter.py, ...
│   │   │   │   └── strategy/                        # LayerName, RatioParam, RatioLayer
│   │   │   ├── seletor/                             # (note: typo is intentional, preserved)
│   │   │   │   ├── selector.py                      # get_selector() factory
│   │   │   │   ├── uniform_selector.py
│   │   │   │   └── usfl_selector.py                 # 300+ lines, multi-phase selection
│   │   │   ├── aggregator/
│   │   │   │   ├── aggregator.py                    # get_aggregator() factory
│   │   │   │   ├── fedavg_aggregator.py
│   │   │   │   └── usfl_aggregator.py               # Label-capped weighted
│   │   │   ├── distributer/
│   │   │   │   ├── distributer.py                   # get_distributer() factory
│   │   │   │   └── shard_dirichlet_distributer.py   # Primary Non-IID strategy
│   │   │   ├── scheduler/
│   │   │   │   └── batch_scheduler.py               # create_schedule() for USFL
│   │   │   ├── stage/
│   │   │   │   ├── sfl_stage_organizer.py           # 694 lines — SFL ML logic + communication
│   │   │   │   ├── usfl_stage_organizer.py          # 1642 lines — USFL ML logic + communication
│   │   │   │   ├── scaffold_stage_organizer.py      # SCAFFOLD
│   │   │   │   └── mix2sfl_stage_organizer.py       # Mix2SFL
│   │   │   └── utils/
│   │   │       └── training_tracker.py              # Metrics logging
│   │   └── ws/                                      # ← ALL COMMUNICATION CODE (NOT reused)
│   │       ├── connection.py                        # WebSocket
│   │       ├── inmemory_connection.py               # Async queue simulation
│   │       └── handler/                             # Message dispatch
│   └── utils/
│       └── g_measurement.py                         # G measurement system
├── client/
│   └── modules/
│       ├── dataset/
│       │   └── maskable_dataset.py                  # USFL data balancing
│       └── trainer/
│           └── model_trainer/
│               ├── sfl_model_trainer.py             # Client-side SFL training
│               └── usfl_model_trainer.py            # Client-side USFL training
└── shared/
    └── update_alignment.py                          # A_cos, M_norm metrics
```

#### GAS Framework
```
GAS_implementation/
├── GAS_main.py                  # 1697 lines — everything (training loop, config, etc.)
└── utils/
    ├── network.py               # 1672 lines — model definitions + splitting
    ├── dataset.py               # 557 lines — data partitioning
    ├── g_measurement.py         # 603 lines — 3-perspective G measurement
    ├── utils.py                 # V-value, feature generation, local adjustment
    ├── drift_measurement.py     # SCAFFOLD-style drift
    └── log_utils.py             # Logging utilities
```

#### MultiSFL Framework
```
multisfl_implementation/
├── run_multisfl.py              # Entry point + arg parsing
└── multisfl/
    ├── trainer.py               # MultiSFLTrainer (~770 lines)
    ├── servers.py               # FedServer, MainServer
    ├── client.py                # Client (forward, backward, replay)
    ├── replay.py                # ScoreVectorTracker, KnowledgeRequestPlanner
    ├── scheduler.py             # SamplingProportionScheduler
    ├── models.py                # Model splitting
    └── data.py                  # Data partitioning
```

---

## Summary

| Aspect | Value |
|--------|-------|
| **New code to write** | ~1,840 lines across 12 files |
| **Existing code reused** | ~5,000+ lines (models, splitters, selectors, aggregators, distributers, datasets, measurement) |
| **Methods supported** | 7 (SFL, USFL, SCAFFOLD, Mix2SFL, GAS, MultiSFL, FedCBS) |
| **Architecture** | Synchronous, no async, no queues, no polling |
| **Model copy strategy** | `load_state_dict()` instead of `deepcopy()` |
| **Method extensibility** | Add one file in `methods/` implementing BaseMethodHook |
| **Expected speedup** | 2-5x over current SFL framework simulation mode |
| **Backwards compatibility** | 100% — existing frameworks untouched |
