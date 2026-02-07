# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference: Four Research Tracks

| Track | Location | Entry Point | Key Focus |
|-------|----------|-------------|-----------|
| **SFL** | `sfl_framework-fork-feature-training-tracker/` | `simulation.py` (method=sfl) | Base split federated learning |
| **USFL** | `sfl_framework-fork-feature-training-tracker/` | `simulation.py` (method=usfl) | Advanced SFL with 6 optimizations for Non-IID |
| **GAS** | `GAS_implementation/` | `GAS_main.py` | Gradient quality-based client selection |
| **MultiSFL** | `multisfl_implementation/` | `run_multisfl.py` | Multi-branch SFL with knowledge replay |

## Overview

This repository contains **four primary research tracks** for federated learning:

1. **SFL (Split Federated Learning)** - Base SFL implementation
2. **USFL (Unified Split Federated Learning)** - Advanced SFL with data balancing and optimization
3. **GAS (Gradient Adjustment Scheme)** - Gradient quality-based client selection
4. **MultiSFL** - Multi-branch Split Federated Learning

Each track has its own implementation with specialized features for Non-IID data distribution environments.

**Repository Structure:**
- `sfl_framework-fork-feature-training-tracker/`: Unified framework supporting SFL, USFL, and 11+ other methods
- `GAS_implementation/`: Standalone GAS method implementation
- `multisfl_implementation/`: Multi-branch SFL architecture
- `experiment_core/`: Unified experiment pipeline (adapters, spec generation, batch runner)
- `experiment_configs/`: Shared `common.json` + per-method config JSONs
- `shared/`: Cross-framework utilities (update alignment metrics)

## Project Structure

```
USFL-fork/
├── deploy.sh                  # Multi-GPU server deployment automation
├── deploy/                    # Deployment supporting files
│   ├── remote_run.sh          # Remote experiment wrapper (runs on GPU servers)
│   ├── setup_rclone_gdrive.sh # Google Drive rclone setup for servers
│   └── deploy_servers.json    # Server inventory configuration
│
├── experiment_core/           # Unified experiment framework
│   ├── __init__.py
│   ├── generate_spec.py       # Config → batch_spec.json generator
│   ├── batch_runner.py        # Multi-experiment batch execution
│   ├── runner.py              # Single-experiment orchestrator
│   ├── run_experiment.py      # CLI entry point
│   ├── sfl_runner.py          # SFL bridge (spec → simulation.py in-process)
│   ├── spec.py                # Experiment spec dataclass
│   ├── normalization.py       # Result normalization (unified JSON format)
│   ├── test_adapters.py       # Adapter validation tests
│   └── adapters/              # Framework-specific integration
│       ├── __init__.py
│       ├── base.py            # Base adapter interface
│       ├── sfl_adapter.py     # SFL/USFL/SCAFFOLD/MIX2SFL
│       ├── gas_adapter.py     # GAS (env vars)
│       └── multisfl_adapter.py # MultiSFL (CLI args)
│
├── experiment_configs/        # Experiment configuration
│   ├── common.json            # Shared params (dataset, model, rounds, alpha, G, drift)
│   ├── sfl.json               # SFL-specific overrides
│   ├── usfl.json              # USFL-specific overrides
│   ├── sfl_iid.json           # SFL with IID data
│   ├── scaffold.json          # SCAFFOLD_SFL config
│   ├── mix2sfl.json           # Mix2SFL config
│   ├── gas.json               # GAS-specific params
│   └── multisfl.json          # MultiSFL-specific params
│
├── shared/                    # Cross-framework utilities
│   ├── __init__.py
│   └── update_alignment.py    # A_cos & M_norm computation
│
├── docs/                      # Documentation
│   ├── DEAD_CODE_AUDIT.md     # Dead code audit report across all frameworks
│   ├── G_MEASUREMENT_GUIDE.md
│   ├── METRICS_REFERENCE.md   # Complete metrics reference (G, drift, alignment, etc.)
│   ├── SCAFFOLD_SFL_USAGE.md
│   └── UPDATE_ALIGNMENT_DOCS.md
│
├── sfl_framework-fork-feature-training-tracker/  # Track 1 & 2: SFL/USFL
│   ├── simulation.py          # Primary entry point
│   ├── server/
│   │   ├── server_args.py     # Config dataclass
│   │   └── modules/
│   │       ├── dataset/       # CIFAR, MNIST, GLUE datasets
│   │       ├── model/         # ResNet, VGG, AlexNet, DistilBERT
│   │       └── trainer/
│   │           ├── aggregator/ # FedAvg, USFL aggregation
│   │           ├── seletor/    # Client selection (uniform, usfl)
│   │           ├── distributer/# Data distribution strategies
│   │           ├── splitter/   # Model splitting strategies
│   │           ├── scheduler/  # Dynamic batch scheduling (USFL)
│   │           ├── stage/      # Stage organizers (sfl, usfl, scaffold, mix2sfl)
│   │           └── utils/      # Training tracker, USFL logger
│   ├── client/
│   │   └── modules/
│   │       ├── dataset/maskable_dataset.py  ★ USFL balancing
│   │       └── trainer/
│   │           ├── model_trainer/  # Per-method client trainers
│   │           └── stage/          # Client-side stage organizers
│   └── docs/
│       └── COMPREHENSIVE_GUIDE.md  # 950-line Korean USFL specification
│
├── GAS_implementation/         # Track 3: GAS Method
│   ├── GAS_main.py            # Single-file implementation
│   └── utils/                 # G-score, models, data, drift
│
└── multisfl_implementation/    # Track 4: MultiSFL
    ├── run_multisfl.py        # Main entry point
    └── multisfl/              # Multi-branch SFL package
```

**Note:** The main framework directory has a long name (`sfl_framework-fork-feature-training-tracker`) - this is intentional and should be preserved.

## Four Research Tracks

### Track 1: SFL (Split Federated Learning)

**Location:** `sfl_framework-fork-feature-training-tracker/` with `-M sfl`

**Description:** Base implementation of Split Federated Learning where models are split between client (bottom layers) and server (top layers). Clients perform forward pass and send activations; server computes loss and sends gradients back.

**Key Features:**
- Model splitting strategies: layer_name, ratio_param, ratio_layer
- Efficient communication (activations/gradients instead of full model)
- Privacy-preserving (clients don't send raw data or labels)

**Run Command:**
```bash
cd sfl_framework-fork-feature-training-tracker
# Edit simulation.py to set method = "sfl"
python simulation.py
```

### Track 2: USFL (Unified Split Federated Learning)

**Location:** `sfl_framework-fork-feature-training-tracker/` with `-M usfl`

**Description:** Advanced SFL variant designed for extreme Non-IID environments. Implements six core optimizations to handle severe class imbalance across clients.

**Key Features:**
1. **Data Balancing**: Trimming/Replication/Target strategies to balance class distribution
2. **Gradient Shuffle**: Four strategies (random, inplace, average, adaptive_alpha) to mix gradients
3. **USFL Selector**: Missing label + freshness scoring for optimal client selection
4. **USFL Aggregator**: Label-capped weighted aggregation
5. **Dynamic Batch Scheduler**: Full data utilization with consistent batch sizes
6. **Cumulative Usage Tracking**: Exponential bins to track and prioritize fresh data

**Run Command:**
```bash
cd sfl_framework-fork-feature-training-tracker
# Edit simulation.py USFL_OPTIONS to configure experiments
python simulation.py
```

**Documentation:** See `sfl_framework-fork-feature-training-tracker/docs/COMPREHENSIVE_GUIDE.md` (950 lines, Korean) for complete USFL specification.

### Track 3: GAS (Gradient Adjustment Scheme)

**Location:** `GAS_implementation/`

**Description:** SFL-based method using G-measurement (gradient quality metrics) for intelligent client selection and gradient adjustment. Focuses on improving convergence by selecting clients that contribute high-quality gradients.

**Key Features:**
- G-measurement system for gradient quality assessment
- V-value computation for client ranking
- Local adjustment computation
- Sample/generate features for client representation
- Supports ResNet, AlexNet, VGG16
- Label Dirichlet (hybrid shard + Dirichlet) distribution

**Key Files:**
- `GAS_main.py`: Main training loop (single-file implementation)
- `utils/g_measurement.py`: G-score computation
- `utils/utils.py`: Helper functions (calculate_v_value, replace_user, etc.)
- `utils/network.py`: Model definitions and splitting logic
- `utils/dataset.py`: Data partitioning

**Configuration (in GAS_main.py):**
```python
# Data settings
iid = False
label_dirichlet = True
shard = 2  # classes per client
alpha = 0.3  # Dirichlet alpha

# Training settings
epochs = 300
localEpoch = 5
user_num = 100
user_parti_num = 10

# Model settings
split_layer = "layer1.1.bn2"  # Fine-grained split point
```

**Run Command:**
```bash
cd GAS_implementation
python GAS_main.py

# Or with environment overrides
CUDA_VISIBLE_DEVICES=0 python GAS_main.py
```

### Track 4: MultiSFL (Multi-Branch Split Federated Learning)

**Location:** `multisfl_implementation/`

**Description:** Multi-model SFL architecture where multiple branch servers run different model splits simultaneously. Uses knowledge replay and adaptive sampling to balance branch learning.

**Key Features:**
- **Multiple Branch Servers**: Each branch maintains separate model state
- **MainServer + FedServers**: Hierarchical architecture with main aggregation and branch coordination
- **ScoreVectorTracker**: Tracks performance across branches
- **KnowledgeRequestPlanner**: Plans replay of past client data
- **SamplingProportionScheduler**: Dynamically adjusts client sampling per branch
- **BranchClientState/BranchServerState**: Per-branch state management

**Key Components:**
- `run_multisfl.py`: Main entry point with full argument parsing
- `multisfl/trainer.py`: MultiSFLTrainer orchestrates training across branches
- `multisfl/servers.py`: FedServer, MainServer implementations
- `multisfl/client.py`: Client with multi-branch support
- `multisfl/replay.py`: Knowledge replay system
- `multisfl/scheduler.py`: Branch sampling scheduler
- `multisfl/models.py`: Model splitting and initialization
- `multisfl/data.py`: Dataset partitioning (IID, Dirichlet, Shard-Dirichlet)

**Key Arguments:**
```bash
--dataset: synthetic, cifar10, fmnist
--partition: iid, dirichlet, shard_dirichlet
--branches: Number of branch servers (default computed from n_main)
--n_main: Number of clients in main training
--alpha_master_pull: Weight for pulling from main server
--p_update: Sampling proportion update strategy
--use_torchvision_init: Use pretrained torchvision weights
```

**Run Command:**
```bash
cd multisfl_implementation
python run_multisfl.py \
    --dataset cifar10 \
    --partition shard_dirichlet \
    --shards 2 \
    --alpha_dirichlet 0.3 \
    --rounds 100 \
    --num_clients 100 \
    --n_main 10 \
    --branches 3 \
    --batch_size 50 \
    --local_steps 5
```

## Deployment

Distributed experiments across GPU servers are managed by `deploy.sh`:

```bash
./deploy.sh run usfl@server-a:0 gas@server-b:1   # Deploy experiments
./deploy.sh status                                 # Check all servers
./deploy.sh collect --local                        # Collect results
./deploy.sh kill <run_name>                        # Kill a running experiment
./deploy.sh check <run_name>                       # Check experiment status
```

The deployment pipeline: `deploy.sh` → SSH → `deploy/remote_run.sh` (conda activation) → `experiment_core/batch_runner.py` (batch execution) → per-experiment adapters.

Server inventory is configured in `deploy/deploy_servers.json`. Deploy history is tracked in `deploy/.deploy_history.json` (auto-generated, not tracked by git).

## Running Experiments

### Unified Framework (SFL/USFL)

**Primary Command:**
```bash
cd sfl_framework-fork-feature-training-tracker
python simulation.py
```

**Configuration:**
Experiments are configured by modifying `simulation.py` directly. Look for configuration dictionaries like:
- `USFL_OPTIONS`: USFL experiment variants (A, B, C, etc.)
- Parameter grids for batch testing
- `START_INDEX` and `END_INDEX` for workload range control
- `SKIP_FILTERS` for skipping specific parameter combinations

**Quick Start Examples:**
```python
# In simulation.py, configure USFL_OPTIONS:
USFL_OPTIONS = {
    "A": {  # USFL with all features
        "method": "usfl",
        "balancing_strategy": "target",
        "balancing_target": "mean",
        "gradient_shuffle": "true",
        "gradient_shuffle_strategy": "random",
        "use_dynamic_batch_scheduler": "true",
    },
    "B": {  # Base SFL without USFL features
        "method": "sfl",
        "balancing_strategy": "none",
        "gradient_shuffle": "false",
        "use_dynamic_batch_scheduler": "false",
    },
}
```

### Unified Experiment Pipeline

The preferred way to run experiments across all frameworks:

```bash
# Generate experiment spec from configs
python -m experiment_core.generate_spec

# Run a batch of experiments
python -m experiment_core.batch_runner --spec <spec.json> --repo-root .

# Or run a single experiment
python -m experiment_core.run_experiment --spec <spec.json>
```

The pipeline reads from `experiment_configs/common.json` (shared parameters) plus method-specific JSONs (e.g., `usfl.json`, `gas.json`). Adapters translate unified config into each framework's native format (CLI args for SFL/MultiSFL, env vars for GAS).

### Emulation Mode

```bash
# Terminal 1 - Start server
cd sfl_framework-fork-feature-training-tracker/server
python main.py [server args]

# Terminal 2+ - Start clients
cd sfl_framework-fork-feature-training-tracker/client
python main.py [client args]
```

### Testing

```bash
# Run test script
cd sfl_framework-fork-feature-training-tracker
python test.py
```

## Key Architecture Patterns

### Stage Organizer Pattern

All learning methods follow a 3-stage pattern per round:

1. **Pre-Round** (`_pre_round`): Client selection, model splitting, data balancing calculation
2. **In-Round** (`_in_round`): Training execution (FL local training or SFL forward/backward)
3. **Post-Round** (`_post_round`): Model aggregation, evaluation, usage tracking update

Stage organizers are in `server/modules/trainer/stage/`:
- `fl_stage_organizer.py`: Federated Learning
- `sfl_stage_organizer.py`: Split Federated Learning
- `usfl_stage_organizer.py`: USFL (1148 lines, core research component)

### USFL Core Components

**Server Side:**
- `usfl_stage_organizer.py`: Main USFL orchestration with balancing, gradient shuffle, cumulative tracking
- `usfl_selector.py`: Client selection using missing label + freshness scoring
- `usfl_aggregator.py`: Label-capped weighted aggregation
- `batch_scheduler.py`: Dynamic batch size scheduling across clients

**Client Side:**
- `maskable_dataset.py`: Applies trimming/replication based on server instructions
- `usfl_model_trainer.py`: Handles dynamic batching and activation submission
- `usfl_stage_organizer.py`: Client-side stage coordinator

### Communication Flow

**Simulation Mode:**
- Uses `InMemoryConnection` for zero-latency communication
- Server and clients run concurrently via `asyncio.gather()`
- Direct object passing (no serialization)

**Emulation Mode:**
- WebSocket-based communication
- Real network overhead
- Separate processes for server and each client

## Configuration System

### Config Dataclass

All configuration is managed through the `Config` dataclass in `server/server_args.py`. Key parameter categories:

**Basic Training:**
- `-d`: dataset (cifar10, fmnist, mnist, cola, etc.)
- `-m`: model (resnet18, alexnet, vgg11, distilbert, etc.)
- `-M`: method (fl, sfl, sfl-u, usfl, fitfl, scala, etc.)
- `-le`: local_epochs
- `-gr`: global_round
- `-bs`: batch_size
- `-nc`: num_clients
- `-ncpr`: num_clients_per_round

**Data Distribution:**
- `-distr`: distributer (uniform, dirichlet, label, shard_dirichlet)
- `-diri-alpha`: dirichlet_alpha (lower = more Non-IID)
- `-lpc`: labels_per_client
- `-mrs`: min_require_size

**SFL-Specific:**
- `-ss`: split_strategy (layer_name, ratio_param, ratio_layer)
- `-sl`: split_layer (e.g., "layer1.1.bn2" for ResNet)
- `-sr`: split_ratio (for ratio_* strategies)

**USFL-Specific:**
- `-bstrat`: balancing_strategy (trimming, replication, target)
- `-btarget`: balancing_target (mean, median, or numeric)
- `-gs`: gradient_shuffle (flag)
- `-gss`: gradient_shuffle_strategy (random, inplace, average, average_adaptive_alpha)
- `-gaw`: gradient_average_weight
- `-aab`: adaptive_alpha_beta
- `-udbs`: use_dynamic_batch_scheduler (flag)
- `-ucu`: use_cumulative_usage (flag)
- `-ufs`: use_fresh_scoring (flag)
- `-udf`: usage_decay_factor
- `-fdr`: freshness_decay_rate

**Selector/Aggregator:**
- `-s`: selector (uniform, usfl, missing_class, fedcbs)
- `-aggr`: aggregator (fedavg, usfl, fitfl)

### Boolean Arguments

The framework uses a custom `str_to_bool` converter. Accepted values:
- True: "true", "1", "yes"
- False: "false", "0", "no"

## USFL-Specific Features (Track 2)

The following features are specific to USFL implementation in the unified framework.

### 1. Data Balancing Strategies

**Purpose:** Address class imbalance in Non-IID settings

**Strategies:**
- **Trimming** (default): Reduce all classes to minimum class size
- **Replication**: Replicate underrepresented classes to maximum size
- **Target** (recommended): Hybrid approach using mean/median/custom target

**Implementation:** Server calculates `augmented_dataset_sizes` in `_pre_round`, clients apply via `MaskableDataset.update_amount_per_label()`

### 2. Gradient Shuffle

**Purpose:** Mix gradients across clients to prevent bias and enhance privacy

**Strategies:**
- `random`: Simple random permutation
- `inplace`: Class-balanced shuffle (ensures equal class distribution per client)
- `average`: Mix with global mean gradient
- `average_adaptive_alpha`: Cosine similarity-based adaptive mixing (most sophisticated)

**Location:** `usfl_stage_organizer.py` `_in_round` method (server-side)

### 3. Dynamic Batch Scheduler

**Purpose:** Fully utilize all client data while maintaining consistent batch sizes

**Algorithm:** Calculates optimal iteration count and per-client batch sizes to exhaust data evenly

**Files:**
- `scheduler/batch_scheduler.py`: `create_schedule()` function
- `usfl_model_trainer.py`: Client-side batch handling

### 4. Cumulative Usage Tracking

**Purpose:** Track data freshness using exponential bins to prioritize underused data

**Structure:** `cumulative_usage[client_id][label] = {bin_0: count, bin_1: count, ...}`

**Bins:** Exponential grouping (0, 1, 2-3, 4-7, 8-15, ...)

**Update:** In `_post_round` after aggregation

### 5. Fresh Scoring

**Purpose:** Select clients with fresher (less frequently used) data during client selection

**Integration:** Part of USFL Selector's Phase 3 (after missing label filtering)

## Important Implementation Details

### Model Splitting

For SFL methods, models are split between client and server:
- **Client model**: Bottom layers (feature extraction)
- **Server model**: Top layers (classification head)
- Split point configured via `-sl` (layer name) or `-sr` (ratio)

### Activation/Gradient Exchange

SFL training flow:
1. Client forward → generates activations
2. Client sends activations to server
3. Server forward (server model) → computes loss
4. Server backward → computes gradients
5. Server sends gradients back to client
6. Client backward (client model)

USFL modifies step 5 with gradient shuffle before returning.

### Non-IID Data Distribution

**Recommended:** `shard_dirichlet` distributer
- Guarantees exact label assignments per client
- Uses Dirichlet distribution for within-class heterogeneity
- Parameters: `-diri-alpha` (heterogeneity) + `-lpc` (labels per client)

### Result Parsing

```bash
cd sfl_framework-fork-feature-training-tracker
python result_parser.py
```

Parses experiment logs and generates summary statistics.

## Development Guidelines

### Track-Specific Modifications

#### Unified Framework (SFL/USFL)

**Modifying USFL Behavior:**
1. **Balancing strategy:** Edit `_pre_round` in `server/.../usfl_stage_organizer.py` (lines ~550-700)
2. **Gradient shuffle:** Edit `_shuffle_gradients` in `_in_round` (lines ~800-950)
3. **Selector logic:** Edit `server/.../usfl_selector.py`
4. **Aggregator weights:** Edit `server/.../usfl_aggregator.py`

**Adding New Methods:**
1. Create stage organizer in `server/modules/trainer/stage/`
2. Inherit from `BaseStageOrganizer`
3. Implement `_pre_round`, `_in_round`, `_post_round`
4. Register in `server/server.py` method routing

**Adding New Models:**
1. Add model class in `server/modules/model/`
2. Inherit from `BaseModel`
3. Implement `get_model()`, `get_optimizer()`, etc.
4. Register in `server/modules/model/model.py`

#### GAS Implementation

**Modifying GAS Behavior:**
- All configuration in `GAS_main.py` (single-file implementation)
- G-measurement logic in `utils/g_measurement.py`
- V-value and adjustment logic in `utils/utils.py`
- Model/split definitions in `utils/network.py`

**Key Variables to Tune:**
- Client selection: `user_parti_num`
- Gradient quality threshold (in g_measurement)
- V-value computation parameters
- Split point: `split_layer`

#### MultiSFL Implementation

**Modifying MultiSFL Behavior:**
- Main training loop: `multisfl/trainer.py` (`MultiSFLTrainer` class)
- Branch coordination: `multisfl/servers.py` (`MainServer`, `FedServer`)
- Replay strategy: `multisfl/replay.py` (`KnowledgeRequestPlanner`)
- Sampling scheduler: `multisfl/scheduler.py` (`SamplingProportionScheduler`)

**Key Parameters to Tune:**
- `--branches`: Number of branch servers
- `--alpha_master_pull`: Main server pull weight
- `--p_update`: Sampling proportion update strategy (abs_ratio, loss_inverse, etc.)
- `--p_min`, `--p_max`: Sampling proportion bounds

## Debugging Tips

### Common Issues (All Tracks)

1. **CUDA Out of Memory:**
   - **Unified Framework**: Reduce `-bs` (batch_size), `-nc`, or `-ncpr`
   - **GAS**: Reduce `batchSize`, `user_num`, or `user_parti_num` in `GAS_main.py`
   - **MultiSFL**: Reduce `--batch_size`, `--num_clients`, or `--branches`
   - Use smaller models (e.g., tiny_vgg11 instead of vgg11)

2. **Stuck Workload:**
   - **Unified Framework**: Set `WORKLOAD_TIMEOUT` in `simulation.py`, use `START_INDEX`/`END_INDEX`
   - **GAS**: Check for infinite loops in client selection or gradient computation
   - **MultiSFL**: Check branch synchronization issues in trainer

3. **Data Distribution Issues:**
   - **All tracks**: Increase `min_require_size` (or `-mrs`) for shard_dirichlet
   - Adjust Dirichlet alpha (0.1 = extreme Non-IID, 10.0 = nearly IID)
   - **MultiSFL**: Ensure all branches have sufficient client coverage

4. **Split Layer Issues:**
   - Use `layer1.1.bn2` for ResNet18 (tested and stable)
   - For AlexNet/VGG: check split layer names with `print(model)` first
   - **GAS**: `split_layer` variable in `GAS_main.py`
   - **Unified**: `-sl` argument or in config dictionary

### Track-Specific Debugging

#### Unified Framework (SFL/USFL)
- `USFLLogger`: USFL-specific events (balancing decisions, shuffle stats)
- `TrainingTracker`: Per-round metrics (accuracy, loss, timing)
- Console output shows round-by-round progress
- Check `augmented_dataset_sizes` for balancing correctness

#### GAS
- Print G-scores to verify gradient quality measurement
- Check V-values for client ranking
- Monitor local adjustment effectiveness
- Verify feature sampling/generation logic

#### MultiSFL
- Monitor branch-wise accuracies (can diverge significantly)
- Check `ScoreVectorTracker` for performance tracking
- Verify `SamplingProportionScheduler` adjustments
- Ensure `alpha_master_pull` prevents branch drift

## Language Note

The comprehensive guide (`sfl_framework-fork-feature-training-tracker/docs/COMPREHENSIVE_GUIDE.md`) is written in Korean and contains the complete technical specification of USFL (950 lines). Refer to it for detailed algorithmic explanations, mathematical formulas, and implementation specifics.

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch 2.4.1 (CUDA 12.x compatible)
- Transformers 4.45.1 (for NLP models)
- FastAPI/WebSockets (for emulation mode)
- NumPy, Pandas, scikit-learn (data processing)

Install with:
```bash
cd sfl_framework-fork-feature-training-tracker
pip install -r requirements.txt
```

For CUDA support:
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```
