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

## GPU Server Deployment (deploy.sh)

This section explains how to run experiments on remote GPU servers. `deploy.sh` automates the entire lifecycle: git push → SSH → experiment launch → monitoring → result collection.

### Architecture Overview

```
┌─ Local Machine ─────────────────────────────────────────────────────┐
│  experiment_configs/       deploy.sh                                │
│  ├── common.json    ───→  1. git push                              │
│  ├── usfl.json             2. generate_spec (per method)            │
│  ├── gas.json              3. scp spec to server                    │
│  └── ...                   4. SSH → tmux + remote_run.sh            │
└──────────────────────────────────────────────────────────────────────┘
                                    │ SSH
                                    ▼
┌─ GPU Server (tmux session) ──────────────────────────────────────────┐
│  deploy/remote_run.sh                                                │
│  ├── conda activate                                                  │
│  └── python -m experiment_core.batch_runner --spec batch_spec.json   │
│       └── runner.py → adapter.build_command() → subprocess           │
│            ├── SFL:      sfl_runner.py → simulation.py (in-process)  │
│            ├── GAS:      GAS_main.py (env vars)                      │
│            └── MultiSFL: run_multisfl.py (CLI args)                  │
│                                                                      │
│  Results → results/<run_name>/                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

**1. SSH config** — Password-less SSH to GPU servers must work:
```bash
# ~/.ssh/config example
Host xsailor4-pj
    HostName 10.0.0.4
    User jhkang
    IdentityFile ~/.ssh/id_ed25519
```
Test: `ssh xsailor4-pj "hostname"` should succeed without prompts.

**2. Server environment** — Each GPU server needs:
- Git clone of this repo (same remote)
- Conda environment with all dependencies installed
- `nvidia-smi` available (for GPU status checks)
- `tmux` installed

**3. Local tools** — `jq` is required: `brew install jq`

### Server Configuration

Edit `deploy/deploy_servers.json`:
```json
{
  "servers": {
    "xsailor4": {
      "ssh_host": "xsailor4-pj",
      "remote_repo": "/home/xsailor5/jhkang/USFL-fork",
      "conda_env": "usfl_env",
      "gpus": [0, 1, 2, 3]
    },
    "xsailor5": {
      "ssh_host": "xsailor5-pj",
      "remote_repo": "/home/xsailor5/jhkang/USFL-fork",
      "conda_env": "usfl_env",
      "gpus": [0, 1, 2, 3]
    }
  },
  "gdrive_rclone_remote": "",
  "tmux_session": "usfl-exp"
}
```

Fields:
- `ssh_host`: SSH alias from `~/.ssh/config` (NOT raw IP)
- `remote_repo`: Absolute path to git clone on the server
- `conda_env`: Conda environment name on the server
- `gpus`: Available GPU indices (informational, for `servers` command)
- `gdrive_rclone_remote`: Optional rclone remote for Google Drive result upload
- `tmux_session`: Prefix for tmux session names (default: `usfl-exp`)

### Experiment Configuration

All experiment parameters live in `experiment_configs/`:

**`common.json`** — Shared across ALL methods:
```json
{
  "dataset": "cifar10",
  "model": "resnet18_flex",
  "rounds": 100,
  "alpha": 0.3,
  "labels_per_client": 2,
  "total_clients": 100,
  "clients_per_round": 10,
  "local_epochs": 5,
  "learning_rate": 0.001,
  "split_layer": "layer2",
  "enable_drift_measurement": true,
  "enable_g_measurement": false,
  "g_measure_frequency": 10
}
```

**Per-method JSONs** — Override/extend common params:
- `sfl.json`, `usfl.json`, `scaffold.json`, `mix2sfl.json` — SFL framework variants
- `gas.json` — GAS-specific (generate, v_test, sample_frequency)
- `multisfl.json` — MultiSFL-specific (branches, alpha_master_pull, p_update)

Keys prefixed with `_` are comments (filtered by `build_overrides()` during spec generation).

**To change experiment parameters:**
1. Edit `common.json` for shared params (dataset, rounds, alpha, etc.)
2. Edit the per-method JSON for method-specific params
3. Deploy — `deploy.sh` reads these files to generate specs

### Deploying Experiments

#### Step-by-Step Workflow

```bash
# 1. Edit experiment configs
#    (Change dataset, alpha, rounds, method-specific params, etc.)

# 2. Deploy to GPU servers
./deploy.sh run usfl@xsailor4:0 gas@xsailor5:0 sfl@xsailor4:1 multisfl@xsailor5:1

# 3. Monitor progress
./deploy.sh status          # Overview of all servers
./deploy.sh check           # Detailed status of latest deployment
./deploy.sh logs xsailor4 usfl   # Tail log of specific experiment

# 4. Wait for completion, then collect results
./deploy.sh collect --local  # rsync results to local ./results/

# 5. Generate dashboard
python sfl_dashboard.py results/<run_name>/ --open
```

#### Run Command Formats

```bash
# Inline format: method@server:gpu
./deploy.sh run usfl@xsailor4:0 gas@xsailor5:1

# Server flag format: all methods on one server
./deploy.sh run --server xsailor4 --methods "usfl sfl gas" --gpus "0 1 2"

# Interactive mode: SSH into server with conda activated
./deploy.sh run -i --server xsailor4

# Skip git push (if already pushed)
./deploy.sh run --no-push usfl@xsailor4:0

# Reuse existing run name (add experiments to existing result directory)
./deploy.sh run --run-name cifar10_a0.3_r100_20260207_222446 scaffold@xsailor4:2
```

#### What `deploy.sh run` Does Internally

1. **Git push** — Commits any uncommitted changes and pushes to remote
2. **Generate spec** — For each method, runs `generate_spec.py` with method-specific config:
   ```bash
   python -m experiment_core.generate_spec \
       --config-dir experiment_configs \
       --methods usfl \
       --gpu-map '{"usfl": 0}' \
       --output-dir results/<run_name> \
       --output /tmp/usfl_spec.json
   ```
3. **Git sync** — SSH to each server: `git fetch && git checkout <branch> && git reset --hard origin/<branch>`
4. **SCP spec** — Transfer generated `batch_spec_<method>.json` to server
5. **Tmux launch** — Start each experiment in its own tmux session:
   ```bash
   tmux new-session -d -s 'usfl-exp-usfl-222446' \
       'bash deploy/remote_run.sh usfl_env batch_spec_usfl.json 2>&1 | tee experiment_usfl-222446.log'
   ```
6. **Record history** — Saves deployment info to `deploy/.deploy_history.json`

#### Run Naming Convention

Auto-generated: `{dataset}_a{alpha}_r{rounds}_{YYYYMMDD_HHMMSS}`
Example: `cifar10_a0.3_r100_20260207_222446`

All methods in a single deployment share the same `run_name` → results go to `results/<run_name>/` on each server.

### Monitoring

```bash
# Overview: all servers, running experiments, GPU utilization
./deploy.sh status

# Detailed check: experiment completion status for a specific deployment
./deploy.sh check                    # Latest pending deployment
./deploy.sh check cifar10_a0.3_r100_20260207_222446

# Live log tail
./deploy.sh logs xsailor4 usfl       # Tail usfl experiment log
./deploy.sh logs xsailor4            # List available logs

# Attach to tmux session (interactive, real-time output)
./deploy.sh attach xsailor4 usfl     # Ctrl+B D to detach
./deploy.sh attach xsailor4          # List sessions

# View deploy history
./deploy.sh collect --list
```

### Killing Experiments

```bash
# Interactive: shows numbered list, pick which to kill
./deploy.sh kill

# Kill all experiments on all servers
./deploy.sh kill --all

# Kill all experiments on a specific server
./deploy.sh kill xsailor4

# Kill a specific method on a server
./deploy.sh kill xsailor4 usfl
```

### Collecting Results

```bash
# Collect ALL pending results (rsync to local ./results/)
./deploy.sh collect --local

# Collect specific run
./deploy.sh collect --local cifar10_a0.3_r100_20260207_222446

# Re-collect already-collected runs
./deploy.sh collect --local --all

# Upload to Google Drive (requires rclone setup on servers)
./deploy.sh collect
```

Result files are merged from multiple servers into a single `results/<run_name>/` directory. The `collect` command:
- Checks if experiments are still running (marks as `partial` if so)
- Marks as `collected` only when all tmux sessions have exited
- On re-run, picks up any `partial` results automatically

### Result Analysis

```bash
# Generate interactive HTML dashboard with all experiments compared
python sfl_dashboard.py results/<run_name>/ --open

# Custom output path
python sfl_dashboard.py results/<run_name>/ --output my_report.html
```

The dashboard auto-detects SFL, GAS, and MultiSFL result formats, extracts accuracy curves, drift metrics, G measurements, V-values, and per-round metrics, and generates an interactive Plotly dashboard.

**Hosting dashboard for mobile/remote access (via Tailscale):**

This machine has Tailscale VPN configured. To view the dashboard on a phone or another device on the Tailscale network:

```bash
# 1. Generate dashboard
python sfl_dashboard.py results/<run_name>/

# 2. Start HTTP server (bind to all interfaces so Tailscale can reach it)
nohup python -m http.server 8765 --bind 0.0.0.0 \
    --directory results/<run_name>/ > /tmp/dashboard_server.log 2>&1 &
echo "PID: $!"

# 3. Get Tailscale IP
tailscale ip -4    # e.g., 100.108.120.69

# Access from phone/other device:
#   http://<tailscale-ip>:8765/sfl_dashboard.html
#   e.g., http://100.108.120.69:8765/sfl_dashboard.html

# 4. Stop server when done
kill <PID>
```

Key points:
- `--bind 0.0.0.0` is required (default `127.0.0.1` blocks external access)
- Use `nohup ... &` so the server survives after the shell exits
- Port 8765 is arbitrary — any unused port works
- Only accessible within the Tailscale VPN network (secure)

### Typical Experiment Recipes

**Compare all 5 methods on CIFAR-10 Non-IID:**
```bash
# 1. Set common.json: dataset=cifar10, alpha=0.3, rounds=100
# 2. Deploy all methods across 2 servers:
./deploy.sh run sfl@xsailor4:0 usfl@xsailor4:1 scaffold@xsailor4:2 \
               gas@xsailor5:0 multisfl@xsailor5:1
```

**Re-run a crashed experiment:**
```bash
# Check which experiments crashed
./deploy.sh check cifar10_a0.3_r100_20260207_222446

# Re-run only the crashed method, reusing the same run_name
./deploy.sh run --run-name cifar10_a0.3_r100_20260207_222446 gas@xsailor5:0
```

**Run locally without GPU servers (single method):**
```bash
# Generate spec manually
python -m experiment_core.generate_spec \
    --config-dir experiment_configs \
    --methods usfl \
    --gpu-map '{"usfl": 0}' \
    --output-dir results/local_test \
    --output /tmp/local_spec.json

# Run batch_runner directly
python -m experiment_core.batch_runner --spec /tmp/local_spec.json --repo-root .
```

### File Reference

| File | Purpose |
|------|---------|
| `deploy.sh` | Main CLI tool (run, status, check, kill, logs, attach, collect, servers) |
| `deploy/deploy_servers.json` | Server inventory (SSH, repo paths, conda env, GPUs) |
| `deploy/.deploy_history.json` | Auto-generated deployment history (not in git) |
| `deploy/remote_run.sh` | Runs inside tmux on GPU server (conda activate → batch_runner) |
| `deploy/setup_rclone_gdrive.sh` | One-time rclone Google Drive setup for servers |
| `experiment_core/generate_spec.py` | Config JSONs → batch_spec.json |
| `experiment_core/batch_runner.py` | Runs multiple experiments from a spec file |
| `experiment_core/runner.py` | Runs a single experiment via adapter |
| `experiment_core/adapters/` | Translates unified config → framework-native format |
| `sfl_dashboard.py` | Result visualization (HTML dashboard generator) |

## Running Experiments Locally

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
