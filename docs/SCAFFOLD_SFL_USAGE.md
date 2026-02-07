# SCAFFOLD-SFL Usage Guide

## Overview

SCAFFOLD-SFL integrates the SCAFFOLD algorithm (Stochastic Controlled Averaging for Federated Learning) into SplitFed v2.

**Key Features:**
- Control variates applied to **client-side model only** (SplitFed v2 architecture)
- Server-side model unchanged (standard SplitFed v2 behavior)
- Variance reduction through gradient correction
- No additional hyperparameters needed (uses existing learning rate)

## Quick Start

### Method 1: Edit simulation.py

Add to `USFL_OPTIONS` in `simulation.py`:

```python
USFL_OPTIONS = {
    "SCAFFOLD_SFL": {
        "method": "scaffold_sfl",
        "dataset": "cifar10",
        "model": "resnet18",
        "split_layer": "layer1.1.bn2",
        "batch_size": "64",
        "labels_per_client": "2",
        "dirichlet_alpha": "0.3",
        "num_clients": "100",
        "num_clients_per_round": "10",
        "global_round": "100",
        "local_epochs": "5",
        "learning_rate": "0.01",
        "optimizer": "sgd",
        "momentum": "0.9",
    },
    # Compare with baseline SFL
    "SFL_BASELINE": {
        "method": "sfl",
        "dataset": "cifar10",
        "model": "resnet18",
        "split_layer": "layer1.1.bn2",
        "batch_size": "64",
        "labels_per_client": "2",
        "dirichlet_alpha": "0.3",
        "num_clients": "100",
        "num_clients_per_round": "10",
        "global_round": "100",
        "local_epochs": "5",
        "learning_rate": "0.01",
        "optimizer": "sgd",
        "momentum": "0.9",
    },
}
```

Then run:
```bash
cd sfl_framework-fork-feature-training-tracker
python simulation.py
```

### Method 2: Direct Command Line (Emulation Mode)

**Server:**
```bash
cd sfl_framework-fork-feature-training-tracker/server
python main.py \
    -M scaffold_sfl \
    -d cifar10 \
    -m resnet18 \
    -sl layer1.1.bn2 \
    -nc 100 \
    -ncpr 10 \
    -gr 100 \
    -le 5 \
    -bs 64 \
    -lpc 2 \
    -diri-alpha 0.3 \
    -distr shard_dirichlet \
    -lr 0.01 \
    -opt sgd \
    -mom 0.9
```

**Client (repeat for each client):**
```bash
cd sfl_framework-fork-feature-training-tracker/client
python main.py -cid 0
python main.py -cid 1
# ... up to -cid 99
```

## Algorithm Details

### SCAFFOLD Option II (Client-Side Only)

**Client Training Loop:**
1. Receive global model θ and global control variate c from server
2. Initialize c_i (client control variate) to zeros if first round
3. Snapshot initial parameters: θ₀ = θ
4. For each local step k = 1 to K:
   - Compute gradient g via SplitFed v2 (activation/gradient exchange)
   - Apply correction: g ← g - c_i + c
   - Update: θ ← θ - η·g
5. Compute update: Δθ = θ_K - θ₀
6. Compute control variate update: c_i^new = c_i - c - Δθ/(K·η)
7. Send model θ and Δc = c_i^new - c_i to server

**Server Aggregation:**
1. Aggregate client models: θ ← Σ (w_i · θ_i)
2. Aggregate control variates: c ← c + (N/m) · Σ (w_i · Δc_i)
   - N = total clients, m = participating clients
   - Accounts for partial client participation
3. Broadcast updated θ and c to selected clients next round

### Why Client-Side Only?

In SplitFed v2:
- Client model is federated (aggregated across clients)
- Server model is global (single instance, not aggregated)

SCAFFOLD variance reduction is needed for the federated part (client model), not the global part (server model).

## Comparison with Baselines

### vs. SFL (SplitFed v2)
- **Same architecture**: Client/server split, activation/gradient exchange
- **Different optimization**: SCAFFOLD adds gradient correction for variance reduction
- **Expected improvement**: Better convergence in high Non-IID settings

### vs. FedAvg + SCAFFOLD
- **Different architecture**: SFL uses split learning (no local labels), SCAFFOLD-FL uses full local training
- **Same algorithm**: Control variate mechanism identical
- **Different application**: Client-side only in SCAFFOLD-SFL, both sides in SCAFFOLD-FL

## Configuration Options

All standard SFL options apply. No additional SCAFFOLD-specific parameters needed.

**Key Parameters:**
- `-M scaffold_sfl` or `"method": "scaffold_sfl"`: Enable SCAFFOLD-SFL
- `-lr`: Learning rate (used in control variate computation)
- `-le`: Local epochs (affects control variate update)
- All other params same as SFL

## Expected Behavior

**Console Output:**
```
[SCAFFOLD Server] Initialized global control variate c with 266 parameters
[SCAFFOLD Client 0] Received global control variate c
[SCAFFOLD Client 0] Computed delta_c after 50 steps
[SCAFFOLD Client 0] Submitting delta_c
[SCAFFOLD Server] Received delta_c from client 0 (weight=5000)
[SCAFFOLD Server] Updated global control variate c from 10 clients
```

**Performance Expectations:**
- Slower initial convergence (control variates need warm-up)
- Better final accuracy in Non-IID settings
- More stable training (reduced variance)
- ~10-20% communication overhead (sending delta_c)

## Troubleshooting

**"Invalid method: scaffold_sfl"**
- Check that routing files were properly updated
- Verify imports in `model_trainer.py` and `stage_organizer.py`

**No delta_c in logs**
- Ensure clients complete training (not timeout)
- Check that local_step_count > 0

**Poor performance**
- SCAFFOLD needs multiple rounds to warm up control variates
- Try increasing global_round to 200+
- Ensure learning rate is appropriate (0.01 typical for SGD)

## Implementation Files

**Client-side:**
- `client/modules/trainer/model_trainer/scaffold_sfl_model_trainer.py`: Training loop with gradient correction
- `client/modules/trainer/stage/scaffold_sfl_stage_organizer.py`: Client orchestration with delta_c submission

**Server-side:**
- `server/modules/trainer/stage/scaffold_sfl_stage_organizer.py`: Server orchestration with control variate management

**Routing:**
- `client/modules/trainer/model_trainer/model_trainer.py`: Added scaffold_sfl case
- `client/modules/trainer/stage/stage_organizer.py`: Added scaffold_sfl case
- `server/modules/trainer/stage/stage_organizer.py`: Added scaffold_sfl case

## References

- Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (ICML 2020)
- SplitFed v2: "SplitFed: When Federated Learning Meets Split Learning" (AAAI 2022)
