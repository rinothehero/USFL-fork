#!/bin/bash
# ============================================================
# Unified Experiment Runner - 7 Methods Comparison
# ============================================================
# Methods:
#   1. SFL (baseline)
#   2. FedCBS
#   3. Mix2SFL
#   4. SCAFFOLD
#   5. USFL
#   6. GAS
#   7. MultiSFL
# ============================================================

set -e  # Exit on error

# ============================================================
# COMMON CONFIGURATION
# ============================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="cifar10"
MODEL="resnet18_flex"
SPLIT_LAYER="layer2"

# Data distribution
DISTRIBUTER="shard_dirichlet"
LABELS_PER_CLIENT=2
DIRICHLET_ALPHA=0.3
MIN_REQUIRE_SIZE=10

# Training parameters
GLOBAL_ROUNDS=300
LOCAL_EPOCHS=5
TOTAL_CLIENTS=100
CLIENTS_PER_ROUND=10
BATCH_SIZE=50
LEARNING_RATE=0.001
MOMENTUM=0.0
SEED=42

# Directories
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SFL_DIR="$ROOT_DIR/sfl_framework-fork-feature-training-tracker"
GAS_DIR="$ROOT_DIR/GAS_implementation"
MULTISFL_DIR="$ROOT_DIR/multisfl_implementation"

# Results directory
RESULTS_DIR="$ROOT_DIR/experiment_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ============================================================
# METHOD SELECTION
# ============================================================
# Set to "true" to run, "false" to skip
RUN_SFL="${RUN_SFL:-true}"
RUN_FEDCBS="${RUN_FEDCBS:-true}"
RUN_MIX2SFL="${RUN_MIX2SFL:-true}"
RUN_SCAFFOLD="${RUN_SCAFFOLD:-true}"
RUN_USFL="${RUN_USFL:-true}"
RUN_GAS="${RUN_GAS:-true}"
RUN_MULTISFL="${RUN_MULTISFL:-true}"

# ============================================================
# LOGGING
# ============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/experiment.log"
}

print_config() {
    log "============================================================"
    log "EXPERIMENT CONFIGURATION"
    log "============================================================"
    log "Dataset:          $DATASET"
    log "Model:            $MODEL"
    log "Split Layer:      $SPLIT_LAYER"
    log "Distribution:     $DISTRIBUTER (shards=$LABELS_PER_CLIENT, alpha=$DIRICHLET_ALPHA)"
    log "Global Rounds:    $GLOBAL_ROUNDS"
    log "Local Epochs:     $LOCAL_EPOCHS"
    log "Clients:          $TOTAL_CLIENTS (per round: $CLIENTS_PER_ROUND)"
    log "Batch Size:       $BATCH_SIZE"
    log "Learning Rate:    $LEARNING_RATE"
    log "Momentum:         $MOMENTUM"
    log "Seed:             $SEED"
    log "Results Dir:      $RESULTS_DIR"
    log "============================================================"
    log ""
    log "Methods to run:"
    log "  [1] SFL:       $RUN_SFL"
    log "  [2] FedCBS:    $RUN_FEDCBS"
    log "  [3] Mix2SFL:   $RUN_MIX2SFL"
    log "  [4] SCAFFOLD:  $RUN_SCAFFOLD"
    log "  [5] USFL:      $RUN_USFL"
    log "  [6] GAS:       $RUN_GAS"
    log "  [7] MultiSFL:  $RUN_MULTISFL"
    log "============================================================"
}

# ============================================================
# SFL FRAMEWORK RUNNER (Methods 1-5)
# ============================================================
run_sfl_framework() {
    local METHOD_NAME="$1"
    local METHOD="$2"
    local SELECTOR="$3"
    local AGGREGATOR="${4:-fedavg}"
    local EXTRA_ARGS="${5:-}"

    log ">>> Starting $METHOD_NAME (method=$METHOD, selector=$SELECTOR)"

    cd "$SFL_DIR"

    # Build command
    CMD="python -c \"
import asyncio
import sys
sys.path.insert(0, 'client')
sys.path.insert(0, 'server')

from client.client_args import parse_args as c_parse_args
from client.modules.ws.inmemory_connection import InMemoryConnection as InMemoryClientConnection
from client.simulation_client import SimulationClient
from server.modules.global_dict.global_dict import GlobalDict as GlobalDictServer
from server.modules.ws.inmemory_connection import InMemoryConnection as InMemoryServerConnection
from server.server_args import parse_args as s_parse_args
from server.simulation_server import SimulationServer

async def run():
    server_args = s_parse_args([
        '-d', '$DATASET',
        '-m', '$MODEL',
        '-M', '$METHOD',
        '-s', '$SELECTOR',
        '-aggr', '$AGGREGATOR',
        '-le', '$LOCAL_EPOCHS',
        '-gr', '$GLOBAL_ROUNDS',
        '-de', 'cuda',
        '-nc', '$TOTAL_CLIENTS',
        '-ncpr', '$CLIENTS_PER_ROUND',
        '-distr', '$DISTRIBUTER',
        '-lpc', '$LABELS_PER_CLIENT',
        '-diri-alpha', '$DIRICHLET_ALPHA',
        '-mrs', '$MIN_REQUIRE_SIZE',
        '-ss', 'layer_name',
        '-sl', '$SPLIT_LAYER',
        '-lr', '$LEARNING_RATE',
        '-mt', '$MOMENTUM',
        '-bs', '$BATCH_SIZE',
        '-p', '3000',
        '--seed', '$SEED',
        $EXTRA_ARGS
    ])

    client_args_list = [
        c_parse_args(['-cid', str(i), '-d', f'cuda:{i % 1}', '-su', 'localhost:3000'])
        for i in range($TOTAL_CLIENTS)
    ]

    server_global_dict = GlobalDictServer(server_args)
    server_conn = InMemoryServerConnection(server_args, server_global_dict)
    client_conns = [InMemoryClientConnection(args, server_conn) for args in client_args_list]

    server = SimulationServer(server_args, server_conn)
    clients = [SimulationClient(args, client_conns[i]) for i, args in enumerate(client_args_list)]

    await asyncio.gather(server.run(), *[c.run() for c in clients])

asyncio.run(run())
\" 2>&1 | tee '$RESULTS_DIR/${METHOD_NAME}.log'"

    eval $CMD

    log ">>> Completed $METHOD_NAME"
}

# ============================================================
# GAS RUNNER (Method 6)
# ============================================================
run_gas() {
    log ">>> Starting GAS"

    cd "$GAS_DIR"

    export GAS_DATASET="$DATASET"
    export GAS_MODEL="resnet18"
    export GAS_USE_RESNET_IMAGE_STYLE="false"
    export GAS_SPLIT_LAYER="$SPLIT_LAYER"
    export GAS_LABELS_PER_CLIENT="$LABELS_PER_CLIENT"
    export GAS_DIRICHLET_ALPHA="$DIRICHLET_ALPHA"
    export GAS_MIN_REQUIRE_SIZE="$MIN_REQUIRE_SIZE"
    export GAS_GLOBAL_EPOCHS="$GLOBAL_ROUNDS"
    export GAS_LOCAL_EPOCHS="$LOCAL_EPOCHS"
    export GAS_TOTAL_CLIENTS="$TOTAL_CLIENTS"
    export GAS_CLIENTS_PER_ROUND="$CLIENTS_PER_ROUND"
    export GAS_BATCH_SIZE="$BATCH_SIZE"
    export GAS_LR="$LEARNING_RATE"
    export GAS_MOMENTUM="$MOMENTUM"
    export GAS_SEED="$SEED"

    python GAS_main.py 2>&1 | tee "$RESULTS_DIR/GAS.log"

    log ">>> Completed GAS"
}

# ============================================================
# MULTISFL RUNNER (Method 7)
# ============================================================
run_multisfl() {
    log ">>> Starting MultiSFL"

    cd "$MULTISFL_DIR"

    python run_multisfl.py \
        --dataset "$DATASET" \
        --model_type "resnet18_flex" \
        --split_layer "$SPLIT_LAYER" \
        --partition "$DISTRIBUTER" \
        --shards "$LABELS_PER_CLIENT" \
        --alpha_dirichlet "$DIRICHLET_ALPHA" \
        --min_samples_per_client "$MIN_REQUIRE_SIZE" \
        --rounds "$GLOBAL_ROUNDS" \
        --num_clients "$TOTAL_CLIENTS" \
        --n_main "$CLIENTS_PER_ROUND" \
        --branches "$CLIENTS_PER_ROUND" \
        --batch_size "$BATCH_SIZE" \
        --local_steps "$LOCAL_EPOCHS" \
        --lr_client "$LEARNING_RATE" \
        --lr_server "$LEARNING_RATE" \
        --momentum "$MOMENTUM" \
        --seed "$SEED" \
        --alpha_master_pull 0.1 \
        --gamma 0.5 \
        --p0 0.01 \
        --p_update "paper" \
        --replay_budget_mode "local_dataset" \
        --enable_g_measurement "true" \
        --use_sfl_transform "true" \
        2>&1 | tee "$RESULTS_DIR/MultiSFL.log"

    log ">>> Completed MultiSFL"
}

# ============================================================
# MAIN EXECUTION
# ============================================================
main() {
    print_config

    START_TIME=$(date +%s)

    # Method 1: SFL (baseline)
    if [ "$RUN_SFL" = "true" ]; then
        run_sfl_framework "SFL" "sfl" "uniform"
    fi

    # Method 2: FedCBS
    if [ "$RUN_FEDCBS" = "true" ]; then
        run_sfl_framework "FedCBS" "sfl" "fedcbs"
    fi

    # Method 3: Mix2SFL
    if [ "$RUN_MIX2SFL" = "true" ]; then
        run_sfl_framework "Mix2SFL" "mix2sfl" "uniform"
    fi

    # Method 4: SCAFFOLD
    if [ "$RUN_SCAFFOLD" = "true" ]; then
        run_sfl_framework "SCAFFOLD" "scaffold_sfl" "uniform"
    fi

    # Method 5: USFL
    if [ "$RUN_USFL" = "true" ]; then
        run_sfl_framework "USFL" "usfl" "usfl" "usfl" "'-bstrat', 'target', '-btarget', 'mean', '-gs', '-gss', 'random'"
    fi

    # Method 6: GAS
    if [ "$RUN_GAS" = "true" ]; then
        run_gas
    fi

    # Method 7: MultiSFL
    if [ "$RUN_MULTISFL" = "true" ]; then
        run_multisfl
    fi

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    log ""
    log "============================================================"
    log "ALL EXPERIMENTS COMPLETED"
    log "Total time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
    log "Results saved to: $RESULTS_DIR"
    log "============================================================"
}

# ============================================================
# USAGE
# ============================================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --only METHOD    Run only specified method (sfl|fedcbs|mix2sfl|scaffold|usfl|gas|multisfl)"
    echo "  --skip METHOD    Skip specified method"
    echo "  --rounds N       Set global rounds (default: 300)"
    echo "  --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all 7 methods"
    echo "  $0 --only sfl                # Run only SFL"
    echo "  $0 --skip gas --skip multisfl  # Skip GAS and MultiSFL"
    echo "  $0 --rounds 100              # Run with 100 rounds"
    echo ""
    echo "Environment variables:"
    echo "  CUDA_VISIBLE_DEVICES=0,1     # Set GPU devices"
    echo "  RUN_SFL=false                # Disable specific method"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --only)
            # Disable all, then enable specified
            RUN_SFL="false"; RUN_FEDCBS="false"; RUN_MIX2SFL="false"
            RUN_SCAFFOLD="false"; RUN_USFL="false"; RUN_GAS="false"; RUN_MULTISFL="false"
            case $2 in
                sfl) RUN_SFL="true" ;;
                fedcbs) RUN_FEDCBS="true" ;;
                mix2sfl) RUN_MIX2SFL="true" ;;
                scaffold) RUN_SCAFFOLD="true" ;;
                usfl) RUN_USFL="true" ;;
                gas) RUN_GAS="true" ;;
                multisfl) RUN_MULTISFL="true" ;;
                *) echo "Unknown method: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --skip)
            case $2 in
                sfl) RUN_SFL="false" ;;
                fedcbs) RUN_FEDCBS="false" ;;
                mix2sfl) RUN_MIX2SFL="false" ;;
                scaffold) RUN_SCAFFOLD="false" ;;
                usfl) RUN_USFL="false" ;;
                gas) RUN_GAS="false" ;;
                multisfl) RUN_MULTISFL="false" ;;
                *) echo "Unknown method: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --rounds)
            GLOBAL_ROUNDS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

main
