#!/bin/bash
# MultiSFL Experiment Runner - Layer2 Split (Fair Comparison)
# Matched with SFL Framework and GAS settings

export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "MultiSFL Experiment - Layer2 Split (Fair Comparison)"
echo "============================================================"

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Dataset
DATASET="cifar10"

# Model: resnet18_flex = CIFAR-style (3x3 conv, no maxpool)
MODEL_TYPE="resnet18_flex"

# Split layer
SPLIT_LAYER="layer2"

# Data distribution (Non-IID)
PARTITION="shard_dirichlet"
SHARDS=2
ALPHA_DIRICHLET=0.3
MIN_SAMPLES=10

# Training parameters (matched with SFL/GAS)
ROUNDS=300
NUM_CLIENTS=100
N_MAIN=10
BRANCHES=10
BATCH_SIZE=50
LOCAL_STEPS=5
LR_CLIENT=0.001
LR_SERVER=0.001
MOMENTUM=0.0
SEED=42

# MultiSFL-specific parameters
ALPHA_MASTER_PULL=0.1
GAMMA=0.5
P0=0.01
P_UPDATE="paper"
REPLAY_BUDGET_MODE="local_dataset"

# G Measurement
ENABLE_G_MEASUREMENT="true"
G_MEASURE_FREQUENCY=10
USE_VARIANCE_G="false"

# ============================================================
# PRINT CONFIGURATION
# ============================================================

echo "Dataset:          $DATASET"
echo "Model:            $MODEL_TYPE"
echo "Split Layer:      $SPLIT_LAYER"
echo "Partition:        $PARTITION (shards=$SHARDS, alpha=$ALPHA_DIRICHLET)"
echo "Rounds:           $ROUNDS"
echo "Clients:          $NUM_CLIENTS (main=$N_MAIN, branches=$BRANCHES)"
echo "Batch Size:       $BATCH_SIZE"
echo "Local Steps:      $LOCAL_STEPS"
echo "LR:               client=$LR_CLIENT, server=$LR_SERVER"
echo "Momentum:         $MOMENTUM"
echo "Seed:             $SEED"
echo "============================================================"

# ============================================================
# RUN EXPERIMENT
# ============================================================

cd "$(dirname "$0")"

python run_multisfl.py \
    --dataset "$DATASET" \
    --model_type "$MODEL_TYPE" \
    --split_layer "$SPLIT_LAYER" \
    --partition "$PARTITION" \
    --shards "$SHARDS" \
    --alpha_dirichlet "$ALPHA_DIRICHLET" \
    --min_samples_per_client "$MIN_SAMPLES" \
    --rounds "$ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --n_main "$N_MAIN" \
    --branches "$BRANCHES" \
    --batch_size "$BATCH_SIZE" \
    --local_steps "$LOCAL_STEPS" \
    --lr_client "$LR_CLIENT" \
    --lr_server "$LR_SERVER" \
    --momentum "$MOMENTUM" \
    --seed "$SEED" \
    --alpha_master_pull "$ALPHA_MASTER_PULL" \
    --gamma "$GAMMA" \
    --p0 "$P0" \
    --p_update "$P_UPDATE" \
    --replay_budget_mode "$REPLAY_BUDGET_MODE" \
    --enable_g_measurement "$ENABLE_G_MEASUREMENT" \
    --g_measure_frequency "$G_MEASURE_FREQUENCY" \
    --use_variance_g "$USE_VARIANCE_G" \
    --use_sfl_transform "true" \
    "$@"
