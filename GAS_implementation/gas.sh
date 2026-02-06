#!/bin/bash
# GAS Experiment Runner
# Usage: ./run_gas.sh [--use-full-epochs]

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Dataset: cifar10, mnist, fmnist, cinic, cifar100, svhn
export GAS_DATASET="cifar10"

# Model: resnet18, alexnet
export GAS_MODEL="resnet18"

# ResNet style: true=ImageNet(7x7), false=CIFAR(3x3)
# For fair comparison with SFL framework, use false for CIFAR10
export GAS_USE_RESNET_IMAGE_STYLE="false"

# Split layer: layer1, layer2, layer3, layer4, or fine-grained like layer1.1.bn2
export GAS_SPLIT_LAYER="layer2"

# ============================================================
# DATA DISTRIBUTION (Non-IID Settings)
# ============================================================

# Labels per client (shard)
export GAS_LABELS_PER_CLIENT="2"

# Dirichlet alpha (lower = more heterogeneous)
export GAS_DIRICHLET_ALPHA="0.3"

# Minimum samples per client
export GAS_MIN_REQUIRE_SIZE="10"

# ============================================================
# TRAINING PARAMETERS
# ============================================================

# Number of global rounds
export GAS_GLOBAL_EPOCHS="300"

# Local epochs per round
export GAS_LOCAL_EPOCHS="5"

# Total number of clients
export GAS_TOTAL_CLIENTS="100"

# Clients per round
export GAS_CLIENTS_PER_ROUND="10"

# Batch size
export GAS_BATCH_SIZE="50"

# Learning rate
export GAS_LR="0.001"

# Momentum
export GAS_MOMENTUM="0.0"

# Random seed
export GAS_SEED="42"

# Use full epochs (iterate entire dataset) instead of fixed local epochs
# Set to "true" to match SFL framework behavior
export GAS_USE_FULL_EPOCHS="false"

# ============================================================
# GPU SETTINGS
# ============================================================

# CUDA device (comma-separated for multi-GPU)
export CUDA_VISIBLE_DEVICES="0"

# ============================================================
# RUN EXPERIMENT
# ============================================================

echo "============================================================"
echo "GAS Experiment Configuration"
echo "============================================================"
echo "Dataset:          $GAS_DATASET"
echo "Model:            $GAS_MODEL"
echo "Image Style:      $GAS_USE_RESNET_IMAGE_STYLE"
echo "Split Layer:      $GAS_SPLIT_LAYER"
echo "Labels/Client:    $GAS_LABELS_PER_CLIENT"
echo "Dirichlet Alpha:  $GAS_DIRICHLET_ALPHA"
echo "Global Epochs:    $GAS_GLOBAL_EPOCHS"
echo "Local Epochs:     $GAS_LOCAL_EPOCHS"
echo "Total Clients:    $GAS_TOTAL_CLIENTS"
echo "Clients/Round:    $GAS_CLIENTS_PER_ROUND"
echo "Batch Size:       $GAS_BATCH_SIZE"
echo "Learning Rate:    $GAS_LR"
echo "Momentum:         $GAS_MOMENTUM"
echo "Seed:             $GAS_SEED"
echo "Use Full Epochs:  $GAS_USE_FULL_EPOCHS"
echo "CUDA Devices:     $CUDA_VISIBLE_DEVICES"
echo "============================================================"

cd "$(dirname "$0")"

# Pass through any additional arguments (e.g., --use-full-epochs)
python GAS_main.py "$@"

