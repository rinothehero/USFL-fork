#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "======================================================================"
echo "  MultiSFL Sequential Experiments (Matched with GAS-fork)"
echo "  Target GPU: 1"
echo "======================================================================"

echo "======================================================================"
# ==============================================================================
# 1. ResNet-18 Light (Split at Layer 1) - CIFAR10
# Matches GAS: split_layer='layer1.1.bn2'
# ==============================================================================


echo ">>> [2/6] ResNet Light / CIFAR10 / Strong Non-IID (Shards=2, Alpha=0.3)"
python run_multisfl.py \
    --alpha_master_pull 0.1 \
    --gamma 0.5 \
    --num_clients 100 \
    --n_main 10 \
    --branches 10 \
    --p0 0.01 \
    --p_update paper \
    --replay_budget_mode local_dataset \
    --local_steps 5 \
    --batch_size 250 \
    --lr_client 0.001 \
    --lr_server 0.001 \
    --rounds 300 \
    --model_type resnet18_image_style \
    --split_layer layer1.1.bn2 \
    --dataset cifar10 \
    --partition shard_dirichlet \
    --shards 2 \
    --alpha_dirichlet 0.3 \
    --momentum 0.0 \
    --enable_g_measurement true \
    --g_measure_frequency 10 \
    --use_sfl_transform true \