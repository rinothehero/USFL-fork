#!/bin/bash
#===============================================================================
# MultiSFL Drift Measurement Experiment Script
#
# SFL Framework 기법들과 1:1 비교를 위한 설정
# - ResNet18 (ImageNet style) + CIFAR10
# - Split layer: layer1.1.bn2
# - Shard-Dirichlet 데이터 분포
#===============================================================================

set -e

# 실험 이름 (결과 파일 구분용)
EXPERIMENT_NAME="${1:-default}"
echo "=========================================="
echo "MultiSFL Drift Measurement Experiment"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "=========================================="

#-------------------------------------------------------------------------------
# 설정 변수
#-------------------------------------------------------------------------------

# 1. 모델 & 데이터셋
DATASET="cifar10"
MODEL_TYPE="resnet18_image_style"  # torchvision ResNet18 stem 스타일
SPLIT_LAYER="layer1.1.bn2"         # SFL Framework와 동일한 split point
NUM_CLASSES="10"

# 2. 훈련 하이퍼파라미터 (SFL Framework BASE_CONFIG와 동일)
ROUNDS="300"
LOCAL_STEPS="5"                    # local_epochs에 해당
BATCH_SIZE="50"
LR_CLIENT="0.001"
LR_SERVER="0.001"
MOMENTUM="0.0"

# 3. 클라이언트 설정
NUM_CLIENTS="100"
N_MAIN="10"                        # clients_per_round에 해당
BRANCHES="10"                      # n_main과 동일하게 설정 (기본 동작)

# 4. 데이터 분포 설정 (shard_dirichlet)
PARTITION="shard_dirichlet"
SHARDS="2"                         # labels_per_client
ALPHA_DIRICHLET="0.3"
MIN_SAMPLES_PER_CLIENT="10"

# 5. Full Epoch 모드 (공정 비교 필수)
USE_FULL_EPOCHS="--use-full-epochs"

# 6. Drift Measurement 설정
ENABLE_DRIFT_MEASUREMENT="true"
DRIFT_SAMPLE_INTERVAL="1"          # 매 스텝마다 측정

# 7. 재현성을 위한 시드
SEED="42"

# 8. 기타 설정
DEVICE="cuda"
DATA_ROOT="./data"

# MultiSFL 고유 설정 (기본값 사용)
ALPHA_MASTER_PULL="0.1"            # Main server pull weight
P0="0.01"
P_MIN="0.01"
P_MAX="0.5"
P_UPDATE="abs_ratio"
GAMMA="0.5"
DELTA_CLIP="0.2"

# G Measurement 비활성화 (Drift만 측정)
ENABLE_G_MEASUREMENT="false"

# Torchvision 초기화 사용 (ImageNet style과 함께)
USE_TORCHVISION_INIT="true"

# SFL 스타일 transform (공정 비교용)
USE_SFL_TRANSFORM="true"

#-------------------------------------------------------------------------------
# 실행
#-------------------------------------------------------------------------------
echo ""
echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Model: ${MODEL_TYPE}"
echo "  Split Layer: ${SPLIT_LAYER}"
echo "  Rounds: ${ROUNDS}"
echo "  Local Steps: ${LOCAL_STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  LR: client=${LR_CLIENT}, server=${LR_SERVER}, Momentum: ${MOMENTUM}"
echo "  Clients: ${NUM_CLIENTS} total, ${N_MAIN} per round, ${BRANCHES} branches"
echo "  Data Distribution: ${PARTITION} (shards=${SHARDS}, alpha=${ALPHA_DIRICHLET})"
echo "  Full Epochs: enabled"
echo "  Drift Measurement: ${ENABLE_DRIFT_MEASUREMENT} (interval=${DRIFT_SAMPLE_INTERVAL})"
echo "  Torchvision Init: ${USE_TORCHVISION_INIT}"
echo "  SFL Transform: ${USE_SFL_TRANSFORM}"
echo "  Seed: ${SEED}"
echo ""

cd "$(dirname "$0")/multisfl_implementation"

# 결과 디렉토리 생성
mkdir -p results

echo "Starting MultiSFL training..."
python run_multisfl.py \
    --dataset "${DATASET}" \
    --model_type "${MODEL_TYPE}" \
    --split_layer "${SPLIT_LAYER}" \
    --num_classes "${NUM_CLASSES}" \
    \
    --rounds "${ROUNDS}" \
    --local_steps "${LOCAL_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr_client "${LR_CLIENT}" \
    --lr_server "${LR_SERVER}" \
    --momentum "${MOMENTUM}" \
    \
    --num_clients "${NUM_CLIENTS}" \
    --n_main "${N_MAIN}" \
    --branches "${BRANCHES}" \
    \
    --partition "${PARTITION}" \
    --shards "${SHARDS}" \
    --alpha_dirichlet "${ALPHA_DIRICHLET}" \
    --min_samples_per_client "${MIN_SAMPLES_PER_CLIENT}" \
    \
    ${USE_FULL_EPOCHS} \
    \
    --enable_drift_measurement "${ENABLE_DRIFT_MEASUREMENT}" \
    --drift_sample_interval "${DRIFT_SAMPLE_INTERVAL}" \
    \
    --enable_g_measurement "${ENABLE_G_MEASUREMENT}" \
    --use_torchvision_init "${USE_TORCHVISION_INIT}" \
    --use_sfl_transform "${USE_SFL_TRANSFORM}" \
    \
    --alpha_master_pull "${ALPHA_MASTER_PULL}" \
    --p0 "${P0}" \
    --p_min "${P_MIN}" \
    --p_max "${P_MAX}" \
    --p_update "${P_UPDATE}" \
    --gamma "${GAMMA}" \
    --delta_clip "${DELTA_CLIP}" \
    \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --data_root "${DATA_ROOT}"

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved in: multisfl_implementation/results/"
echo "=========================================="
