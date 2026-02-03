#!/bin/bash
#===============================================================================
# GAS Drift Measurement Experiment Script
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
echo "GAS Drift Measurement Experiment"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "=========================================="

#-------------------------------------------------------------------------------
# 1. 모델 & 데이터셋 설정
#-------------------------------------------------------------------------------
export GAS_DATASET="cifar10"
export GAS_MODEL="resnet18"
export GAS_USE_RESNET_IMAGE_STYLE="true"  # torchvision ResNet18 stem 스타일
export GAS_SPLIT_LAYER="layer1.1.bn2"     # SFL Framework와 동일한 split point

#-------------------------------------------------------------------------------
# 2. 훈련 하이퍼파라미터 (SFL Framework BASE_CONFIG와 동일)
#-------------------------------------------------------------------------------
export GAS_GLOBAL_EPOCHS="300"
export GAS_LOCAL_EPOCHS="5"
export GAS_BATCH_SIZE="50"
export GAS_LR="0.001"
export GAS_MOMENTUM="0.0"

#-------------------------------------------------------------------------------
# 3. 클라이언트 설정
#-------------------------------------------------------------------------------
export GAS_TOTAL_CLIENTS="100"
export GAS_CLIENTS_PER_ROUND="10"

#-------------------------------------------------------------------------------
# 4. 데이터 분포 설정 (shard_dirichlet 동등)
#-------------------------------------------------------------------------------
# GAS는 내부적으로 label_dirichlet=True 사용 (shard + dirichlet 조합)
export GAS_LABELS_PER_CLIENT="2"      # shard 수 = labels_per_client
export GAS_DIRICHLET_ALPHA="0.3"
export GAS_MIN_REQUIRE_SIZE="10"

#-------------------------------------------------------------------------------
# 5. Full Epoch 모드 (공정 비교 필수)
#-------------------------------------------------------------------------------
# SFL Framework는 기본적으로 전체 데이터 사용
# GAS도 동일하게 전체 데이터 사용하도록 설정
export GAS_USE_FULL_EPOCHS="true"

#-------------------------------------------------------------------------------
# 6. Drift Measurement 설정
#-------------------------------------------------------------------------------
export GAS_DRIFT_MEASUREMENT="true"
export GAS_DRIFT_SAMPLE_INTERVAL="1"  # 매 스텝마다 측정 (정확도 우선)

#-------------------------------------------------------------------------------
# 7. 재현성을 위한 시드 (SFL Framework와 동일)
#-------------------------------------------------------------------------------
export GAS_SEED="42"

#-------------------------------------------------------------------------------
# 8. G Measurement 비활성화 (Drift만 측정)
#-------------------------------------------------------------------------------
# GAS_main.py 내부에서 G_Measurement=True가 기본값
# 필요시 코드 수정 또는 여기서 비활성화 불가 (코드 내 하드코딩)
# → Drift 측정에는 영향 없음

#-------------------------------------------------------------------------------
# 실행
#-------------------------------------------------------------------------------
echo ""
echo "Configuration:"
echo "  Dataset: ${GAS_DATASET}"
echo "  Model: ${GAS_MODEL} (ImageNet style: ${GAS_USE_RESNET_IMAGE_STYLE})"
echo "  Split Layer: ${GAS_SPLIT_LAYER}"
echo "  Global Epochs: ${GAS_GLOBAL_EPOCHS}"
echo "  Local Epochs: ${GAS_LOCAL_EPOCHS}"
echo "  Batch Size: ${GAS_BATCH_SIZE}"
echo "  LR: ${GAS_LR}, Momentum: ${GAS_MOMENTUM}"
echo "  Clients: ${GAS_TOTAL_CLIENTS} total, ${GAS_CLIENTS_PER_ROUND} per round"
echo "  Data Distribution: shard_dirichlet (shards=${GAS_LABELS_PER_CLIENT}, alpha=${GAS_DIRICHLET_ALPHA})"
echo "  Full Epochs: ${GAS_USE_FULL_EPOCHS}"
echo "  Drift Measurement: ${GAS_DRIFT_MEASUREMENT} (interval=${GAS_DRIFT_SAMPLE_INTERVAL})"
echo "  Seed: ${GAS_SEED}"
echo ""

cd "$(dirname "$0")/GAS_implementation"

# 결과 디렉토리 생성
mkdir -p results

echo "Starting GAS training..."
python GAS_main.py

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved in: GAS_implementation/results/"
echo "=========================================="
