#!/bin/bash
set -euo pipefail
###############################################################################
# remote_run.sh — GPU 서버에서 tmux 안에서 실행되는 실험 래퍼
#
# Usage:
#   bash remote_run.sh <conda_env> <spec_path>
#   bash remote_run.sh <conda_env> --interactive
#
# deploy.sh가 SSH를 통해 이 스크립트를 실행합니다.
# 이 파일은 레포에 포함되어 git pull로 동기화됩니다.
###############################################################################

CONDA_ENV="${1:?Usage: remote_run.sh <conda_env> <spec_path|--interactive>}"
MODE="${2:---interactive}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$REPO_ROOT"

# Source shell profile to get conda in PATH (tmux non-login shells skip this)
for rc in "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile"; do
    if [ -f "$rc" ]; then
        source "$rc" 2>/dev/null || true
        break
    fi
done

# Conda activation (supports both conda and mamba)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif command -v mamba &>/dev/null; then
    eval "$(mamba shell.bash hook)"
    mamba activate "$CONDA_ENV"
else
    echo "Error: conda/mamba not found"
    echo "  Searched: conda, mamba"
    echo "  Sourced:  ~/.bash_profile, ~/.bashrc, ~/.profile"
    echo "  PATH:     $PATH"
    exit 1
fi

echo ""
echo "=========================================="
echo "  USFL Remote Experiment Runner"
echo "=========================================="
echo "  Host:      $(hostname)"
echo "  Conda:     $CONDA_ENV"
echo "  Python:    $(python --version 2>&1)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPUs:      $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Repo:      $REPO_ROOT"
echo "  Branch:    $(git branch --show-current 2>/dev/null || echo 'N/A')"
echo "  Started:   $(date)"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

if [ "$MODE" == "--interactive" ]; then
    echo "[Mode] Interactive — launching run_experiments.sh"
    echo ""
    ./run_experiments.sh
else
    SPEC_PATH="$MODE"
    if [ ! -f "$SPEC_PATH" ]; then
        echo "Error: Spec file not found: $SPEC_PATH"
        exit 1
    fi
    echo "[Mode] Batch — spec: $SPEC_PATH"
    echo ""
    python -m experiment_core.batch_runner \
        --spec "$SPEC_PATH" \
        --repo-root "$REPO_ROOT"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS_REMAINING=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "  Experiment Completed"
echo "=========================================="
echo "  Finished:  $(date)"
echo "  Elapsed:   ${HOURS}h ${MINUTES}m ${SECONDS_REMAINING}s"
echo "=========================================="
