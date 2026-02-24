#!/bin/bash
set -euo pipefail
###############################################################################
# remote_run.sh — GPU 서버에서 tmux 안에서 실행되는 실험 래퍼
#
# Usage:
#   bash remote_run.sh <env_type> <env_name> <spec_path>
#   bash remote_run.sh <env_type> <env_name> --interactive
#
# env_type: "conda" or "uv"
# env_name: conda environment name (for conda), ignored (for uv)
#
# deploy.sh가 SSH를 통해 이 스크립트를 실행합니다.
# 이 파일은 레포에 포함되어 git pull로 동기화됩니다.
###############################################################################

ENV_TYPE="${1:?Usage: remote_run.sh <env_type> <env_name> <spec_path|--interactive>}"
ENV_NAME="${2?Usage: remote_run.sh <env_type> <env_name> <spec_path|--interactive>}"
MODE="${3:---interactive}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

# Environment activation
case "$ENV_TYPE" in
  conda)
    # Conda activation
    # Note: ~/.bashrc often has "if not interactive, return" guard at the top,
    # so sourcing it from a script won't reach the conda init block.
    # Instead, we directly source conda's init script or find the binary.
    _conda_found=false

    # Method 1: conda already in PATH
    if command -v conda &>/dev/null; then
        _conda_found=true
    fi

    # Method 2: source conda.sh directly (bypasses bashrc interactive guard)
    if [ "$_conda_found" = false ]; then
        for conda_prefix in \
            "$HOME/anaconda3" \
            "$HOME/miniconda3" \
            "$HOME/miniforge3" \
            "/opt/conda" \
            "$HOME/.conda"; do
            if [ -f "$conda_prefix/etc/profile.d/conda.sh" ]; then
                source "$conda_prefix/etc/profile.d/conda.sh"
                _conda_found=true
                break
            fi
        done
    fi

    # Method 3: mamba
    if [ "$_conda_found" = false ] && command -v mamba &>/dev/null; then
        eval "$(mamba shell.bash hook)"
        mamba activate "$ENV_NAME"
        _conda_found=true
    fi

    if [ "$_conda_found" = false ]; then
        echo "Error: conda/mamba not found"
        echo "  Searched: conda in PATH, ~/anaconda3, ~/miniconda3, ~/miniforge3, /opt/conda"
        echo "  PATH: $PATH"
        exit 1
    fi

    # Temporarily disable -u: conda activate scripts may reference unbound variables
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    set -u
    unset _conda_found

    _env_display="conda ($ENV_NAME)"
    ;;

  uv)
    # uv: activate .venv in repo root
    if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
        set +u
        source "$REPO_ROOT/.venv/bin/activate"
        set -u
    else
        echo "Error: .venv not found at $REPO_ROOT/.venv"
        echo "  Run 'uv venv' in the repo root to create it."
        exit 1
    fi

    _env_display="uv (.venv)"
    ;;

  *)
    echo "Error: Unknown env_type: $ENV_TYPE (expected 'conda' or 'uv')"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "  USFL Remote Experiment Runner"
echo "=========================================="
echo "  Host:      $(hostname)"
echo "  Env:       $_env_display"
echo "  Python:    $(python --version 2>&1)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPUs:      $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Repo:      $REPO_ROOT"
echo "  Branch:    $(git branch --show-current 2>/dev/null || echo 'N/A')"
echo "  Started:   $(date)"
echo "=========================================="
echo ""
unset _env_display

START_TIME=$(date +%s)

# Disable Python output buffering so logs stream in real-time through pipes
export PYTHONUNBUFFERED=1

if [ "$MODE" == "--interactive" ]; then
    echo "[Mode] Interactive — environment activated, dropping to shell"
    echo "  Use: python -m experiment_core.batch_runner --spec <spec.json> --repo-root ."
    echo ""
    exec bash
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
