#!/bin/bash
set -euo pipefail

###############################################################################
# run_experiments.sh - 대화형 실험 실행 스크립트
#
# 사용법:
#   ./run_experiments.sh [--output DIR]
#
# experiment_configs/common.json에서 공통 파라미터를 읽고,
# 기법별 config (experiment_configs/{method}.json)을 사용합니다.
#
# 실행 순서:
#   1. 공통 설정 확인
#   2. 실험 기법 선택 (SFL, USFL, SCAFFOLD, GAS, MultiSFL)
#   3. 기법별 설정 수정 여부 선택
#   4. GPU 할당
#   5. 병렬 실행 → 결과 한 폴더에 수집
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
CONFIG_DIR="$SCRIPT_DIR/experiment_configs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ========================= Help =========================
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ./run_experiments.sh [--output DIR]

experiment_configs/common.json  → 공통 파라미터
experiment_configs/{method}.json → 기법별 default 설정

실행하면 대화형으로 기법 선택, 설정 수정, GPU 할당을 진행합니다.
EOF
    exit 0
fi

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ========================= Common config =========================
COMMON_FILE="$CONFIG_DIR/common.json"
if [[ ! -f "$COMMON_FILE" ]]; then
    echo "Error: $COMMON_FILE not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Common Config"
echo "=========================================="
python3 -c "
import json
with open('$COMMON_FILE') as f:
    c = json.load(f)
for k, v in c.items():
    print(f'  {k}: {v}')
"
echo "=========================================="
echo ""

# ========================= GPU detection =========================
echo "Available GPUs:"
GPU_INFO=""
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.free \
        --format=csv,noheader 2>/dev/null || true)
fi
if [[ -z "$GPU_INFO" ]]; then
    echo "  (none detected - will use CPU)"
else
    echo "$GPU_INFO" | while IFS= read -r line; do
        echo "  GPU $line"
    done
fi
echo ""

# ========================= Method selection =========================
echo "Select methods to run (space-separated numbers, or 'all'):"
echo "  1) SFL        - base Split Federated Learning"
echo "  2) USFL       - SFL + balancing + grad shuffle + DBS"
echo "  3) SCAFFOLD   - SCAFFOLD control variates on split FL"
echo "  4) GAS        - gradient quality-based client selection"
echo "  5) MultiSFL   - multi-branch SFL with replay"
echo ""
read -rp "> " CHOICE

METHOD_KEYS=("sfl" "usfl" "scaffold" "gas" "multisfl")
METHODS=()

if [[ "$CHOICE" == "all" ]] || [[ "$CHOICE" == "a" ]]; then
    METHODS=("sfl" "usfl" "scaffold" "gas" "multisfl")
else
    for num in $CHOICE; do
        idx=$((num - 1))
        if [[ $idx -ge 0 ]] && [[ $idx -lt ${#METHOD_KEYS[@]} ]]; then
            METHODS+=("${METHOD_KEYS[$idx]}")
        else
            echo "Invalid: $num (skipping)"
        fi
    done
fi

if [[ ${#METHODS[@]} -eq 0 ]]; then
    echo "No methods selected. Exiting."
    exit 0
fi

echo ""
echo "Selected: ${METHODS[*]}"
echo ""

# ========================= Per-method config editing =========================
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

for method in "${METHODS[@]}"; do
    METHOD_CONFIG="$CONFIG_DIR/${method}.json"
    TEMP_CONFIG="$TEMP_DIR/${method}.json"

    if [[ ! -f "$METHOD_CONFIG" ]]; then
        echo "Warning: $METHOD_CONFIG not found, using empty config"
        echo '{}' > "$TEMP_CONFIG"
        continue
    fi

    echo "──────────────────────────────────────────"
    echo "  ${method^^} config"
    echo "──────────────────────────────────────────"
    python3 -c "
import json
with open('$METHOD_CONFIG') as f:
    c = json.load(f)
for k, v in c.items():
    if k.startswith('_'): continue
    print(f'  {k}: {v}')
"
    echo ""
    read -rp "  Modify ${method^^} config? [y/N] " modify

    if [[ "$modify" =~ ^[Yy] ]]; then
        python3 << EDIT_EOF
import json, sys

with open('$METHOD_CONFIG') as f:
    config = json.load(f)

modified = {}
for k, v in config.items():
    if k.startswith('_'):
        modified[k] = v
        continue
    new_val = input(f'    {k} [{v}]: ').strip()
    if new_val == '':
        modified[k] = v
    else:
        try:
            modified[k] = json.loads(new_val)
        except json.JSONDecodeError:
            modified[k] = new_val

with open('$TEMP_CONFIG', 'w') as f:
    json.dump(modified, f, indent=2)
print('  -> Updated')
EDIT_EOF
    else
        cp "$METHOD_CONFIG" "$TEMP_CONFIG"
    fi
    echo ""
done

# ========================= GPU assignment =========================
declare -A GPU_MAP

echo "Assign GPU for each method (enter GPU number, or leave empty for CPU):"
for method in "${METHODS[@]}"; do
    read -rp "  ${method^^} -> GPU: " gpu_input
    GPU_MAP[$method]="${gpu_input:-}"
done

echo ""
echo "=========================================="
echo "  Execution Plan"
echo "=========================================="
for method in "${METHODS[@]}"; do
    gpu="${GPU_MAP[$method]}"
    if [[ -n "$gpu" ]]; then
        echo "  ${method^^}  ->  GPU $gpu"
    else
        echo "  ${method^^}  ->  CPU"
    fi
done
echo "=========================================="
echo ""

read -rp "Start? [Y/n] " confirm
if [[ "$confirm" =~ ^[Nn] ]]; then
    echo "Cancelled."
    exit 0
fi

# ========================= Output directory =========================
if [[ -z "$OUTPUT_DIR" ]]; then
    eval "$(python3 -c "
import json
with open('$COMMON_FILE') as f:
    c = json.load(f)
print(f'_DS={c.get(\"dataset\",\"cifar10\")}')
print(f'_AL={c.get(\"alpha\",0.3)}')
print(f'_RD={c.get(\"rounds\",100)}')
")"
    OUTPUT_DIR="results/${_DS}_a${_AL}_r${_RD}_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR/logs"
cp "$COMMON_FILE" "$OUTPUT_DIR/common_config.json"

echo ""
echo "Output: $OUTPUT_DIR/"
echo ""

# ========================= Generate batch_spec.json & run =========================
BATCH_SPEC="$OUTPUT_DIR/batch_spec.json"

# Build GPU map as JSON
GPU_JSON="{"
for method in "${METHODS[@]}"; do
    gpu="${GPU_MAP[$method]}"
    if [[ -n "$gpu" ]]; then
        GPU_JSON+="\"$method\":$gpu,"
    else
        GPU_JSON+="\"$method\":null,"
    fi
done
GPU_JSON="${GPU_JSON%,}}"

python3 -m experiment_core.generate_spec \
    --config-dir "$CONFIG_DIR" \
    --methods "${METHODS[@]}" \
    --gpu-map "$GPU_JSON" \
    --output "$BATCH_SPEC" \
    --output-dir "$OUTPUT_DIR" \
    --method-configs-dir "$TEMP_DIR" \
    --copy-configs-to "$OUTPUT_DIR"

echo ""

# ========================= Run =========================
python3 -m experiment_core.batch_runner \
    --spec "$BATCH_SPEC" \
    --repo-root "$REPO_ROOT"

echo ""
echo "=========================================="
echo "  Results"
echo "=========================================="
echo "  $OUTPUT_DIR/"
ls -1 "$OUTPUT_DIR"/*.normalized.json 2>/dev/null | while read -r f; do
    echo "    $(basename "$f")"
done
echo ""
echo "  Logs:    $OUTPUT_DIR/logs/"
echo "  Configs: $OUTPUT_DIR/*_config.json"
echo "  Summary: $OUTPUT_DIR/summary.json"
echo "=========================================="
