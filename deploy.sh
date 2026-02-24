#!/bin/bash
set -euo pipefail
###############################################################################
# deploy.sh — 멀티 GPU 서버 실험 자동화
#
# 사용법:
#   ./deploy.sh run usfl@server-a:0 gas@server-b:1     # 분산 실행
#   ./deploy.sh run --server server-a --methods "usfl sfl" --gpus "0 1"
#   ./deploy.sh run -i --server server-a                # 대화형 모드
#   ./deploy.sh status                                  # 전체 상태 확인
#   ./deploy.sh logs server-a [method]                  # 실시간 로그
#   ./deploy.sh attach server-a                         # tmux 접속
#   ./deploy.sh collect                                 # pending 결과 수집
#   ./deploy.sh collect --local                         # 로컬 rsync
#   ./deploy.sh collect <run_name>                      # 특정 실행 수집
#   ./deploy.sh collect --list                          # 배포 이력
#   ./deploy.sh servers                                 # 서버 목록 + GPU 상태
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/deploy/deploy_servers.json"
HISTORY_FILE="$SCRIPT_DIR/deploy/.deploy_history.json"

# ========================= Helpers =========================

die() { echo "Error: $*" >&2; exit 1; }

require_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        die "deploy_servers.json not found. Copy the template and fill in your server details."
    fi
}

require_jq() {
    if ! command -v jq &>/dev/null; then
        die "jq is required. Install with: brew install jq"
    fi
}

get_server_field() {
    local server="$1" field="$2"
    jq -r ".servers.\"$server\".$field // empty" "$CONFIG_FILE"
}

get_global_field() {
    local field="$1"
    jq -r ".$field // empty" "$CONFIG_FILE"
}

# Check if method appears as a +-delimited component in exp_name
# Examples: exp_name="usfl+sfl-g0-222446" matches method "usfl" and "sfl"
#           exp_name="usfl-g0-222446" matches only "usfl"
#           exp_name="sfl_iid-g0-222446" does NOT match "sfl"
_exp_matches_method() {
    local exp_name="$1" method="$2"
    local padded="+${exp_name}"
    # +method+ (middle of group), +method- (end of group/before suffix)
    [[ "$padded" == *"+${method}+"* || "$padded" == *"+${method}-"* ]] && return 0
    # Exact match (legacy format: just "method")
    [[ "$exp_name" == "$method" ]] && return 0
    return 1
}

list_servers() {
    jq -r '.servers | keys[]' "$CONFIG_FILE"
}

# ========================= Deploy History =========================

_append_history() {
    # Usage: _append_history <run_name> <branch> <assignment1> [assignment2 ...]
    local run_name="$1" branch="$2"
    shift 2
    # Build assignments JSON array from remaining args
    local assignments_json
    assignments_json=$(python3 -c "
import json, sys
print(json.dumps(sys.argv[1:]))
" "$@")
    python3 -c "
import json, sys, os
from datetime import datetime
path = sys.argv[1]
rec = {
    'run_name': sys.argv[2],
    'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'branch': sys.argv[3],
    'assignments': json.loads(sys.argv[4]),
    'status': 'pending',
}
history = []
if os.path.exists(path):
    with open(path) as f:
        history = json.load(f)
history.append(rec)
with open(path, 'w') as f:
    json.dump(history, f, indent=2)
" "$HISTORY_FILE" "$run_name" "$branch" "$assignments_json"
}

_mark_history_status() {
    # Usage: _mark_history_status <run_name> <status>
    local run_name="$1" new_status="$2"
    python3 -c "
import json, sys, os
path, target, status = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists(path):
    sys.exit(0)
with open(path) as f:
    history = json.load(f)
for rec in history:
    if rec['run_name'] == target:
        rec['status'] = status
with open(path, 'w') as f:
    json.dump(history, f, indent=2)
" "$HISTORY_FILE" "$run_name" "$new_status"
}

_mark_collected() {
    # Alias for backward compat
    _mark_history_status "$1" "collected"
}

_is_run_complete() {
    # Check if ALL experiments for a run have finished (no active tmux sessions).
    # Usage: _is_run_complete <run_name> <assignments_csv>
    # Returns 0 (true) if complete, 1 (false) if still running
    local run_name="$1" assignments_csv="$2"
    local run_id="${run_name##*_}"

    local tmux_session
    tmux_session=$(get_global_field "tmux_session")
    tmux_session="${tmux_session:-usfl-exp}"

    # Check each unique server once: any session with our run_id means still running
    local checked_servers=""
    local IFS=","
    for entry in $assignments_csv; do
        local rest="${entry#*:}"
        local server="${rest%%:*}"

        case " $checked_servers " in
            *" $server "*) continue ;;
        esac
        checked_servers="$checked_servers $server"

        local ssh_host
        ssh_host=$(get_server_field "$server" "ssh_host")
        if [[ -z "$ssh_host" || "$ssh_host" == "YOUR_SSH_ALIAS"* ]]; then
            continue
        fi

        # shellcheck disable=SC2029
        local active
        active=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux ls 2>/dev/null | grep '^${tmux_session}-' | grep '${run_id}' | head -1" 2>/dev/null \
            || echo "")

        if [[ -n "$active" ]]; then
            return 1  # at least one session still running
        fi
    done
    return 0  # all done
}

_get_history_runs() {
    # Usage: _get_history_runs [pending|collected|all]
    # Prints: run_name<TAB>assignments_csv<TAB>status<TAB>timestamp<TAB>branch per line
    # Merges multiple entries with same run_name (deduped assignments, latest status)
    local filter="${1:-pending}"
    python3 -c "
import json, sys, os
from collections import OrderedDict
path = sys.argv[1]
filt = sys.argv[2]
if not os.path.exists(path):
    sys.exit(0)
with open(path) as f:
    history = json.load(f)
# Merge entries by run_name
merged = OrderedDict()
for rec in history:
    rn = rec['run_name']
    if rn not in merged:
        merged[rn] = {
            'run_name': rn,
            'assignments': list(rec['assignments']),
            'status': rec['status'],
            'timestamp': rec.get('timestamp', ''),
            'branch': rec.get('branch', ''),
        }
    else:
        # Merge assignments (dedup)
        existing = set(merged[rn]['assignments'])
        for a in rec['assignments']:
            if a not in existing:
                merged[rn]['assignments'].append(a)
                existing.add(a)
        # Priority: collected > partial > pending
        cur = merged[rn]['status']
        new = rec['status']
        prio = {'pending': 0, 'partial': 1, 'collected': 2}
        if prio.get(new, 0) > prio.get(cur, 0):
            merged[rn]['status'] = new
for m in merged.values():
    if filt == 'pending' and m['status'] not in ('pending', 'partial'):
        continue
    elif filt != 'all' and filt != 'pending' and m['status'] != filt:
        continue
    assignments = ','.join(m['assignments'])
    print(f\"{m['run_name']}\t{assignments}\t{m['status']}\t{m['timestamp']}\t{m['branch']}\")
" "$HISTORY_FILE" "$filter"
}

_servers_from_assignments() {
    # Extract unique server names from comma-separated assignment string
    # Input: "sfl:xsailor5:0,usfl:xsailor4:0"
    local assignments_csv="$1"
    local seen=""
    local IFS=","
    for entry in $assignments_csv; do
        local rest="${entry#*:}"
        local srv="${rest%%:*}"
        case " $seen " in
            *" $srv "*) ;;
            *) seen="$seen $srv"; echo "$srv" ;;
        esac
    done
}

validate_server() {
    local server="$1"
    local ssh_host
    ssh_host=$(get_server_field "$server" "ssh_host")
    if [[ -z "$ssh_host" || "$ssh_host" == "YOUR_SSH_ALIAS"* ]]; then
        die "Server '$server' not configured. Edit deploy_servers.json."
    fi
}

# ========================= Git Push =========================

get_deploy_branch() {
    # Source of truth: current local branch
    git branch --show-current 2>/dev/null || echo "main"
}

do_git_push() {
    local branch
    branch=$(get_deploy_branch)

    echo "── Git Push (branch: $branch) ──"

    if git diff --quiet HEAD 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
        echo "  No uncommitted changes."
        git push origin "$branch" 2>/dev/null || echo "  (push skipped — already up to date)"
    else
        echo "  Uncommitted changes detected."
        # Check if changes are config-only (experiment_configs/ directory)
        local changed_files
        changed_files=$(git diff --name-only HEAD 2>/dev/null; git diff --cached --name-only 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null)
        changed_files=$(echo "$changed_files" | sort -u | grep -v '^$')

        local non_config
        non_config=$(echo "$changed_files" | grep -v '^experiment_configs/' || true)

        local msg
        if [[ -z "$non_config" ]]; then
            # Config-only changes
            msg="exp: deploy $(date +%H%M)"
            echo "  Config-only changes (experiment_configs/)."
        else
            echo "  Changed files outside experiment_configs/:"
            echo "$non_config" | while IFS= read -r f; do echo "    $f"; done
            echo ""
            read -rp "  Commit message: " msg
            if [[ -z "$msg" ]]; then
                echo "  Cancelled (empty message)."
                return 1
            fi
        fi

        echo ""
        echo "  Commit: \"$msg\""
        read -rp "  Proceed? [Y/n] " confirm
        if [[ "$confirm" =~ ^[Nn] ]]; then
            echo "  Cancelled."
            return 1
        fi

        git add -A
        git commit -m "$msg"
        git push origin "$branch"
    fi
    echo ""
}

# ========================= Parse Assignments =========================
# Stores assignments as a flat array of "method:server:gpu" entries.
# Compatible with bash 3.x (no associative arrays).

ASSIGNMENTS=()

parse_inline_assignments() {
    # Parse: usfl@server-a:0 gas@server-b:1
    for arg in "$@"; do
        if [[ "$arg" =~ ^([a-z0-9_]+)@([a-z0-9_-]+):([0-9]+(,[0-9]+)*)$ ]]; then
            ASSIGNMENTS+=("${BASH_REMATCH[1]}:${BASH_REMATCH[2]}:${BASH_REMATCH[3]}")
        else
            die "Invalid assignment format: '$arg'. Expected: method@server:gpu (e.g., usfl@server-a:0)"
        fi
    done
}

parse_server_flag_assignments() {
    # Parse: --server server-a --methods "usfl sfl" --gpus "0 1"
    local server="$1"
    shift
    local methods_str="" gpus_str=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --methods) methods_str="$2"; shift 2 ;;
            --gpus) gpus_str="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -z "$methods_str" ]]; then
        die "--methods is required with --server"
    fi

    local methods_arr=($methods_str)
    local gpus_arr=($gpus_str)

    for i in "${!methods_arr[@]}"; do
        local method="${methods_arr[$i]}"
        local gpu="${gpus_arr[$i]:-0}"
        ASSIGNMENTS+=("${method}:${server}:${gpu}")
    done
}

# Helper: get unique servers from ASSIGNMENTS
get_unique_servers() {
    local seen=""
    for entry in "${ASSIGNMENTS[@]}"; do
        local srv="${entry#*:}"    # remove method:
        srv="${srv%%:*}"           # remove :gpu
        case " $seen " in
            *" $srv "*) ;;
            *) seen="$seen $srv"; echo "$srv" ;;
        esac
    done
}

# Helper: get methods for a given server
methods_for_server() {
    local target="$1"
    for entry in "${ASSIGNMENTS[@]}"; do
        local method="${entry%%:*}"
        local rest="${entry#*:}"
        local srv="${rest%%:*}"
        if [[ "$srv" == "$target" ]]; then
            echo "$method"
        fi
    done
}

# Helper: get GPU for a given method
gpu_for_method() {
    local target="$1"
    for entry in "${ASSIGNMENTS[@]}"; do
        local method="${entry%%:*}"
        if [[ "$method" == "$target" ]]; then
            local rest="${entry#*:}"
            local gpu="${rest#*:}"
            echo "$gpu"
            return
        fi
    done
    echo "null"
}

# ========================= Run =========================

cmd_run() {
    require_config
    require_jq

    local interactive=false
    local server_flag=""
    local run_name_override=""
    local remaining_args=()

    # Parse run flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i|--interactive) interactive=true; shift ;;
            --server) server_flag="$2"; shift 2 ;;
            --methods|--gpus) remaining_args+=("$1" "$2"); shift 2 ;;
            --no-push) local no_push=true; shift ;;
            --run-name) run_name_override="$2"; shift 2 ;;
            *) remaining_args+=("$1"); shift ;;
        esac
    done

    # Parse assignments
    if [[ -n "$server_flag" ]]; then
        validate_server "$server_flag"
        if [[ "$interactive" == true ]]; then
            # Interactive mode: just SSH into the server with tmux
            :
        else
            parse_server_flag_assignments "$server_flag" "${remaining_args[@]}"
        fi
    elif [[ ${#remaining_args[@]} -gt 0 && "$interactive" == false ]]; then
        parse_inline_assignments "${remaining_args[@]}"
    elif [[ "$interactive" == false ]]; then
        die "Usage: ./deploy.sh run method@server:gpu [...]\n       ./deploy.sh run --server NAME --methods '...' --gpus '...'\n       ./deploy.sh run -i --server NAME"
    fi

    # Git push
    if [[ "${no_push:-}" != true ]]; then
        do_git_push
    fi

    local tmux_session
    tmux_session=$(get_global_field "tmux_session")
    tmux_session="${tmux_session:-usfl-exp}"

    # Interactive mode
    if [[ "$interactive" == true ]]; then
        if [[ -z "$server_flag" ]]; then
            die "--server is required for interactive mode"
        fi
        local ssh_host remote_repo conda_env env_type
        ssh_host=$(get_server_field "$server_flag" "ssh_host")
        remote_repo=$(get_server_field "$server_flag" "remote_repo")
        conda_env=$(get_server_field "$server_flag" "conda_env")
        env_type=$(get_server_field "$server_flag" "env_type")
        env_type="${env_type:-conda}"

        local branch
        branch=$(get_deploy_branch)
        local session_name="${tmux_session}-${server_flag}"

        echo "── Interactive Mode: $server_flag ──"
        echo "  Branch:  $branch"
        echo "  Connecting to $ssh_host..."
        echo "  (Use Ctrl+B D to detach tmux session)"
        echo ""

        # Check for existing tmux session
        # shellcheck disable=SC2029
        if ssh -o ConnectTimeout=5 "$ssh_host" "tmux has-session -t '$session_name' 2>/dev/null"; then
            echo "  WARNING: tmux session '$session_name' already exists on $server_flag."
            read -rp "  Kill existing session and start new? [y/N] " kill_confirm
            if [[ ! "$kill_confirm" =~ ^[Yy] ]]; then
                echo "  Cancelled. Use './deploy.sh attach $server_flag' to connect to existing session."
                return
            fi
            # shellcheck disable=SC2029
            ssh "$ssh_host" "tmux kill-session -t '$session_name'"
        fi

        # shellcheck disable=SC2029
        ssh -t "$ssh_host" "cd $remote_repo && \
            git stash -q 2>/dev/null || true && \
            git fetch -q && \
            git checkout $branch -q && \
            git reset --hard origin/$branch && \
            tmux new-session -s '$session_name' \
            'bash deploy/remote_run.sh $env_type \"$conda_env\" --interactive 2>&1 | tee experiment.log'"
        return
    fi

    local branch
    branch=$(get_deploy_branch)

    # Generate shared RUN_NAME early (needed for session naming)
    local _run_name
    if [[ -n "$run_name_override" ]]; then
        _run_name="$run_name_override"
    else
        local _ds _alpha _rounds _ts
        _ds=$(python3 -c "import json; c=json.load(open('experiment_configs/common.json')); print(c.get('dataset','cifar10'))")
        _alpha=$(python3 -c "import json; c=json.load(open('experiment_configs/common.json')); print(c.get('alpha',0.3))")
        _rounds=$(python3 -c "import json; c=json.load(open('experiment_configs/common.json')); print(c.get('rounds',100))")
        _ts=$(date +%Y%m%d_%H%M%S)
        _run_name="${_ds}_a${_alpha}_r${_rounds}_${_ts}"
    fi
    # run_id = HHMMSS from run_name (e.g., "222446" from "cifar10_a0.3_r100_20260207_222446")
    local _run_id="${_run_name##*_}"

    # ── Group assignments by (server, gpu) for sequential execution ──
    local GROUP_KEYS=()    # "server:gpu" entries (unique)
    local GROUP_METHODS=() # corresponding methods, +-joined (e.g., "usfl+sfl")

    for entry in "${ASSIGNMENTS[@]}"; do
        local method="${entry%%:*}"
        local rest="${entry#*:}"
        local server="${rest%%:*}"
        local gpu="${rest#*:}"
        local key="${server}:${gpu}"

        local found=false
        for i in "${!GROUP_KEYS[@]}"; do
            if [[ "${GROUP_KEYS[$i]}" == "$key" ]]; then
                GROUP_METHODS[$i]="${GROUP_METHODS[$i]}+${method}"
                found=true
                break
            fi
        done
        if [[ "$found" == false ]]; then
            GROUP_KEYS+=("$key")
            GROUP_METHODS+=("$method")
        fi
    done

    # Batch mode: per-GPU tmux sessions (grouped methods run sequentially)
    echo "── Execution Plan ──"
    echo "  Branch: $branch"
    if [[ -n "$run_name_override" ]]; then
        echo "  Run:    $_run_name (reusing)"
    else
        echo "  Run:    $_run_name"
    fi
    for i in "${!GROUP_KEYS[@]}"; do
        local key="${GROUP_KEYS[$i]}"
        local server="${key%%:*}"
        local gpu="${key#*:}"
        local methods="${GROUP_METHODS[$i]}"
        local gpu_safe="${gpu//,/-}"
        local session_name="${tmux_session}-${methods}-g${gpu_safe}-${_run_id}"
        local methods_display="${methods//+/, }"
        if [[ "$methods" == *"+"* ]]; then
            echo "  $methods_display → $server GPU $gpu  (tmux: $session_name) [sequential]"
        else
            echo "  $methods_display → $server GPU $gpu  (tmux: $session_name)"
        fi
    done
    echo ""

    # Check for existing tmux sessions BEFORE starting
    local has_conflict=false
    for i in "${!GROUP_KEYS[@]}"; do
        local key="${GROUP_KEYS[$i]}"
        local server="${key%%:*}"
        local gpu="${key#*:}"
        local methods="${GROUP_METHODS[$i]}"
        local ssh_host
        ssh_host=$(get_server_field "$server" "ssh_host")
        local gpu_safe="${gpu//,/-}"
        local session_name="${tmux_session}-${methods}-g${gpu_safe}-${_run_id}"
        # shellcheck disable=SC2029
        if ssh -o ConnectTimeout=5 "$ssh_host" "tmux has-session -t '$session_name' 2>/dev/null" 2>/dev/null; then
            echo "  WARNING: tmux '$session_name' already running on $server"
            has_conflict=true
        fi
    done

    if [[ "$has_conflict" == true ]]; then
        echo ""
        echo "  Existing session(s) will be killed if you proceed."
        read -rp "  Kill existing sessions and start new experiments? [y/N] " kill_confirm
        if [[ ! "$kill_confirm" =~ ^[Yy] ]]; then
            echo "Cancelled. Use './deploy.sh status' to check running experiments."
            return
        fi
        echo ""
    fi

    read -rp "Start? [Y/n] " confirm
    if [[ "$confirm" =~ ^[Nn] ]]; then
        echo "Cancelled."
        return
    fi

    # Phase 1: Git sync (once per server, sequential)
    echo ""
    for server in $(get_unique_servers); do
        validate_server "$server"
        local ssh_host remote_repo
        ssh_host=$(get_server_field "$server" "ssh_host")
        remote_repo=$(get_server_field "$server" "remote_repo")

        echo "[$server] Syncing branch $branch..."
        # shellcheck disable=SC2029
        ssh "$ssh_host" "cd $remote_repo && \
            git stash -q 2>/dev/null || true && \
            git fetch -q && \
            git checkout $branch -q && \
            git reset --hard origin/$branch -q"
        echo "[$server] Synced."
    done
    echo ""

    echo "  Run name: $_run_name"
    echo "  Results:  results/$_run_name/"
    echo ""

    # Phase 2: Per-GPU tmux sessions (parallel across groups, sequential within)
    local pids=()
    for i in "${!GROUP_KEYS[@]}"; do
        local key="${GROUP_KEYS[$i]}"
        local server="${key%%:*}"
        local gpu="${key#*:}"
        local methods="${GROUP_METHODS[$i]}"
        (
            local ssh_host remote_repo conda_env env_type
            ssh_host=$(get_server_field "$server" "ssh_host")
            remote_repo=$(get_server_field "$server" "remote_repo")
            conda_env=$(get_server_field "$server" "conda_env")
            env_type=$(get_server_field "$server" "env_type")
            env_type="${env_type:-conda}"

            # Generate spec (supports multiple methods → sequential execution)
            local methods_space="${methods//+/ }"
            local gpu_map="{"
            local first=true
            for m in $methods_space; do
                if [[ "$first" == true ]]; then first=false; else gpu_map+=","; fi
                gpu_map+="\"$m\":\"$gpu\""
            done
            gpu_map+="}"

            local gpu_safe="${gpu//,/-}"
            local spec_file="/tmp/usfl_spec_${methods}_g${gpu_safe}_$$.json"
            python3 -m experiment_core.generate_spec \
                --config-dir experiment_configs \
                --methods $methods_space \
                --gpu-map "$gpu_map" \
                --output-dir "results/$_run_name" \
                --output "$spec_file"

            # Transfer spec to server
            local remote_spec="batch_spec_${methods}_g${gpu_safe}.json"
            scp -q "$spec_file" "$ssh_host:$remote_repo/$remote_spec"
            rm -f "$spec_file"

            # Kill existing session if present
            local session_name="${tmux_session}-${methods}-g${gpu_safe}-${_run_id}"
            # shellcheck disable=SC2029
            ssh "$ssh_host" "tmux kill-session -t '$session_name' 2>/dev/null || true"

            # Start experiment(s) in tmux session
            # shellcheck disable=SC2029
            ssh "$ssh_host" "cd $remote_repo && \
                tmux new-session -d -s '$session_name' \
                'bash deploy/remote_run.sh $env_type \"$conda_env\" $remote_spec 2>&1 | tee experiment_${methods}-g${gpu_safe}-${_run_id}.log'"

            local methods_display="${methods//+/, }"
            echo "[$server] $methods_display → GPU $gpu (tmux: $session_name)"
        ) &
        pids+=($!)
    done

    # Wait for all SSH commands to complete
    local failed=false
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=true
        fi
    done

    echo ""
    if [[ "$failed" == true ]]; then
        echo "Some experiments failed to start. Check output above."
    else
        # Record deployment in history
        _append_history "$_run_name" "$branch" "${ASSIGNMENTS[@]}"

        echo "=========================================="
        echo "  All experiments started!"
        echo "=========================================="
        echo ""
        echo "  Status:    ./deploy.sh status"
        echo "  Attach:    ./deploy.sh attach <server> <method>"
        echo "  Logs:      ./deploy.sh logs <server> <method>"
        echo "  Collect:   ./deploy.sh collect"
        echo "=========================================="
    fi
}

# ========================= Status =========================

cmd_status() {
    require_config
    require_jq

    local tmux_session
    tmux_session=$(get_global_field "tmux_session")
    tmux_session="${tmux_session:-usfl-exp}"

    echo "=========================================="
    echo "  Experiment Status"
    echo "=========================================="
    echo ""

    for server in $(list_servers); do
        local ssh_host
        ssh_host=$(get_server_field "$server" "ssh_host")
        if [[ -z "$ssh_host" || "$ssh_host" == "YOUR_SSH_ALIAS"* ]]; then
            continue
        fi

        local remote_repo
        remote_repo=$(get_server_field "$server" "remote_repo")

        echo "── $server ($ssh_host) ──"

        # Discover all experiment tmux sessions on this server
        local sessions=""
        # shellcheck disable=SC2029
        sessions=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux ls 2>/dev/null | grep '^${tmux_session}-' | cut -d: -f1" 2>/dev/null \
            || echo "")

        if [[ -z "$sessions" ]]; then
            echo "  Experiments: (none running)"
        else
            echo "  Experiments:"
            for sess in $sessions; do
                local sess_suffix="${sess#${tmux_session}-}"
                # sess_suffix is "method-run_id" (e.g., "sfl-222446") or legacy "method"
                local log_line=""
                # shellcheck disable=SC2029
                log_line=$(ssh "$ssh_host" \
                    "cd $remote_repo && tail -1 experiment_${sess_suffix}.log 2>/dev/null" 2>/dev/null \
                    || echo "")
                # Fallback: try legacy log name (without run_id)
                if [[ -z "$log_line" ]]; then
                    local method_only="${sess_suffix%-*}"
                    # shellcheck disable=SC2029
                    log_line=$(ssh "$ssh_host" \
                        "cd $remote_repo && tail -1 experiment_${method_only}.log 2>/dev/null" 2>/dev/null \
                        || echo "")
                fi
                if [[ -n "$log_line" ]]; then
                    echo "    $sess_suffix: RUNNING  ← $log_line"
                else
                    echo "    $sess_suffix: RUNNING"
                fi
            done
        fi

        # GPU utilization
        echo "  GPUs:"
        # shellcheck disable=SC2029
        ssh -o ConnectTimeout=5 "$ssh_host" \
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null" 2>/dev/null \
            | while IFS= read -r line; do echo "    $line"; done \
            || echo "    (unavailable)"

        echo ""
    done
}

# ========================= Logs =========================

cmd_logs() {
    require_config
    require_jq

    local server="${1:?Usage: ./deploy.sh logs <server> [method]}"
    local method="${2:-}"

    validate_server "$server"
    local ssh_host remote_repo
    ssh_host=$(get_server_field "$server" "ssh_host")
    remote_repo=$(get_server_field "$server" "remote_repo")

    if [[ -n "$method" ]]; then
        # Find log file: try exact match, single-method patterns, then grouped patterns
        # shellcheck disable=SC2029
        local log_file
        log_file=$(ssh "$ssh_host" "cd $remote_repo && \
            if [ -f experiment_${method}.log ]; then echo experiment_${method}.log; \
            else ls -t experiment_${method}-*.log experiment_${method}+*.log experiment_*+${method}-*.log experiment_*+${method}+*.log 2>/dev/null | head -1; fi" 2>/dev/null)

        if [[ -z "$log_file" ]]; then
            echo "No log file found for '$method' on $server."
            echo ""
            echo "Available logs:"
            # shellcheck disable=SC2029
            ssh "$ssh_host" "cd $remote_repo && ls -1 experiment_*.log 2>/dev/null" \
                | while IFS= read -r f; do echo "  $f"; done
            return
        fi
        echo "Tailing: $log_file on $server"
        echo "(Ctrl+C to stop)"
        echo ""
        # shellcheck disable=SC2029
        ssh "$ssh_host" "cd $remote_repo && tail -f $log_file" 2>/dev/null
    else
        # List available experiment logs
        echo "Available experiment logs on $server:"
        # shellcheck disable=SC2029
        ssh "$ssh_host" "cd $remote_repo && ls -1 experiment_*.log 2>/dev/null" \
            | while IFS= read -r f; do echo "  $f"; done
        echo ""
        echo "Usage: ./deploy.sh logs $server <method>"
        echo "  Or:  ./deploy.sh attach $server <method>  (live tmux session)"
    fi
}

# ========================= Attach =========================

cmd_attach() {
    require_config
    require_jq

    local server="${1:?Usage: ./deploy.sh attach <server> <method>}"
    local method="${2:-}"
    validate_server "$server"

    local ssh_host tmux_session
    ssh_host=$(get_server_field "$server" "ssh_host")
    tmux_session=$(get_global_field "tmux_session")
    tmux_session="${tmux_session:-usfl-exp}"

    if [[ -z "$method" ]]; then
        # List available sessions on this server
        echo "Active experiment sessions on $server:"
        # shellcheck disable=SC2029
        ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux ls 2>/dev/null | grep '^${tmux_session}-'" 2>/dev/null \
            | while IFS= read -r line; do echo "  $line"; done
        echo ""
        echo "Usage: ./deploy.sh attach $server <method>"
        return
    fi

    # Find matching session: search for method as a component in session name
    # Handles: "method-g0-123", "method+other-g0-123", "other+method-g0-123"
    local session_name=""
    # shellcheck disable=SC2029
    session_name=$(ssh -o ConnectTimeout=5 "$ssh_host" \
        "tmux ls 2>/dev/null | grep '^${tmux_session}-' | cut -d: -f1 | \
        grep -m1 -E '^${tmux_session}-(${method}[+-]|.*\\+${method}[+-]|.*\\+${method}\$)'" 2>/dev/null \
        || echo "")

    if [[ -z "$session_name" ]]; then
        echo "No session found matching '$method' on $server."
        echo ""
        echo "Active sessions:"
        # shellcheck disable=SC2029
        ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux ls 2>/dev/null | grep '^${tmux_session}-'" 2>/dev/null \
            | while IFS= read -r line; do echo "  $line"; done
        return
    fi

    echo "Attaching to $session_name on $ssh_host..."
    echo "(Ctrl+B D to detach)"
    echo ""
    ssh -t "$ssh_host" "tmux attach-session -t '$session_name'"
}

# ========================= Collect =========================

_collect_run_from_server() {
    # Usage: _collect_run_from_server <server> <run_name> <local_mode> [gdrive_remote]
    local server="$1" run_name="$2" local_mode="$3" gdrive_remote="${4:-}"
    local ssh_host remote_repo
    ssh_host=$(get_server_field "$server" "ssh_host")
    remote_repo=$(get_server_field "$server" "remote_repo")

    if [[ -z "$ssh_host" || "$ssh_host" == "YOUR_SSH_ALIAS"* ]]; then
        return 1
    fi

    local remote_dir="$remote_repo/results/$run_name"
    # shellcheck disable=SC2029
    local exists
    exists=$(ssh -o ConnectTimeout=5 "$ssh_host" \
        "test -d '$remote_dir' && echo yes || echo no" 2>/dev/null || echo "no")

    if [[ "$exists" != "yes" ]]; then
        echo "    $server: results/$run_name/ not found"
        return 1
    fi

    if [[ "$local_mode" == true ]]; then
        local local_dst="./results/${run_name}"
        echo "    $server: rsync → $local_dst"
        mkdir -p "$local_dst"
        rsync -avz --progress "$ssh_host:$remote_dir/" "$local_dst/"
    else
        local gdrive_dst="${gdrive_remote}/${run_name}"
        echo "    $server: rclone → $gdrive_dst"
        # shellcheck disable=SC2029
        ssh "$ssh_host" "rclone copy '$remote_dir/' '$gdrive_dst' -P"
    fi
    return 0
}

cmd_collect() {
    require_config
    require_jq

    local local_mode=false
    local filter="pending"
    local target_run=""
    local show_list=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --local) local_mode=true; shift ;;
            --all)   filter="all"; shift ;;
            --list)  show_list=true; shift ;;
            -*)      die "Unknown option: $1" ;;
            *)       target_run="$1"; shift ;;
        esac
    done

    # --list: show deploy history and exit
    if [[ "$show_list" == true ]]; then
        echo "=========================================="
        echo "  Deploy History"
        echo "=========================================="
        echo ""
        if [[ ! -f "$HISTORY_FILE" ]]; then
            echo "  No history yet. Run './deploy.sh run' to start experiments."
            return
        fi
        printf "  %-44s %-10s %-24s %s\n" "RUN NAME" "STATUS" "TIMESTAMP" "ASSIGNMENTS"
        printf "  %-44s %-10s %-24s %s\n" "--------" "------" "---------" "-----------"
        while IFS=$'\t' read -r rn asgn st ts br; do
            printf "  %-44s %-10s %-24s %s\n" "$rn" "$st" "$ts" "$asgn"
        done < <(_get_history_runs "all")
        echo ""
        return
    fi

    local gdrive_remote
    gdrive_remote=$(get_global_field "gdrive_rclone_remote")

    if [[ "$local_mode" == false && -z "$gdrive_remote" ]]; then
        die "gdrive_rclone_remote not set in deploy_servers.json. Use --local for rsync, or run setup_rclone_gdrive.sh on your servers."
    fi

    echo "=========================================="
    echo "  Collecting Results"
    echo "=========================================="
    echo ""

    # Case 1: Specific run_name given
    if [[ -n "$target_run" ]]; then
        echo "── Collecting: $target_run ──"

        # Try to find servers from history first (already merged by run_name)
        local found_in_history=false
        local assignments_csv=""
        while IFS=$'\t' read -r rn ac st ts br; do
            if [[ "$rn" == "$target_run" ]]; then
                found_in_history=true
                assignments_csv="$ac"
                break  # safe: _get_history_runs already merged duplicates
            fi
        done < <(_get_history_runs "all")

        local collected=false
        if [[ "$found_in_history" == true ]]; then
            for server in $(_servers_from_assignments "$assignments_csv"); do
                if _collect_run_from_server "$server" "$target_run" "$local_mode" "$gdrive_remote"; then
                    collected=true
                fi
            done
        else
            echo "  (not in history — searching all servers)"
            for server in $(list_servers); do
                if _collect_run_from_server "$server" "$target_run" "$local_mode" "$gdrive_remote"; then
                    collected=true
                fi
            done
        fi

        if [[ "$collected" == true ]]; then
            echo ""
            # Check if all experiments are actually done before marking collected
            if [[ "$found_in_history" == true ]] && ! _is_run_complete "$target_run" "$assignments_csv"; then
                _mark_history_status "$target_run" "partial"
                echo "Done (partial). Some experiments still running — will re-collect on next './deploy.sh collect'."
            else
                _mark_collected "$target_run"
                echo "Done. All experiments finished — marked as collected."
            fi
        else
            echo ""
            echo "No results found for '$target_run' on any server."
        fi
        return
    fi

    # Case 2: Collect from history (pending or all)
    if [[ ! -f "$HISTORY_FILE" ]]; then
        echo "  No deploy history. Run './deploy.sh run' first,"
        echo "  or specify a run name: ./deploy.sh collect <run_name>"
        return
    fi

    local run_count=0
    local collected_count=0
    while IFS=$'\t' read -r rn asgn_csv st ts br; do
        run_count=$((run_count + 1))
        echo "── $rn ($st) ──"

        local run_collected=false
        for server in $(_servers_from_assignments "$asgn_csv"); do
            if _collect_run_from_server "$server" "$rn" "$local_mode" "$gdrive_remote"; then
                run_collected=true
            fi
        done

        if [[ "$run_collected" == true ]]; then
            # Check if all experiments are done before marking collected
            if _is_run_complete "$rn" "$asgn_csv"; then
                _mark_collected "$rn"
                collected_count=$((collected_count + 1))
            else
                _mark_history_status "$rn" "partial"
                echo "    (partial — some experiments still running)"
            fi
        else
            echo "    (no results found on any server)"
        fi
        echo ""
    done < <(_get_history_runs "$filter")

    if [[ "$run_count" -eq 0 ]]; then
        if [[ "$filter" == "pending" ]]; then
            echo "  No pending runs to collect."
            echo "  Use --all to re-collect previously collected runs,"
            echo "  or --list to see deploy history."
        else
            echo "  No runs in history."
        fi
    else
        echo "Done. Collected $collected_count / $run_count run(s)."
    fi
}

# ========================= Servers =========================

cmd_servers() {
    require_config
    require_jq

    echo "=========================================="
    echo "  Registered Servers"
    echo "=========================================="
    echo ""

    for server in $(list_servers); do
        local ssh_host gpus conda_env remote_repo env_type
        ssh_host=$(get_server_field "$server" "ssh_host")
        gpus=$(jq -r ".servers.\"$server\".gpus | join(\", \")" "$CONFIG_FILE")
        conda_env=$(get_server_field "$server" "conda_env")
        remote_repo=$(get_server_field "$server" "remote_repo")
        env_type=$(get_server_field "$server" "env_type")
        env_type="${env_type:-conda}"

        echo "  $server"
        echo "    SSH:    $ssh_host"
        echo "    Repo:   $remote_repo"
        if [[ "$env_type" == "conda" ]]; then
            echo "    Env:    conda ($conda_env)"
        else
            echo "    Env:    $env_type"
        fi
        echo "    GPUs:   [$gpus]"

        # Try to get live GPU info
        if [[ "$ssh_host" != "YOUR_SSH_ALIAS"* ]]; then
            echo "    Live:"
            # shellcheck disable=SC2029
            ssh -o ConnectTimeout=3 "$ssh_host" \
                "nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null" 2>/dev/null \
                | while IFS= read -r line; do echo "      GPU $line"; done \
                || echo "      (unreachable)"
        fi
        echo ""
    done
}

# ========================= Kill =========================

cmd_kill() {
    require_config
    require_jq

    local mode="interactive"
    local target_server=""
    local target_method=""

    case "${1:-}" in
        --all)    mode="all" ;;
        "") mode="interactive" ;;
        *)
            target_server="$1"
            validate_server "$target_server"
            target_method="${2:-}"
            mode="targeted"
            ;;
    esac

    local tmux_session
    tmux_session=$(get_global_field "tmux_session")
    tmux_session="${tmux_session:-usfl-exp}"

    # Collect all sessions first
    local sess_list=()    # "server|exp_name|ssh_host|sess_name|last_log"
    for server in $(list_servers); do
        if [[ "$mode" == "targeted" && "$server" != "$target_server" ]]; then
            continue
        fi

        local ssh_host remote_repo
        ssh_host=$(get_server_field "$server" "ssh_host")
        remote_repo=$(get_server_field "$server" "remote_repo")
        if [[ -z "$ssh_host" || "$ssh_host" == "YOUR_SSH_ALIAS"* ]]; then
            continue
        fi

        local sessions=""
        sessions=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux ls 2>/dev/null | grep '^${tmux_session}-' | cut -d: -f1" 2>/dev/null \
            || echo "")

        if [[ -z "$sessions" ]]; then
            continue
        fi

        for sess in $sessions; do
            local exp_name="${sess#${tmux_session}-}"

            if [[ "$mode" == "targeted" && -n "$target_method" ]]; then
                # Match method as a +-delimited component (handles grouped sessions)
                # e.g., "usfl" matches "usfl-g0-123" and "usfl+sfl-g0-123"
                if ! _exp_matches_method "$exp_name" "$target_method"; then
                    continue
                fi
            fi

            local last_log=""
            # shellcheck disable=SC2029
            last_log=$(ssh "$ssh_host" \
                "cd $remote_repo && tail -1 experiment_${exp_name}.log 2>/dev/null" 2>/dev/null \
                || echo "(no log)")

            sess_list+=("${server}|${exp_name}|${ssh_host}|${sess}|${last_log}")
        done
    done

    if [[ ${#sess_list[@]} -eq 0 ]]; then
        echo "No experiment sessions found."
        return
    fi

    # Interactive mode: show numbered list and ask
    if [[ "$mode" == "interactive" ]]; then
        echo "Active experiment sessions:"
        echo ""
        local i=1
        for entry in "${sess_list[@]}"; do
            IFS='|' read -r sv exp _ _ log <<< "$entry"
            echo "  $i) $exp on $sv"
            echo "     └─ $log"
            i=$((i + 1))
        done
        echo ""
        echo "Options: 'all' to kill all, comma-separated numbers (e.g. 1,2,3), or 'q' to cancel"
        read -rp "Kill which sessions? " choice

        if [[ "$choice" == "q" || -z "$choice" ]]; then
            echo "Cancelled."
            return
        fi

        local killed=0
        if [[ "$choice" == "all" ]]; then
            for entry in "${sess_list[@]}"; do
                IFS='|' read -r sv exp ssh_h sess _ <<< "$entry"
                echo "  KILL $exp on $sv"
                # shellcheck disable=SC2029
                ssh "$ssh_h" "tmux kill-session -t '$sess'" 2>/dev/null || true
                killed=$((killed + 1))
            done
        else
            # Parse comma-separated numbers
            IFS=',' read -ra nums <<< "$choice"
            for num in "${nums[@]}"; do
                num=$(echo "$num" | tr -d ' ')
                if [[ "$num" =~ ^[0-9]+$ ]] && [[ "$num" -ge 1 ]] && [[ "$num" -le ${#sess_list[@]} ]]; then
                    local idx=$((num - 1))
                    IFS='|' read -r sv exp ssh_h sess _ <<< "${sess_list[$idx]}"
                    echo "  KILL $exp on $sv"
                    # shellcheck disable=SC2029
                    ssh "$ssh_h" "tmux kill-session -t '$sess'" 2>/dev/null || true
                    killed=$((killed + 1))
                else
                    echo "  Invalid: $num (skipped)"
                fi
            done
        fi

        echo ""
        echo "Killed $killed session(s)."
        return
    fi

    # --all mode: confirm then kill everything
    if [[ "$mode" == "all" ]]; then
        echo "Will kill ALL ${#sess_list[@]} experiment session(s):"
        for entry in "${sess_list[@]}"; do
            IFS='|' read -r sv exp _ _ _ <<< "$entry"
            echo "  $exp on $sv"
        done
        echo ""
        read -rp "Continue? [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy] ]]; then
            echo "Cancelled."
            return
        fi
    fi

    # Targeted or --all (after confirm): kill matching sessions
    local killed=0
    for entry in "${sess_list[@]}"; do
        IFS='|' read -r sv exp ssh_h sess _ <<< "$entry"
        echo "  KILL $exp on $sv"
        # shellcheck disable=SC2029
        ssh "$ssh_h" "tmux kill-session -t '$sess'" 2>/dev/null || true
        killed=$((killed + 1))
    done
    echo ""
    echo "Killed $killed session(s)."
}

# ========================= Check Run =========================

cmd_check() {
    require_config
    require_jq

    local run_name="${1:-}"
    local history_file="$SCRIPT_DIR/.deploy_history.json"

    if [[ ! -f "$history_file" ]]; then
        die "No deploy history found. Run './deploy.sh run' first."
    fi

    # If no run_name given, use latest pending run
    if [[ -z "$run_name" ]]; then
        run_name=$(python3 -c "
import json
with open('$history_file') as f:
    history = json.load(f)
pending = [r for r in history if r.get('status') == 'pending']
if pending:
    print(pending[-1]['run_name'])
else:
    print('')
")
        if [[ -z "$run_name" ]]; then
            echo "No pending runs in history. Use './deploy.sh collect --list' to see all."
            return
        fi
    fi

    # Look up assignments for this run (merge all history entries with same run_name)
    local assignment_list
    assignment_list=$(python3 -c "
import json
with open('$history_file') as f:
    history = json.load(f)
matches = [r for r in history if r.get('run_name') == '$run_name']
if not matches:
    exit(1)
seen = set()
for m in matches:
    for a in m.get('assignments', []):
        if a not in seen:
            seen.add(a)
            print(a)
")

    if [[ -z "$assignment_list" ]]; then
        die "Run '$run_name' not found in deploy history."
    fi

    local tmux_session
    tmux_session=$(get_global_field "tmux_session")
    tmux_session="${tmux_session:-usfl-exp}"

    # run_id from run_name (HHMMSS)
    local run_id="${run_name##*_}"

    echo "── Run: $run_name ──"
    echo ""

    local running=0
    local crashed=0
    local done_count=0
    local total=0

    while IFS= read -r entry; do
        [[ -z "$entry" ]] && continue
        local method="${entry%%:*}"
        local rest="${entry#*:}"
        local server="${rest%%:*}"

        local ssh_host
        ssh_host=$(get_server_field "$server" "ssh_host")
        if [[ -z "$ssh_host" ]]; then
            echo "  $method on $server: UNKNOWN (server not configured)"
            continue
        fi

        local remote_repo
        remote_repo=$(get_server_field "$server" "remote_repo")

        total=$((total + 1))

        # Discover session for this method (handles grouped sessions like "usfl+sfl-g0-123")
        # shellcheck disable=SC2029
        local all_sessions
        all_sessions=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux ls 2>/dev/null | grep '^${tmux_session}-' | cut -d: -f1" 2>/dev/null \
            || echo "")

        local sess="" log_name=""
        for candidate in $all_sessions; do
            local candidate_exp="${candidate#${tmux_session}-}"
            if _exp_matches_method "$candidate_exp" "$method" && [[ "$candidate_exp" == *"${run_id}"* ]]; then
                sess="$candidate"
                log_name="experiment_${candidate_exp}.log"
                break
            fi
        done

        # Legacy fallback: exact session name without run_id
        if [[ -z "$sess" ]]; then
            for candidate in $all_sessions; do
                if [[ "$candidate" == "${tmux_session}-${method}" ]]; then
                    sess="$candidate"
                    log_name="experiment_${method}.log"
                    break
                fi
            done
        fi

        local has_session="no"
        if [[ -n "$sess" ]]; then
            has_session="yes"
        fi

        if [[ "$has_session" == "yes" ]]; then
            # Session exists — check if crashed or running
            local fail_count
            # shellcheck disable=SC2029
            fail_count=$(ssh "$ssh_host" \
                "cd $remote_repo && { tail -20 $log_name 2>/dev/null | grep -c 'FAIL (exit=' || true; }" 2>/dev/null)
            fail_count="${fail_count:-0}"
            if [[ "$fail_count" -gt 0 ]]; then
                echo "  $method on $server: CRASHED"
                crashed=$((crashed + 1))
            else
                # Show last log line for context
                local last_line
                # shellcheck disable=SC2029
                last_line=$(ssh "$ssh_host" \
                    "cd $remote_repo && tail -1 $log_name 2>/dev/null" 2>/dev/null \
                    || echo "")
                echo "  $method on $server: RUNNING  ← $last_line"
                running=$((running + 1))
            fi
        else
            # Session gone — check if results exist
            local has_results
            # shellcheck disable=SC2029
            has_results=$(ssh "$ssh_host" \
                "test -f $remote_repo/results/$run_name/${method}.normalized.json && echo 'yes' || echo 'no'" 2>/dev/null \
                || echo "no")
            if [[ "$has_results" == "yes" ]]; then
                echo "  $method on $server: DONE"
                done_count=$((done_count + 1))
            else
                echo "  $method on $server: EXITED (no results)"
                crashed=$((crashed + 1))
            fi
        fi
    done <<< "$assignment_list"

    echo ""
    echo "Total: $total  |  Done: $done_count  |  Running: $running  |  Crashed: $crashed"

    if [[ $running -eq 0 && $crashed -eq 0 ]]; then
        echo "All experiments completed. Ready to collect."
    elif [[ $running -eq 0 ]]; then
        echo "No experiments running. $crashed crashed — fix and rerun with --run-name."
    fi
}

# ========================= Help =========================

cmd_help() {
    cat <<'EOF'
deploy.sh — 멀티 GPU 서버 실험 자동화

각 실험은 독립 tmux 세션에서 실행됩니다.

Commands:
  run       실험 실행 (push → pull → 실험별 tmux 세션 생성)
  status    전체 서버/실험 상태 확인
  check     특정 배포의 실험 완료 여부 확인
  kill      crashed/전체 실험 세션 종료
  attach    실험 tmux 세션 접속 (실시간 로그)
  logs      실험 로그 파일 tail
  collect   결과 수집 (Google Drive 또는 로컬)
  servers   등록된 서버 목록 + GPU 상태

Run Usage:
  ./deploy.sh run usfl@server-a:0 gas@server-b:1
  ./deploy.sh run --server server-a --methods "usfl sfl" --gpus "0 1"
  ./deploy.sh run usfl@s:0 sfl@s:0 gas@s:1     # usfl+sfl run sequentially on GPU 0
  ./deploy.sh run -i --server server-a          # 대화형 모드
  ./deploy.sh run --no-push usfl@server-a:0     # git push 스킵
  ./deploy.sh run --run-name <name> usfl@s:0    # 기존 run에 실험 추가/재실행

Monitor:
  ./deploy.sh status                # 모든 서버의 실험 현황
  ./deploy.sh attach server-a sfl   # sfl 실험 tmux 접속 (실시간)
  ./deploy.sh attach server-a       # 세션 목록 보기
  ./deploy.sh logs server-a sfl     # sfl 로그 파일 tail

Kill:
  ./deploy.sh kill                 # 세션 목록 → 번호 선택하여 종료
  ./deploy.sh kill --all           # 모든 실험 세션 종료 (확인 필요)
  ./deploy.sh kill <server>        # 특정 서버 세션 모두 종료
  ./deploy.sh kill <server> <method>  # 특정 실험 세션 종료

Check:
  ./deploy.sh check                # 최근 pending 배포의 실험 상태
  ./deploy.sh check <run_name>     # 특정 배포의 실험 상태

Collect:
  ./deploy.sh collect              # pending 실행 결과 수집 (GDrive)
  ./deploy.sh collect --local      # pending 실행 결과 수집 (로컬 rsync)
  ./deploy.sh collect <run_name>   # 특정 실행 결과 수집
  ./deploy.sh collect --all        # 모든 실행 결과 재수집 (collected 포함)
  ./deploy.sh collect --list       # 배포 이력 보기

Setup:
  1. Edit deploy_servers.json with your server details
  2. Run setup_rclone_gdrive.sh on each GPU server (for Google Drive)
EOF
}

# ========================= Main =========================

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    run)     cmd_run "$@" ;;
    status)  cmd_status "$@" ;;
    check)   cmd_check "$@" ;;
    kill)    cmd_kill "$@" ;;
    logs)    cmd_logs "$@" ;;
    attach)  cmd_attach "$@" ;;
    collect) cmd_collect "$@" ;;
    servers) cmd_servers "$@" ;;
    help|-h|--help) cmd_help ;;
    *) die "Unknown command: $COMMAND. Run './deploy.sh help' for usage." ;;
esac
