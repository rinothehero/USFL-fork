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
#   ./deploy.sh collect                                 # GDrive 업로드
#   ./deploy.sh collect --local                         # 로컬 rsync
#   ./deploy.sh servers                                 # 서버 목록 + GPU 상태
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/deploy_servers.json"

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

list_servers() {
    jq -r '.servers | keys[]' "$CONFIG_FILE"
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
        read -rp "  Commit message [exp: deploy $(date +%H%M)]: " msg
        msg="${msg:-exp: deploy $(date +%Y%m%d_%H%M%S)}"
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
        if [[ "$arg" =~ ^([a-z0-9_]+)@([a-z0-9_-]+):([0-9]+)$ ]]; then
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
    local remaining_args=()

    # Parse run flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i|--interactive) interactive=true; shift ;;
            --server) server_flag="$2"; shift 2 ;;
            --methods|--gpus) remaining_args+=("$1" "$2"); shift 2 ;;
            --no-push) local no_push=true; shift ;;
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
        local ssh_host remote_repo conda_env
        ssh_host=$(get_server_field "$server_flag" "ssh_host")
        remote_repo=$(get_server_field "$server_flag" "remote_repo")
        conda_env=$(get_server_field "$server_flag" "conda_env")

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
            'bash remote_run.sh $conda_env --interactive 2>&1 | tee experiment.log'"
        return
    fi

    local branch
    branch=$(get_deploy_branch)

    # Batch mode: per-experiment tmux sessions
    echo "── Execution Plan ──"
    echo "  Branch: $branch"
    for entry in "${ASSIGNMENTS[@]}"; do
        local method="${entry%%:*}"
        local rest="${entry#*:}"
        local server="${rest%%:*}"
        local gpu="${rest#*:}"
        echo "  $method → $server GPU $gpu  (tmux: ${tmux_session}-${method})"
    done
    echo ""

    # Check for existing tmux sessions BEFORE starting
    local has_conflict=false
    for entry in "${ASSIGNMENTS[@]}"; do
        local method="${entry%%:*}"
        local rest="${entry#*:}"
        local server="${rest%%:*}"
        local ssh_host
        ssh_host=$(get_server_field "$server" "ssh_host")
        local session_name="${tmux_session}-${method}"
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

    # Phase 2: Per-experiment tmux sessions (parallel)
    local pids=()
    for entry in "${ASSIGNMENTS[@]}"; do
        local method="${entry%%:*}"
        local rest="${entry#*:}"
        local server="${rest%%:*}"
        local gpu="${rest#*:}"
        (
            local ssh_host remote_repo conda_env
            ssh_host=$(get_server_field "$server" "ssh_host")
            remote_repo=$(get_server_field "$server" "remote_repo")
            conda_env=$(get_server_field "$server" "conda_env")

            # Generate single-experiment spec
            local spec_file="/tmp/usfl_spec_${method}_$$.json"
            python3 -m experiment_core.generate_spec \
                --config-dir experiment_configs \
                --methods "$method" \
                --gpu-map "{\"$method\":$gpu}" \
                --output "$spec_file"

            # Transfer spec to server
            local remote_spec="batch_spec_${method}.json"
            scp -q "$spec_file" "$ssh_host:$remote_repo/$remote_spec"
            rm -f "$spec_file"

            # Kill existing session if present
            local session_name="${tmux_session}-${method}"
            # shellcheck disable=SC2029
            ssh "$ssh_host" "tmux kill-session -t '$session_name' 2>/dev/null || true"

            # Start experiment in its own tmux session
            # shellcheck disable=SC2029
            ssh "$ssh_host" "cd $remote_repo && \
                tmux new-session -d -s '$session_name' \
                'bash remote_run.sh $conda_env $remote_spec 2>&1 | tee experiment_${method}.log'"

            echo "[$server] $method → GPU $gpu (tmux: $session_name)"
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
                local exp_name="${sess#${tmux_session}-}"
                local log_line=""
                # shellcheck disable=SC2029
                log_line=$(ssh "$ssh_host" \
                    "cd $remote_repo && tail -1 experiment_${exp_name}.log 2>/dev/null" 2>/dev/null \
                    || echo "")
                if [[ -n "$log_line" ]]; then
                    echo "    $exp_name: RUNNING  ← $log_line"
                else
                    echo "    $exp_name: RUNNING"
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
        echo "Tailing: experiment_${method}.log on $server"
        echo "(Ctrl+C to stop)"
        echo ""
        # shellcheck disable=SC2029
        ssh "$ssh_host" "cd $remote_repo && tail -f experiment_${method}.log" 2>/dev/null
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

    local session_name="${tmux_session}-${method}"
    echo "Attaching to $session_name on $ssh_host..."
    echo "(Ctrl+B D to detach)"
    echo ""
    ssh -t "$ssh_host" "tmux attach-session -t '$session_name'"
}

# ========================= Collect =========================

cmd_collect() {
    require_config
    require_jq

    local local_mode=false
    if [[ "${1:-}" == "--local" ]]; then
        local_mode=true
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

    for server in $(list_servers); do
        local ssh_host
        ssh_host=$(get_server_field "$server" "ssh_host")
        if [[ -z "$ssh_host" || "$ssh_host" == "YOUR_SSH_ALIAS"* ]]; then
            continue
        fi

        local remote_repo
        remote_repo=$(get_server_field "$server" "remote_repo")

        echo "── $server ──"

        # Find latest results directory
        # shellcheck disable=SC2029
        local latest
        latest=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "ls -td $remote_repo/results/*/ 2>/dev/null | head -1" 2>/dev/null || echo "")

        if [[ -z "$latest" ]]; then
            echo "  No results found."
            echo ""
            continue
        fi

        local dirname
        dirname=$(basename "$latest")
        echo "  Latest: $dirname"

        if [[ "$local_mode" == true ]]; then
            local local_dst="./results/${dirname}_${server}"
            echo "  rsync → $local_dst"
            mkdir -p "$local_dst"
            rsync -avz --progress "$ssh_host:$latest" "$local_dst/"
        else
            local gdrive_dst="${gdrive_remote}/${dirname}_${server}"
            echo "  rclone → $gdrive_dst"
            # shellcheck disable=SC2029
            ssh "$ssh_host" "rclone copy '$latest' '$gdrive_dst' -P"
        fi

        echo ""
    done

    echo "Done."
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
        local ssh_host gpus conda_env remote_repo
        ssh_host=$(get_server_field "$server" "ssh_host")
        gpus=$(jq -r ".servers.\"$server\".gpus | join(\", \")" "$CONFIG_FILE")
        conda_env=$(get_server_field "$server" "conda_env")
        remote_repo=$(get_server_field "$server" "remote_repo")

        echo "  $server"
        echo "    SSH:    $ssh_host"
        echo "    Repo:   $remote_repo"
        echo "    Conda:  $conda_env"
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

# ========================= Help =========================

cmd_help() {
    cat <<'EOF'
deploy.sh — 멀티 GPU 서버 실험 자동화

각 실험은 독립 tmux 세션에서 실행됩니다.

Commands:
  run       실험 실행 (push → pull → 실험별 tmux 세션 생성)
  status    전체 서버/실험 상태 확인
  attach    실험 tmux 세션 접속 (실시간 로그)
  logs      실험 로그 파일 tail
  collect   결과 수집 (Google Drive 또는 로컬)
  servers   등록된 서버 목록 + GPU 상태

Run Usage:
  ./deploy.sh run usfl@server-a:0 gas@server-b:1
  ./deploy.sh run --server server-a --methods "usfl sfl" --gpus "0 1"
  ./deploy.sh run -i --server server-a          # 대화형 모드
  ./deploy.sh run --no-push usfl@server-a:0     # git push 스킵

Monitor:
  ./deploy.sh status                # 모든 서버의 실험 현황
  ./deploy.sh attach server-a sfl   # sfl 실험 tmux 접속 (실시간)
  ./deploy.sh attach server-a       # 세션 목록 보기
  ./deploy.sh logs server-a sfl     # sfl 로그 파일 tail

Other:
  ./deploy.sh collect              # Google Drive 업로드
  ./deploy.sh collect --local      # 로컬 rsync

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
    logs)    cmd_logs "$@" ;;
    attach)  cmd_attach "$@" ;;
    collect) cmd_collect "$@" ;;
    servers) cmd_servers "$@" ;;
    help|-h|--help) cmd_help ;;
    *) die "Unknown command: $COMMAND. Run './deploy.sh help' for usage." ;;
esac
