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
        ssh -t "$ssh_host" "cd $remote_repo && git fetch -q && git checkout $branch && git pull origin $branch && \
            tmux new-session -s '$session_name' \
            'bash remote_run.sh $conda_env --interactive 2>&1 | tee experiment.log'"
        return
    fi

    local branch
    branch=$(get_deploy_branch)

    # Batch mode: per-server parallel execution
    echo "── Execution Plan ──"
    echo "  Branch: $branch"
    for server in $(get_unique_servers); do
        local methods
        methods=$(methods_for_server "$server")
        echo "  $server:"
        for m in $methods; do
            local gpu
            gpu=$(gpu_for_method "$m")
            echo "    $m → GPU $gpu"
        done
    done
    echo ""

    # Check for existing tmux sessions BEFORE starting
    local has_conflict=false
    for server in $(get_unique_servers); do
        local ssh_host
        ssh_host=$(get_server_field "$server" "ssh_host")
        local session_name="${tmux_session}-${server}"
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

    local pids=()
    for server in $(get_unique_servers); do
        (
            validate_server "$server"
            local ssh_host remote_repo conda_env
            ssh_host=$(get_server_field "$server" "ssh_host")
            remote_repo=$(get_server_field "$server" "remote_repo")
            conda_env=$(get_server_field "$server" "conda_env")

            local methods
            methods=$(methods_for_server "$server")
            local methods_arr=($methods)

            # Build GPU map JSON
            local gpu_json="{"
            local first=true
            for m in "${methods_arr[@]}"; do
                local gpu
                gpu=$(gpu_for_method "$m")
                if [[ "$first" == true ]]; then
                    first=false
                else
                    gpu_json+=","
                fi
                gpu_json+="\"$m\":$gpu"
            done
            gpu_json+="}"

            echo "[$server] Generating batch_spec..."
            local spec_file="/tmp/usfl_spec_${server}_$$.json"
            python3 -m experiment_core.generate_spec \
                --config-dir experiment_configs \
                --methods "${methods_arr[@]}" \
                --gpu-map "$gpu_json" \
                --output "$spec_file"

            echo "[$server] Transferring spec to $ssh_host..."
            scp -q "$spec_file" "$ssh_host:$remote_repo/batch_spec.json"
            rm -f "$spec_file"

            echo "[$server] Starting experiments..."
            local session_name="${tmux_session}-${server}"

            # Kill existing session if present
            # shellcheck disable=SC2029
            ssh "$ssh_host" "tmux kill-session -t '$session_name' 2>/dev/null || true"

            # Sync branch and start
            # shellcheck disable=SC2029
            ssh "$ssh_host" "cd $remote_repo && git fetch -q && git checkout $branch && git pull -q origin $branch && \
                tmux new-session -d -s '$session_name' \
                'bash remote_run.sh $conda_env batch_spec.json 2>&1 | tee experiment.log'"

            echo "[$server] Started: ${methods_arr[*]}"
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
        echo "Some servers failed to start. Check output above."
    else
        echo "=========================================="
        echo "  All experiments started!"
        echo "=========================================="
        echo ""
        echo "  Monitor:   ./deploy.sh status"
        echo "  Logs:      ./deploy.sh logs <server> [method]"
        echo "  Attach:    ./deploy.sh attach <server>"
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
        local session_name="${tmux_session}-${server}"

        echo "── $server ($ssh_host) ──"

        # Check tmux session
        # shellcheck disable=SC2029
        local status_str
        status_str=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux has-session -t '$session_name' 2>/dev/null && echo 'RUNNING' || echo 'IDLE'" 2>/dev/null \
            || echo "UNREACHABLE")
        echo "  Status: $status_str"

        if [[ "$status_str" == "RUNNING" ]]; then
            # Show last 3 lines of log
            echo "  Recent log:"
            # shellcheck disable=SC2029
            ssh "$ssh_host" "cd $remote_repo && tail -3 experiment.log 2>/dev/null" 2>/dev/null \
                | while IFS= read -r line; do echo "    $line"; done
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
        # Tail specific method log
        local latest
        # shellcheck disable=SC2029
        latest=$(ssh "$ssh_host" "ls -td $remote_repo/results/*/ 2>/dev/null | head -1")
        if [[ -n "$latest" ]]; then
            echo "Tailing: $latest/logs/${method}.log"
            echo "(Ctrl+C to stop)"
            echo ""
            # shellcheck disable=SC2029
            ssh "$ssh_host" "tail -f '$latest/logs/${method}.log'" 2>/dev/null
        else
            echo "No results directory found."
        fi
    else
        # Tail main experiment log
        echo "Tailing: experiment.log on $server"
        echo "(Ctrl+C to stop)"
        echo ""
        # shellcheck disable=SC2029
        ssh "$ssh_host" "cd $remote_repo && tail -f experiment.log" 2>/dev/null
    fi
}

# ========================= Attach =========================

cmd_attach() {
    require_config
    require_jq

    local server="${1:?Usage: ./deploy.sh attach <server>}"
    validate_server "$server"

    local ssh_host tmux_session
    ssh_host=$(get_server_field "$server" "ssh_host")
    tmux_session=$(get_global_field "tmux_session")
    tmux_session="${tmux_session:-usfl-exp}"
    local session_name="${tmux_session}-${server}"

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

Commands:
  run       실험 실행 (push → pull → tmux에서 실행)
  status    전체 서버 상태 확인
  logs      실시간 로그 보기
  attach    tmux 세션 접속
  collect   결과 수집 (Google Drive 또는 로컬)
  servers   등록된 서버 목록 + GPU 상태

Run Usage:
  ./deploy.sh run usfl@server-a:0 gas@server-b:1
  ./deploy.sh run --server server-a --methods "usfl sfl" --gpus "0 1"
  ./deploy.sh run -i --server server-a          # 대화형 모드
  ./deploy.sh run --no-push usfl@server-a:0     # git push 스킵

Other:
  ./deploy.sh status
  ./deploy.sh logs server-a [method]
  ./deploy.sh attach server-a
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
