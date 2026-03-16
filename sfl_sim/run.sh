#!/usr/bin/env bash
set -euo pipefail
#
# sfl_sim deploy tool
#
# Usage:
#   ./sfl_sim/run.sh launch <config.json> <server>:<gpu> [--no-push]
#   ./sfl_sim/run.sh status [<server>]
#   ./sfl_sim/run.sh attach <server> [<keyword>]
#   ./sfl_sim/run.sh logs <server> [<keyword>]
#   ./sfl_sim/run.sh kill [<server>] [<keyword>]
#   ./sfl_sim/run.sh collect [<server>]
#
# Examples:
#   ./sfl_sim/run.sh launch experiments/usfl_cifar10.json xsailor7:2
#   ./sfl_sim/run.sh status
#   ./sfl_sim/run.sh attach xsailor7
#   ./sfl_sim/run.sh logs xsailor7 usfl
#   ./sfl_sim/run.sh collect xsailor7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SERVERS_JSON="$REPO_ROOT/deploy/deploy_servers.json"
SESSION_PREFIX="sfl-sim"

# ── Helpers ──────────────────────────────────────────────────────────

die() { echo "Error: $*" >&2; exit 1; }

require_jq() {
    command -v jq &>/dev/null || die "jq required: brew install jq"
}

get_server() {
    local server="$1" field="$2"
    jq -r ".servers[\"$server\"].$field // empty" "$SERVERS_JSON"
}
get_ssh()  { get_server "$1" "ssh_host"; }
get_repo() { get_server "$1" "remote_repo"; }
get_env()  { get_server "$1" "conda_env"; }

validate_server() {
    local ssh_host
    ssh_host=$(get_ssh "$1")
    [[ -n "$ssh_host" ]] || die "Unknown server: $1 (check deploy/deploy_servers.json)"
}

list_sessions() {
    local ssh_host="$1"
    ssh -o ConnectTimeout=5 "$ssh_host" \
        "tmux ls 2>/dev/null | grep '^${SESSION_PREFIX}-' | cut -d: -f1" 2>/dev/null || true
}

all_servers() {
    jq -r '.servers | keys[]' "$SERVERS_JSON"
}

# ── launch ───────────────────────────────────────────────────────────

cmd_launch() {
    local config_file="" target="" no_push=false

    for arg in "$@"; do
        case "$arg" in
            --no-push) no_push=true ;;
            *:*)       target="$arg" ;;
            *)         [[ -z "$config_file" ]] && config_file="$arg" ;;
        esac
    done

    [[ -f "$config_file" ]] || die "Config file not found: $config_file"
    [[ "$target" =~ ^([a-zA-Z0-9_-]+):([0-9]+)$ ]] || \
        die "Target format: <server>:<gpu>  (e.g. xsailor7:2)"

    local server="${BASH_REMATCH[1]}"
    local gpu="${BASH_REMATCH[2]}"

    require_jq
    validate_server "$server"

    local ssh_host repo conda_env method ts run_id session log_file
    ssh_host=$(get_ssh "$server")
    repo=$(get_repo "$server")
    conda_env=$(get_env "$server")
    method=$(jq -r '.method // "sfl"' "$config_file")
    ts=$(date +%Y%m%d_%H%M%S)
    run_id="${ts: -6}"
    session="${SESSION_PREFIX}-${method}-g${gpu}-${run_id}"
    log_file="sfl_sim_${method}_g${gpu}_${ts}.log"

    # 1. Git push
    local branch
    branch=$(git -C "$REPO_ROOT" branch --show-current)

    if [[ "$no_push" == false ]]; then
        echo "==> Pushing to $branch..."
        git -C "$REPO_ROOT" push origin "$branch" 2>&1 | tail -1
    fi

    # 2. Git sync on server
    echo "==> Syncing $server..."
    ssh -o ConnectTimeout=5 "$ssh_host" \
        "cd $repo && git fetch -q origin && git reset --hard origin/$branch -q" \
        || die "Git sync failed on $server"

    # 3. Transfer config
    local remote_config="$repo/experiments/${config_file##*/}"
    ssh "$ssh_host" "mkdir -p $repo/experiments"
    scp -q "$config_file" "$ssh_host:$remote_config"

    # 4. Write run script on remote (avoids shell escaping issues in tmux)
    local remote_script="$repo/_run_${method}_g${gpu}.sh"
    ssh "$ssh_host" "cat > $remote_script" <<REMOTE_EOF
#!/bin/bash
cd $repo

# Conda activation
for prefix in \$HOME/anaconda3 \$HOME/miniconda3 \$HOME/miniforge3 /opt/conda; do
    if [[ -f "\$prefix/etc/profile.d/conda.sh" ]]; then
        source "\$prefix/etc/profile.d/conda.sh"
        break
    fi
done
conda activate $conda_env

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$gpu
python -m sfl_sim --config $remote_config
REMOTE_EOF
    ssh "$ssh_host" "chmod +x $remote_script"

    # 5. Launch in tmux
    echo "==> Launching on $server GPU $gpu..."
    ssh "$ssh_host" \
        "cd $repo && \
         tmux kill-session -t '$session' 2>/dev/null || true; \
         tmux new-session -d -s '$session' \
             'bash $remote_script 2>&1 | tee $log_file'"

    echo ""
    echo "  Session: $session"
    echo "  Server:  $server ($ssh_host) GPU $gpu"
    echo "  Method:  $method"
    echo "  Config:  ${config_file##*/}"
    echo "  Log:     $log_file"
    echo ""
    echo "  Next:"
    echo "    $0 status $server"
    echo "    $0 attach $server"
    echo "    $0 logs $server $method"
}

# ── status ───────────────────────────────────────────────────────────

cmd_status() {
    local filter="${1:-}"
    require_jq

    local servers
    if [[ -n "$filter" ]]; then
        servers="$filter"
    else
        servers=$(all_servers)
    fi

    for server in $servers; do
        local ssh_host
        ssh_host=$(get_ssh "$server")
        [[ -n "$ssh_host" ]] || continue

        echo "=== $server ($ssh_host) ==="

        # Active sessions
        local sessions
        sessions=$(list_sessions "$ssh_host")
        if [[ -z "$sessions" ]]; then
            echo "  (no active experiments)"
        else
            while IFS= read -r sess; do
                # Get latest round output
                local last
                last=$(ssh -o ConnectTimeout=5 "$ssh_host" \
                    "tmux capture-pane -t '$sess' -p 2>/dev/null | grep '^\[Round' | tail -1" \
                    2>/dev/null || true)
                if [[ -n "$last" ]]; then
                    echo "  ▶ $sess  $last"
                else
                    echo "  ▶ $sess  (starting...)"
                fi
            done <<< "$sessions"
        fi

        # GPU utilization
        local gpu_info
        gpu_info=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null || true)
        if [[ -n "$gpu_info" ]]; then
            echo "  GPUs:"
            while IFS=', ' read -r idx util used total; do
                printf "    [%s] %s%% util, %s/%s MiB\n" "$idx" "$util" "$used" "$total"
            done <<< "$gpu_info"
        fi
        echo ""
    done
}

# ── attach ───────────────────────────────────────────────────────────

cmd_attach() {
    local server="${1:-}" keyword="${2:-}"
    [[ -n "$server" ]] || die "Usage: attach <server> [keyword]"

    require_jq
    validate_server "$server"
    local ssh_host
    ssh_host=$(get_ssh "$server")

    if [[ -z "$keyword" ]]; then
        echo "Active sessions on $server:"
        list_sessions "$ssh_host"
        echo ""
        read -rp "Session: " keyword
        [[ -n "$keyword" ]] || return
    fi

    local match
    match=$(ssh -o ConnectTimeout=5 "$ssh_host" \
        "tmux ls 2>/dev/null | grep '$keyword' | head -1 | cut -d: -f1" 2>/dev/null || true)
    [[ -n "$match" ]] || die "No session matching '$keyword' on $server"

    echo "Attaching to $match ... (Ctrl+B D to detach)"
    ssh -t "$ssh_host" "tmux attach-session -t '$match'"
}

# ── logs ─────────────────────────────────────────────────────────────

cmd_logs() {
    local server="${1:-}" keyword="${2:-}"
    [[ -n "$server" ]] || die "Usage: logs <server> [keyword]"

    require_jq
    validate_server "$server"
    local ssh_host repo
    ssh_host=$(get_ssh "$server")
    repo=$(get_repo "$server")

    if [[ -z "$keyword" ]]; then
        echo "Recent logs on $server:"
        ssh -o ConnectTimeout=5 "$ssh_host" \
            "cd $repo && ls -t sfl_sim_*.log 2>/dev/null | head -10" || true
        echo ""
        read -rp "Log file or keyword: " keyword
        [[ -n "$keyword" ]] || return
    fi

    local log_file
    log_file=$(ssh -o ConnectTimeout=5 "$ssh_host" \
        "cd $repo && ls -t sfl_sim_*${keyword}*.log 2>/dev/null | head -1" 2>/dev/null || true)
    [[ -n "$log_file" ]] || die "No log matching '$keyword' on $server"

    echo "Tailing $log_file ... (Ctrl+C to stop)"
    ssh "$ssh_host" "cd $repo && tail -f $log_file"
}

# ── kill ─────────────────────────────────────────────────────────────

cmd_kill() {
    local server="${1:-}" keyword="${2:-}"
    require_jq

    # No args: kill all everywhere
    if [[ -z "$server" ]]; then
        echo "Kill ALL sfl-sim experiments on ALL servers?"
        read -rp "[y/N] " yn
        [[ "$yn" =~ ^[yY] ]] || return

        for s in $(all_servers); do
            local h
            h=$(get_ssh "$s")
            [[ -n "$h" ]] || continue
            local sessions
            sessions=$(list_sessions "$h")
            [[ -z "$sessions" ]] && continue
            while IFS= read -r sess; do
                ssh -o ConnectTimeout=5 "$h" "tmux kill-session -t '$sess' 2>/dev/null" || true
                echo "  killed $sess on $s"
            done <<< "$sessions"
        done
        return
    fi

    validate_server "$server"
    local ssh_host
    ssh_host=$(get_ssh "$server")

    if [[ -z "$keyword" ]]; then
        local sessions
        sessions=$(list_sessions "$ssh_host")
        if [[ -z "$sessions" ]]; then
            echo "No active sessions on $server"
            return
        fi
        echo "Active sessions on $server:"
        echo "$sessions"
        echo ""
        read -rp "Kill which? (name/keyword or 'all'): " keyword
        [[ -n "$keyword" ]] || return
    fi

    if [[ "$keyword" == "all" ]]; then
        local sessions
        sessions=$(list_sessions "$ssh_host")
        while IFS= read -r sess; do
            ssh -o ConnectTimeout=5 "$ssh_host" "tmux kill-session -t '$sess' 2>/dev/null" || true
            echo "  killed $sess"
        done <<< "$sessions"
    else
        local match
        match=$(ssh -o ConnectTimeout=5 "$ssh_host" \
            "tmux ls 2>/dev/null | grep '$keyword' | head -1 | cut -d: -f1" 2>/dev/null || true)
        [[ -n "$match" ]] || die "No session matching '$keyword'"
        ssh -o ConnectTimeout=5 "$ssh_host" "tmux kill-session -t '$match'" || true
        echo "  killed $match"
    fi
}

# ── collect ──────────────────────────────────────────────────────────

cmd_collect() {
    local server="${1:-}"
    require_jq

    if [[ -z "$server" ]]; then
        echo "Collecting from all servers..."
        for s in $(all_servers); do
            cmd_collect "$s"
        done
        return
    fi

    validate_server "$server"
    local ssh_host repo
    ssh_host=$(get_ssh "$server")
    repo=$(get_repo "$server")

    # Check if results exist
    local count
    count=$(ssh -o ConnectTimeout=5 "$ssh_host" \
        "ls $repo/results/result_*.json 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    count=$(echo "$count" | tr -d '[:space:]')

    if [[ "$count" == "0" ]]; then
        echo "  $server: no results"
        return
    fi

    echo "  $server: $count result file(s) — syncing..."
    mkdir -p "$REPO_ROOT/results"
    rsync -avz --progress \
        "$ssh_host:$repo/results/" \
        "$REPO_ROOT/results/" 2>/dev/null
    echo "  Done → ./results/"
}

# ── main ─────────────────────────────────────────────────────────────

cmd="${1:-help}"
shift 2>/dev/null || true

case "$cmd" in
    launch|run)  cmd_launch "$@" ;;
    status|st)   cmd_status "$@" ;;
    attach|at)   cmd_attach "$@" ;;
    logs|log|l)  cmd_logs "$@" ;;
    kill|k)      cmd_kill "$@" ;;
    collect|col) cmd_collect "$@" ;;
    help|--help|-h|*)
        cat <<'USAGE'
sfl_sim deploy tool

Commands:
  launch <config.json> <server>:<gpu> [--no-push]
      Push code, sync server, start experiment in tmux

  status [<server>]
      Show running experiments and GPU utilization

  attach <server> [<keyword>]
      Attach to tmux session (Ctrl+B D to detach)

  logs <server> [<keyword>]
      Tail experiment log file

  kill [<server>] [<keyword>]
      Kill experiment session(s)

  collect [<server>]
      Rsync results to local ./results/

Examples:
  ./sfl_sim/run.sh launch experiments/usfl_cifar10.json xsailor7:2
  ./sfl_sim/run.sh status
  ./sfl_sim/run.sh attach xsailor7
  ./sfl_sim/run.sh logs xsailor7 usfl
  ./sfl_sim/run.sh kill xsailor7
  ./sfl_sim/run.sh collect
USAGE
        ;;
esac
