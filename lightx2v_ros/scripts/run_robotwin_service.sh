#!/usr/bin/env bash
# =============================================================================
# One-click launcher for the RoboTwin continuous-eval ROS service.
#
# Brings up the three nodes that make up the demo:
#   1. simulator/robotwin_node   - SAPIEN dual-arm sim
#   2. inference/fastwam_node    - FastWAM policy inference
#   3. visualization/image_web_viewer - web dashboard (default 0.0.0.0:6061)
#
# By default the simulator starts in "ready" state and runs a SINGLE episode
# when you press 开始 in the web dashboard (or publish a `start` control
# command). Start/pause/restart and task/scenario switching are all available
# from the dashboard. Set LOOP=true for the old continuous-eval behaviour and
# AUTOSTART=true to begin evaluating immediately on launch.
#
# Usage:
#   run_robotwin_service.sh start     # launch all three nodes (default)
#   run_robotwin_service.sh stop      # stop all nodes
#   run_robotwin_service.sh restart   # stop then start
#   run_robotwin_service.sh status    # show pid / running state + log tails
#   run_robotwin_service.sh logs [sim|fastwam|viewer]   # follow a node log
#
# Everything below is overridable via environment variables, e.g.
#   VIEWER_PORT=8080 TASK_NAME=click_alarmclock ./run_robotwin_service.sh start
# =============================================================================
set -uo pipefail

# ------------------------------- configuration --------------------------------
LIGHTX2V="${LIGHTX2V:-/home/fuhaiwen/LightX2V}"
ROS_WS="${ROS_WS:-$LIGHTX2V/lightx2v_ros}"
ROS_UNDERLAY="${ROS_UNDERLAY:-/root/ros2_lyrical/install/setup.bash}"
ROS_OVERLAY="${ROS_OVERLAY:-$ROS_WS/install/setup.bash}"
# RoboTwin uses relative asset paths (e.g. ./assets/...), so the simulator node must
# run with its working directory set to the vendored RoboTwin root.
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$ROS_WS/src/simulator/simulator/robotwin_node/RoboTwin}"

# Task / embodiment (simulator node params)
TASK_NAME="${TASK_NAME:-lift_pot}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
EMBODIMENT="${EMBODIMENT:-aloha-agilex}"
SEED="${SEED:-0}"
# <=0 means "use RoboTwin's per-task step limit"; hitting the cap = FAILURE.
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
# Single-episode by default; control via the web dashboard.
LOOP="${LOOP:-false}"
AUTOSTART="${AUTOSTART:-false}"
# Skip seeds the scripted expert cannot solve + generate real instructions
# (requires curobo; falls back gracefully when unavailable).
EXPERT_CHECK="${EXPERT_CHECK:-true}"
# Stream an intermediate viewer frame every N physics steps during an action
# (higher FPS in the dashboard at some simulation-speed cost; 0 disables).
RENDER_PUBLISH_EVERY="${RENDER_PUBLISH_EVERY:-15}"

# Policy (fastwam node params)
CONFIG_JSON="${CONFIG_JSON:-$LIGHTX2V/configs/fastwam/robotwin_i2va.json}"
MODEL_PATH="${MODEL_PATH:-/data/nvme7/yongyang/models/Wan-AI/Wan2.2-TI2V-5B}"

# Web viewer
VIEWER_HOST="${VIEWER_HOST:-0.0.0.0}"
VIEWER_PORT="${VIEWER_PORT:-6061}"

# GPU assignment (sim is light, policy is heavy)
SIM_GPU="${SIM_GPU:-5}"
POLICY_GPU="${POLICY_GPU:-4}"

# Runtime dirs
RUN_DIR="${RUN_DIR:-/tmp/robotwin_service}"
LOG_SIM="$RUN_DIR/sim.log"
LOG_FASTWAM="$RUN_DIR/fastwam.log"
LOG_VIEWER="$RUN_DIR/viewer.log"
PID_SIM="$RUN_DIR/sim.pid"
PID_FASTWAM="$RUN_DIR/fastwam.pid"
PID_VIEWER="$RUN_DIR/viewer.pid"

# ------------------------------- helpers --------------------------------------
c_info()  { printf '\033[1;34m[service]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[service]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[service]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[service]\033[0m %s\n' "$*" >&2; }

source_ros() {
    # ROS setup scripts reference unbound vars; disable -u while sourcing them.
    set +u
    # shellcheck disable=SC1090
    source "$ROS_UNDERLAY"
    # shellcheck disable=SC1090
    source "$ROS_OVERLAY"
    set -u
    export PYTHONPATH="$LIGHTX2V:${PYTHONPATH:-}"
}

is_running() {  # $1 = pidfile
    local pf="$1" pid
    [[ -f "$pf" ]] || return 1
    pid="$(cat "$pf" 2>/dev/null)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

# Launch a command in its own process group so we can kill the whole subtree.
# $1=name  $2=logfile  $3=pidfile  $4=command string (run through bash -c)
launch() {
    local name="$1" log="$2" pidfile="$3" cmd="$4"
    if is_running "$pidfile"; then
        c_warn "$name already running (pid $(cat "$pidfile")); skipping"
        return 0
    fi
    c_info "starting $name -> $log"
    setsid bash -c "$cmd" >"$log" 2>&1 &
    echo "$!" >"$pidfile"
    c_ok "$name started (pid $(cat "$pidfile"))"
}

# Wait until $2 (regex) shows up in log $1, or timeout $3 seconds.
wait_for_log() {
    local log="$1" pattern="$2" timeout="${3:-180}" waited=0
    c_info "waiting for '$pattern' in $(basename "$log") (<=${timeout}s)..."
    while (( waited < timeout )); do
        if grep -Eq "$pattern" "$log" 2>/dev/null; then
            c_ok "ready: $(basename "$log")"
            return 0
        fi
        sleep 2; waited=$((waited + 2))
    done
    c_warn "timed out waiting for '$pattern' in $(basename "$log") (continuing anyway)"
    return 1
}

stop_one() {  # $1=name $2=pidfile
    local name="$1" pf="$2" pid
    if ! is_running "$pf"; then
        c_info "$name not running"
        rm -f "$pf"
        return 0
    fi
    pid="$(cat "$pf")"
    c_info "stopping $name (pgid $pid)"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        c_warn "$name did not exit; sending SIGKILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    fi
    rm -f "$pf"
    c_ok "$name stopped"
}

# ------------------------------- commands -------------------------------------
cmd_start() {
    mkdir -p "$RUN_DIR"
    [[ -f "$ROS_UNDERLAY" ]] || { c_err "ROS underlay not found: $ROS_UNDERLAY"; exit 1; }
    [[ -f "$ROS_OVERLAY"  ]] || { c_err "ROS overlay not found: $ROS_OVERLAY";  exit 1; }
    [[ -d "$MODEL_PATH"   ]] || c_warn "MODEL_PATH does not exist: $MODEL_PATH"
    [[ -f "$CONFIG_JSON"  ]] || c_warn "CONFIG_JSON does not exist: $CONFIG_JSON"

    c_info "config: task=$TASK_NAME/$TASK_CONFIG embodiment=$EMBODIMENT seed=$SEED"
    c_info "gpus: sim=$SIM_GPU policy=$POLICY_GPU | viewer=http://$VIEWER_HOST:$VIEWER_PORT"

    # NOTE: PYTHONPATH must use double quotes so ${PYTHONPATH:-} expands *inside* the
    # launched shell (after sourcing ROS), otherwise it clobbers the ros2 paths.
    local ros_prelude
    ros_prelude="set +u; source '$ROS_UNDERLAY'; source '$ROS_OVERLAY'; set -u; export PYTHONPATH=\"$LIGHTX2V:\${PYTHONPATH:-}\";"

    # 1) simulator (continuous loop). cd into RoboTwin root for its relative asset paths.
    launch "simulator" "$LOG_SIM" "$PID_SIM" \
        "$ros_prelude cd '$ROBOTWIN_ROOT'; CUDA_VISIBLE_DEVICES=$SIM_GPU exec ros2 run simulator robotwin_node --ros-args \
            -p task_name:=$TASK_NAME -p task_config:=$TASK_CONFIG -p embodiment:=$EMBODIMENT \
            -p seed:=$SEED -p loop:=$LOOP -p autostart:=$AUTOSTART -p max_episode_steps:=$MAX_EPISODE_STEPS \
            -p expert_check:=$EXPERT_CHECK -p render_publish_every:=$RENDER_PUBLISH_EVERY"
    wait_for_log "$LOG_SIM" "control on" 300 || true

    # 2) web viewer
    launch "viewer" "$LOG_VIEWER" "$PID_VIEWER" \
        "$ros_prelude exec ros2 run visualization image_web_viewer --ros-args \
            -p env:=robotwin -p host:=$VIEWER_HOST -p port:=$VIEWER_PORT"
    wait_for_log "$LOG_VIEWER" "image web viewer on" 60 || true

    # 3) policy (heavy model load; can take a couple of minutes)
    launch "fastwam" "$LOG_FASTWAM" "$PID_FASTWAM" \
        "$ros_prelude CUDA_VISIBLE_DEVICES=$POLICY_GPU exec ros2 run inference fastwam_node --ros-args \
            -p env:=robotwin -p config_json:=$CONFIG_JSON -p model_path:=$MODEL_PATH"
    wait_for_log "$LOG_FASTWAM" "fastwam_node ready" 600 || true

    echo
    c_ok  "RoboTwin service is up."
    c_ok  "Dashboard:   http://$VIEWER_HOST:$VIEWER_PORT  (press 开始 to run an episode)"
    c_info "Follow logs: $0 logs [sim|fastwam|viewer]"
    c_info "Stop:        $0 stop"
    cmd_status
}

cmd_stop() {
    # Stop policy first so it stops feeding actions, then sim, then viewer.
    stop_one "fastwam"   "$PID_FASTWAM"
    stop_one "simulator" "$PID_SIM"
    stop_one "viewer"    "$PID_VIEWER"
}

cmd_status() {
    echo
    printf '%-12s %-10s %s\n' "NODE" "STATE" "PID / LOG"
    for entry in "simulator:$PID_SIM:$LOG_SIM" "fastwam:$PID_FASTWAM:$LOG_FASTWAM" "viewer:$PID_VIEWER:$LOG_VIEWER"; do
        IFS=':' read -r name pf log <<<"$entry"
        if is_running "$pf"; then
            printf '%-12s \033[1;32m%-10s\033[0m %s\n' "$name" "running" "$(cat "$pf")  ($log)"
        else
            printf '%-12s \033[1;31m%-10s\033[0m %s\n' "$name" "stopped" "-  ($log)"
        fi
    done
    echo
}

cmd_logs() {
    local which="${1:-sim}" log
    case "$which" in
        sim|simulator) log="$LOG_SIM" ;;
        fastwam|policy) log="$LOG_FASTWAM" ;;
        viewer|web) log="$LOG_VIEWER" ;;
        *) c_err "unknown log '$which' (use: sim|fastwam|viewer)"; exit 1 ;;
    esac
    [[ -f "$log" ]] || { c_err "log not found: $log"; exit 1; }
    exec tail -n 100 -f "$log"
}

# ------------------------------- dispatch -------------------------------------
case "${1:-start}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; sleep 2; cmd_start ;;
    status)  cmd_status ;;
    logs)    shift; cmd_logs "${1:-sim}" ;;
    *) c_err "usage: $0 {start|stop|restart|status|logs [sim|fastwam|viewer]}"; exit 1 ;;
esac
