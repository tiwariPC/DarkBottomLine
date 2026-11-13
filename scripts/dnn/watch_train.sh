#!/bin/bash
# ---------------------------------------------------------------------------
# Watchdog for the full (single) DNN training run.
#
# Every CHECK_INTERVAL seconds it checks whether the training process is alive:
#   - alive -> reset the consecutive-stop counter
#   - dead  -> if this is already the 2nd consecutive stop, give up (exit);
#              otherwise wait CHECK_INTERVAL seconds and restart the training
#
# "Finished normally" is detected from the training log, so the watchdog exits
# instead of restarting after a successful run.
#
# Note: unlike the per-masspoint training, a single full model has no
# skip-what-is-done resume — a restart begins the training again from scratch.
# The breakage window is normally the multi-minute loading phase, before any
# training happens, so this is still worth having.
#
# Usage:  bash scripts/dnn/watch_train.sh
# ---------------------------------------------------------------------------

set -u

CHECK_INTERVAL=300          # seconds between checks (5 min)
REPO_DIR="${REPO_DIR:-/home/zzq/DarkBottomLine}"
# Defaults describe the current job: the 29 per-masspoint models on full data.
# Override RUN_CMD / FINISHED_PATTERN / LOG_DIR for a different run, e.g.
#   RUN_CMD="bash scripts/dnn/run_dnn_full.sh" LOG_DIR=reports/dnn_full \
#   FINISHED_PATTERN="DNN training complete" PROC_PATTERN=train-dnn \
#   bash scripts/dnn/watch_train.sh
RUN_CMD="${RUN_CMD:-bash ${REPO_DIR}/scripts/dnn/run_masspoint29_full.sh}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/reports/masspoint29}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train.log}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch.log}"
PROC_PATTERN="${PROC_PATTERN:-train_per_masspoint.py}"
FINISHED_PATTERN="${FINISHED_PATTERN:-Done. Trained}"

mkdir -p "$LOG_DIR"

log_msg() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" >> "$WATCH_LOG"; }

training_finished() {
    [ -f "$TRAIN_LOG" ] && grep -qa "$FINISHED_PATTERN" "$TRAIN_LOG"
}

start_training() {
    log_msg "restarting training (log: ${TRAIN_LOG})"
    nohup bash -c "cd ${REPO_DIR} && ${RUN_CMD}" >> "$TRAIN_LOG" 2>&1 &
    sleep 10
}

consecutive_stops=0
log_msg "watchdog started (check every ${CHECK_INTERVAL}s; give up after 2 consecutive stops)"

while true; do
    sleep "$CHECK_INTERVAL"

    if training_finished; then
        log_msg "training finished normally - watchdog exiting"
        break
    fi

    if pgrep -f "$PROC_PATTERN" > /dev/null 2>&1; then
        if [ "$consecutive_stops" -ne 0 ]; then
            log_msg "training is running again (stop counter reset)"
        fi
        consecutive_stops=0
        continue
    fi

    consecutive_stops=$((consecutive_stops + 1))
    log_msg "training not running (consecutive stop #${consecutive_stops})"

    if [ "$consecutive_stops" -ge 2 ]; then
        log_msg "stopped 2 times in a row - giving up, no further restarts"
        break
    fi

    log_msg "waiting ${CHECK_INTERVAL}s before restart"
    sleep "$CHECK_INTERVAL"
    start_training
done
