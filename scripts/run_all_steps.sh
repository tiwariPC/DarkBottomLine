#!/bin/bash
#
# DarkBottomLine Framework - Complete Workflow Script
#
# Runs the full pipeline end to end, in order:
#   1. eventsel  - Event selection (NanoAOD -> EVENTSELECTION.root, one job per input file)
#   2. dnn       - DNN training (event-selection output -> dnn_model.pt)
#   3. region    - Region analysis (event-selection output -> region plots + ROOT
#                  histograms, with the trained DNN applied as ml_score)
#   4. combine   - Combine limit-setting pipeline (datacard -> merge -> fits -> plots)
#
# By default this launches itself inside a detached tmux session so the run
# survives an SSH disconnect; re-attach with `tmux attach -t darkbottomline`.
# Pass --no-tmux to run in the current shell instead (e.g. already inside
# tmux/screen, or running under Condor/CI where a nested session makes no sense).
#
# Usage:
#   scripts/run_all_steps.sh [--dry-run] [--no-tmux] [--steps eventsel,dnn,region,combine]
#
# Examples:
#   scripts/run_all_steps.sh                        # all 4 steps, in tmux
#   scripts/run_all_steps.sh --steps eventsel        # only event selection
#   scripts/run_all_steps.sh --steps dnn,region      # skip eventsel and combine
#   scripts/run_all_steps.sh --no-tmux --dry-run     # preview commands, current shell
#
# Configure via environment variables (all have defaults below):
#   YEAR, CONFIG, REGIONS_CONFIG, PLOT_CONFIG, DNN_CONFIG, COMBINE_CONFIG,
#   RAW_INPUT_DIR, EVENTSEL_DIR, REGION_PLOTS_DIR, VERSION,
#   DNN_OUTDIR, DNN_PLOT_DIR, XSEC_BKG_JSON, XSEC_SIG_JSON,
#   SIGNAL_SCALE, COMBINE_ERA, COMBINE_STAGES

set -e

ALL_STEPS="eventsel,dnn,region,combine"
DRY_RUN=0
NO_TMUX=0
STEPS="${ALL_STEPS}"

args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --no-tmux) NO_TMUX=1 ;;
        --steps) STEPS="$2"; shift ;;
        --steps=*) STEPS="${1#--steps=}" ;;
        *) args+=("$1") ;;
    esac
    shift
done

step_enabled() {
    # step_enabled <name> -> 0 (yes) if <name> is in the comma-separated STEPS list
    case ",${STEPS}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Validate --steps against known step names up front, so a typo fails fast
# instead of silently skipping everything.
IFS=',' read -ra _requested_steps <<< "${STEPS}"
for _s in "${_requested_steps[@]}"; do
    case ",${ALL_STEPS}," in
        *",${_s},"*) ;;
        *) echo "Unknown step '${_s}' — valid steps: ${ALL_STEPS}" >&2; exit 1 ;;
    esac
done

# Created here (before the tmux relaunch below) so the log file exists the
# instant this command returns, not a few seconds later once the detached
# pane gets around to it. Truncated (not just touch'd) so each new run
# starts with a clean log instead of appending onto whatever a previous
# run left behind — this line runs once in the outer process before either
# relaunching into tmux or falling through to --no-tmux, so it's safe to
# truncate unconditionally (nothing has been logged yet either way).
LOG_DIR=${LOG_DIR:-logs}
LOG_FILE="${LOG_DIR}/run_all_steps.log"
mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

# Re-launch inside a detached tmux session unless already inside one (or
# explicitly opted out with --no-tmux) — $TMUX is set by tmux itself for
# every process running inside any session, so this check is self-relaunch-safe.
if [ "$NO_TMUX" -eq 0 ] && [ -z "${TMUX:-}" ]; then
    if ! command -v tmux &>/dev/null; then
        echo "tmux not found on PATH — install it, or pass --no-tmux to run in this shell." >&2
        exit 1
    fi
    SESSION="darkbottomline"
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux session '${SESSION}' already exists — attach with: tmux attach -t ${SESSION}"
        echo "(or run: tmux kill-session -t ${SESSION}  to clear it first)"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    REPO_DIR="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
    # Invoke via `bash <path>` rather than executing the path directly — the
    # relaunched pane shouldn't depend on the script's execute bit having
    # survived however it got onto this machine (git clone/scp/rsync don't
    # always preserve it, and a bare "permission denied" inside a detached
    # tmux pane is easy to miss).
    RELAUNCH_CMD="$(printf '%q ' bash "${SCRIPT_PATH}" --no-tmux --steps "${STEPS}")"
    [ $DRY_RUN -eq 1 ] && RELAUNCH_CMD="${RELAUNCH_CMD}$(printf '%q ' --dry-run)"
    for _a in "${args[@]}"; do
        RELAUNCH_CMD="${RELAUNCH_CMD}$(printf '%q ' "${_a}")"
    done
    # tmux's new pane runs the given command directly (not an interactive
    # login shell), so conda's `activate` function is never defined unless
    # we source local_setup.sh first — without this, darkbottomline/combine
    # resolve to nothing on PATH inside the session.
    RELAUNCH_CMD="cd $(printf '%q' "${REPO_DIR}") && source local_setup.sh && ${RELAUNCH_CMD}"
    # Keep the pane open after the script exits (success or failure) so an
    # attaching user sees the final log output instead of the session
    # vanishing the moment the command completes.
    RELAUNCH_CMD="${RELAUNCH_CMD}; echo; echo '[pipeline exited — press any key to close this pane]'; read -n 1"
    tmux new-session -d -s "${SESSION}" "${RELAUNCH_CMD}"
    echo "Launched in detached tmux session '${SESSION}'."
    echo "  Attach : tmux attach -t ${SESSION}"
    echo "  Detach : Ctrl-b d (once attached)"
    echo "  Log    : ${LOG_FILE}"
    exit 0
fi

# Configuration
YEAR=${YEAR:-2024}
CONFIG_DIR=${CONFIG_DIR:-configs}
CONFIG=${CONFIG:-${CONFIG_DIR}/${YEAR}.yaml}
REGIONS_CONFIG=${REGIONS_CONFIG:-${CONFIG_DIR}/regions.yaml}
PLOT_CONFIG=${PLOT_CONFIG:-${CONFIG_DIR}/plotting.yaml}
DNN_CONFIG=${DNN_CONFIG:-${CONFIG_DIR}/dnn.yaml}
COMBINE_CONFIG=${COMBINE_CONFIG:-${CONFIG_DIR}/combine.yaml}

RAW_INPUT_DIR=${RAW_INPUT_DIR:-../TestingSamples/NanoAODv15_2024}
EVENTSEL_DIR=${EVENTSEL_DIR:-outputs/eventsel}
REGION_PLOTS_DIR=${REGION_PLOTS_DIR:-outputs/region_plots}
VERSION=${VERSION:-$(date +%Y%m%d)_$(git rev-parse --short HEAD 2>/dev/null || echo local)_${YEAR}}

DNN_OUTDIR=${DNN_OUTDIR:-data/dnn}
DNN_PLOT_DIR=${DNN_PLOT_DIR:-outputs/dnn}
DNN_WEIGHT_BRANCH=${DNN_WEIGHT_BRANCH:-full_event_weight}
DNN_MODEL=${DNN_OUTDIR}/dnn_model.pt

XSEC_BKG_JSON=${XSEC_BKG_JSON:-data/cross-section/xsection_background_run3.json}
XSEC_SIG_JSON=${XSEC_SIG_JSON:-data/cross-section/xsection_signal.json}
SIGNAL_SCALE=${SIGNAL_SCALE:-10}

COMBINE_ERA=${COMBINE_ERA:-full}
COMBINE_STAGES=${COMBINE_STAGES:-all}

# LOG_DIR/LOG_FILE were already set (and the file truncated fresh) above,
# before the tmux relaunch block — re-declaring here is a no-op when
# relaunched inside tmux, and covers the --no-tmux direct-run path too.
LOG_DIR=${LOG_DIR:-logs}
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_all_steps.log}"
mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

handle_error() {
    log "ERROR: $1"
    log "Workflow failed at step: $2"
    exit 1
}

run_step() {
    # run_step <step-label> <command...>
    local label="$1"
    shift
    log "$label"
    if [ $DRY_RUN -eq 1 ]; then
        echo "  $*"
        return 0
    fi
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && handle_error "$label failed (rc=$rc)" "$label"
    return 0
}

log "=========================================="
log "DarkBottomLine pipeline"
log "  Steps           : ${STEPS}"
log "  Year            : ${YEAR}"
log "  Config          : ${CONFIG}"
log "  Raw input dir   : ${RAW_INPUT_DIR}"
log "  Event-sel dir   : ${EVENTSEL_DIR}"
log "  Region-plots dir: ${REGION_PLOTS_DIR}"
log "  Version         : ${VERSION}"
[ $DRY_RUN -eq 1 ] && log "  Mode            : DRY RUN"
log "=========================================="

# Step 1: Event selection (NanoAOD -> EVENTSELECTION.root, per input file)
if step_enabled eventsel; then
    log "Step 1: Event selection..."
    if [ $DRY_RUN -eq 1 ]; then
        "$(dirname "$0")/run_eventsel_all.sh" "${RAW_INPUT_DIR}" "${EVENTSEL_DIR}" "${CONFIG}" --dry-run
    else
        "$(dirname "$0")/run_eventsel_all.sh" "${RAW_INPUT_DIR}" "${EVENTSEL_DIR}" "${CONFIG}" \
            2>&1 | tee -a "${LOG_FILE}"
        rc=${PIPESTATUS[0]}
        [ $rc -ne 0 ] && handle_error "Event selection failed (rc=$rc)" "Step 1"
    fi
    log "Event selection completed"
else
    log "Step 1: Event selection... SKIPPED (not in --steps)"
fi

# Step 2: DNN training
if step_enabled dnn; then
    run_step "Step 2: Training DNN model..." \
        darkbottomline train-dnn \
        --dnn-config "${DNN_CONFIG}" \
        --input "${EVENTSEL_DIR}" \
        --weight-branch "${DNN_WEIGHT_BRANCH}" \
        --outdir "${DNN_OUTDIR}" \
        --plot-dir "${DNN_PLOT_DIR}" \
        --xsection-signal-json "${XSEC_SIG_JSON}" \
        --xsection-json "${XSEC_BKG_JSON}"
    log "DNN training completed"
else
    log "Step 2: DNN training... SKIPPED (not in --steps)"
fi

# Step 3: Region analysis (event-selection -> region plots + ROOT histograms,
# with the trained DNN applied)
if step_enabled region; then
    run_step "Step 3: Running region analysis + region plots..." \
        darkbottomline analyze \
        --mode region-analysis \
        --config "${CONFIG}" \
        --regions-config "${REGIONS_CONFIG}" \
        --input "${EVENTSEL_DIR}" \
        --output-dir "${REGION_PLOTS_DIR}" \
        --version "${VERSION}" \
        --xsection-json "${XSEC_BKG_JSON}" \
        --plot-config "${PLOT_CONFIG}" \
        --make-region-plots \
        --apply-dnn --dnn-model "${DNN_MODEL}" --dnn-config "${DNN_CONFIG}" \
        --xsection-signal-json "${XSEC_SIG_JSON}" --signal-scale "${SIGNAL_SCALE}"
    log "Region analysis completed"
else
    log "Step 3: Region analysis... SKIPPED (not in --steps)"
fi

# Step 4: Full Combine limit-setting pipeline (datacard -> merge -> fits -> plots)
if step_enabled combine; then
    run_step "Step 4: Running full Combine pipeline (run-all)..." \
        darkbottomline run-all \
        --combine-config "${COMBINE_CONFIG}" \
        --era "${COMBINE_ERA}" \
        --stages ${COMBINE_STAGES}
    log "Combine pipeline completed"
else
    log "Step 4: Combine pipeline... SKIPPED (not in --steps)"
fi

log "=========================================="
log "DarkBottomLine pipeline completed successfully!"
log "=========================================="
log "  Steps run       : ${STEPS}"
log "  Event-selection : ${EVENTSEL_DIR}"
log "  DNN model       : ${DNN_MODEL}"
log "  Region plots    : ${REGION_PLOTS_DIR}/${VERSION}"
log "  Combine outputs : outputs/combine/"
log "  Log             : ${LOG_FILE}"
log "=========================================="
