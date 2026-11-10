#!/bin/bash
# ---------------------------------------------------------------------------
# Train the single "full" DNN: one binary background-vs-signal classifier over
# the complete event-selection data set — all signal mass points together, no
# per-masspoint split, no background fraction, no event cap.
#
# Input (from /home/zzq/eventsel-merged):
#   signal      1 file ,  2,107,550 events  (BBDM-2HDMa-5f only; the
#                                           BBDM-2HDMa_500 test file is
#                                           refused by the loader)
#   background 35 files, 100,882,258 events
#   data       18 files, excluded
#
# Kept as its own script so the watchdog and a manual run use the exact same
# command line.  Override any of the variables below from the environment.
#
# Usage:  bash scripts/run_dnn_full.sh
# ---------------------------------------------------------------------------
set -u

REPO="${REPO:-/home/zzq/DarkBottomLine}"
PY="${PY:-/home/zzq/miniconda3/envs/darkbottomline/bin/python}"
INPUT="${INPUT:-/home/zzq/eventsel-merged}"
OUTDIR="${OUTDIR:-data/dnn}"
PLOTDIR="${PLOTDIR:-outputs/dnn}"
DNN_CFG="${DNN_CFG:-configs/dnn.yaml}"
# 1.0 = the whole data set.  The CLI default is 0.5 (a random half of every
# file); this run is explicitly the full one.
EVENT_FRACTION="${EVENT_FRACTION:-1.0}"

export LD_LIBRARY_PATH="/home/zzq/miniconda3/envs/darkbottomline/lib:${LD_LIBRARY_PATH:-}"
# Each loader worker holds one file's feature frame and pickles it back to the
# parent process, so the default (one worker per core) peaks far above what
# merging the files actually needs.  Keep it modest for a run this size.
export DNN_LOAD_WORKERS="${DNN_LOAD_WORKERS:-8}"

cd "$REPO"
exec "$PY" -m darkbottomline.cli train-dnn \
    --dnn-config "$DNN_CFG" \
    --input "$INPUT" \
    --signal-prefix BBDM-2HDMa-5f \
    --weight-branch full_event_weight \
    --xsection-signal-json data/cross-section/xsection_signal.json \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --outdir "$OUTDIR" \
    --plot-dir "$PLOTDIR" \
    --event-fraction "$EVENT_FRACTION"
