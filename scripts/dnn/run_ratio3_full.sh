#!/bin/bash
# ---------------------------------------------------------------------------
# Train one DNN per M_H3/M_H4 bin instead of one per mass point.
#
# The 29 mass points are grouped into 3 contiguous bins of M_H3/M_H4, and each
# bin's signal events (the OR of its mass points' GenModel_* branches) are
# trained against the shared full background.  Per-event weights are unchanged:
# every signal event keeps the cross-section scaling of its own mass point.
#
# Bins (equal count, with round edges):
#   M_H3/M_H4 <= 2      10 mass points   1,033,711 signal events
#   2 < M_H3/M_H4 <= 5  10 mass points     746,386 signal events
#   M_H3/M_H4 > 5        9 mass points     327,453 signal events
#
# Output: data/dnn_ratio3/<bin>/ (+ ratio_bins.json), plots in
#         outputs/dnn_ratio3/<bin>/
#
# Usage:  bash scripts/dnn/run_ratio3_full.sh
#   Override the edges with e.g. RATIO_EDGES="2,6" ... or pass --ratio-bin-edges
#   through the script.  Leave RATIO_EDGES empty for equal-count bins.
# ---------------------------------------------------------------------------
set -u

REPO="${REPO:-/home/zzq/DarkBottomLine}"
PY="${PY:-/home/zzq/miniconda3/envs/darkbottomline/bin/python}"
INPUT="${INPUT:-/home/zzq/eventsel-merged}"
OUTDIR_BASE="${OUTDIR_BASE:-data/dnn_ratio3}"
PLOTDIR_BASE="${PLOTDIR_BASE:-outputs/dnn_ratio3}"
DNN_CFG="${DNN_CFG:-configs/dnn.yaml}"
BKG_FRACTION="${BKG_FRACTION:-1.0}"
RATIO_BINS="${RATIO_BINS:-3}"
RATIO_EDGES="${RATIO_EDGES:-2,5}"

export LD_LIBRARY_PATH="/home/zzq/miniconda3/envs/darkbottomline/lib:${LD_LIBRARY_PATH:-}"

edge_args=()
if [ -n "$RATIO_EDGES" ]; then
    edge_args=(--ratio-bin-edges "$RATIO_EDGES")
fi

cd "$REPO"
exec "$PY" scripts/dnn/train_per_masspoint.py \
    --dnn-config "$DNN_CFG" \
    --input "$INPUT" \
    --weight-branch full_event_weight \
    --xsection-signal-json data/cross-section/xsection_signal.json \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --outdir-base "$OUTDIR_BASE" \
    --plot-dir-base "$PLOTDIR_BASE" \
    --bkg-fraction "$BKG_FRACTION" \
    --ratio-bins "$RATIO_BINS" \
    "${edge_args[@]}"
