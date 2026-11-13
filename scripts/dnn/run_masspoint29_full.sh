#!/bin/bash
# ---------------------------------------------------------------------------
# Train the 29 per-masspoint DNNs on the FULL data set.
#
# Each mass point gets its own binary classifier (that mass point's signal
# events vs the shared background), but unlike the earlier runs there is no
# background fraction at all: every background file is read in full, and every
# signal file in full.
#
# Input (from /home/zzq/eventsel-merged):
#   signal      1 file ,  2,107,550 events  (BBDM-2HDMa-5f; the
#                                           BBDM-2HDMa_500 test file is
#                                           refused by the loader)
#   background 35 files, 100,882,258 events
#   data       18 files, excluded
#
# The 29 models are trained one after another in a single process (the script
# loops over the GenModel_* branches), so only one model ever holds GPU memory.
#
# Output: data/dnn_masspoint29/<mass point>/ , plots in
#         outputs/dnn_masspoint29/<mass point>/
#
# Usage:  bash scripts/dnn/run_masspoint29_full.sh
# ---------------------------------------------------------------------------
set -u

REPO="${REPO:-/home/zzq/DarkBottomLine}"
PY="${PY:-/home/zzq/miniconda3/envs/darkbottomline/bin/python}"
INPUT="${INPUT:-/home/zzq/eventsel-merged}"
OUTDIR_BASE="${OUTDIR_BASE:-data/dnn_masspoint29}"
PLOTDIR_BASE="${PLOTDIR_BASE:-outputs/dnn_masspoint29}"
DNN_CFG="${DNN_CFG:-configs/dnn.yaml}"
# 1.0 = keep every background event.  The per-masspoint script only subsamples
# when 0 < fraction < 1, so 1.0 means "read the whole file".
BKG_FRACTION="${BKG_FRACTION:-1.0}"

export LD_LIBRARY_PATH="/home/zzq/miniconda3/envs/darkbottomline/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
exec "$PY" scripts/dnn/train_per_masspoint.py \
    --dnn-config "$DNN_CFG" \
    --input "$INPUT" \
    --weight-branch full_event_weight \
    --xsection-signal-json data/cross-section/xsection_signal.json \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --outdir-base "$OUTDIR_BASE" \
    --plot-dir-base "$PLOTDIR_BASE" \
    --bkg-fraction "$BKG_FRACTION"
