#!/bin/bash
# ---------------------------------------------------------------------------
# Parametric DNN ("PNN") on the M_H3 = 1500 mass points, full background.
#
# One network takes (MH3, MH4) as extra inputs, so a single model covers the
# whole M_H4 range instead of 18 separate classifiers.  Four mass points are
# held out of training entirely (they are dropped from the signal, never used
# as background) so the trained network can be tested for interpolating to
# masses it has never seen:
#
#   train grid : the 14 remaining MH3=1500 points
#   held out   : MH4 = 100, 300, 700, 1150   (spread over the range; the
#                endpoints 10 and 1450 stay in training so nothing has to be
#                extrapolated at the edges)
#
# Signal: production sample only (BBDM-2HDMa-5f); the BBDM-2HDMa_500 test file
# is refused by the loader.  Background: every event of every background file
# (--event-fraction 1.0), with masses sampled at random from the training grid.
#
# Output: data/dnn_pnn_mh31500/ (+ mass_scaler.json), plots in
#         outputs/dnn_pnn_mh31500/
#
# Usage:  bash scripts/dnn/run_pnn_mh31500.sh
# ---------------------------------------------------------------------------
set -u

REPO="${REPO:-/home/zzq/DarkBottomLine}"
PY="${PY:-/home/zzq/miniconda3/envs/darkbottomline/bin/python}"
INPUT="${INPUT:-/home/zzq/eventsel-merged}"
DNN_CFG="${DNN_CFG:-configs/dnn_parametric.yaml}"
OUTDIR="${OUTDIR:-data/dnn_pnn_mh31500}"
PLOTDIR="${PLOTDIR:-outputs/dnn_pnn_mh31500}"
EVENT_FRACTION="${EVENT_FRACTION:-1.0}"
MASSPOINT_PATTERN="${MASSPOINT_PATTERN:-MH3_1500_}"
HOLDOUT="${HOLDOUT:-MH3_1500_MH4_100_Mchi_1,MH3_1500_MH4_300_Mchi_1,MH3_1500_MH4_700_Mchi_1,MH3_1500_MH4_1150_Mchi_1}"

export LD_LIBRARY_PATH="/home/zzq/miniconda3/envs/darkbottomline/lib:${LD_LIBRARY_PATH:-}"
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
    --event-fraction "$EVENT_FRACTION" \
    --masspoint-pattern "$MASSPOINT_PATTERN" \
    --exclude-masspoints "$HOLDOUT"
