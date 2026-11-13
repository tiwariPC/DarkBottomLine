#!/bin/bash
# Region PKLs (chunked) + region plots/ROOT histograms for one year from merged event-selection outputs.
# usage: run_region_year.sh BASE YEAR LUMI_TAG [JOBS] [SET]
#   BASE      production directory; expects BASE/YEAR/{input/eventsel_merged, configs/YEAR.yaml, configs/regions_SET.yaml}
#   LUMI_TAG  e.g. 109p82 (2024), 110p628 (2025), 25p794 (2026); output dir is BASE/YEAR/region_lumi<TAG>_<SET>
# Run from anywhere; the repository is located from this script. Needs the conda env with darkbottomline.
BASE=$1; Y=$2; LT=$3; JOBS=${4:-6}; SET=${5:-incl}
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R" || exit 1
B=$BASE/$Y; IN=$B/input/eventsel_merged; NAME=region_lumi${LT}_$SET
mkdir -p $B/logs $B/$NAME/config_used
cp $B/configs/regions_$SET.yaml $B/$NAME/config_used/regions_used.yaml; cp configs/plotting.yaml $B/$NAME/config_used/plotting_used.yaml
echo "== $Y pkls start $(date)"
python scripts/production/chunked_region_pkls.py --input $IN --out $B/pkls/$SET --config $B/configs/$Y.yaml \
  --regions-config $B/configs/regions_$SET.yaml --xsection-json data/cross-section/xsection_background_run3.json \
  --dnn-model data/dnn/dnn_model.pt --dnn-config configs/dnn.yaml --repo $R --step-size 300000 --jobs $JOBS || { echo "PKL FAILED"; exit 1; }
echo "== $Y regions start $(date)"
darkbottomline analyze --mode region-analysis --config $B/configs/$Y.yaml --regions-config $B/configs/regions_$SET.yaml \
  --input $IN --region-results $B/pkls/$SET/per_sample --output-dir $B --version $NAME \
  --xsection-json data/cross-section/xsection_background_run3.json --plot-config configs/plotting.yaml \
  --make-region-plots --xsection-signal-json data/cross-section/xsection_signal.json --signal-scale 10 > $B/logs/regions_$SET.log 2>&1
echo "== $Y regions exit=$? $(date)"
echo "JOBDONE $Y"
