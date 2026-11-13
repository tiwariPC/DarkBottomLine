#!/bin/bash
# Expected limits of the era combination and of each year, with the mass points split over several parallel processes.
# usage: run_limits_parallel.sh CONFIG [CHUNKS_COMBINATION=8] [CHUNKS_PER_YEAR=4] [--build]
#   --build  first run the card stages (merge-emu ... merge-eras); without it the cards of a previous run in CONFIG's output dir are used
# Plots: <base_dir>/../plots (with the Run 2 curves if RUN2_FILE is set).
# Each process runs `run-all --stages limit --mass-points <subset>`; the summary tables are then collected once from all points
# (the per-process summaries only hold their own subset). Output: <base_dir>/limits/limits_bbDM_<2024|2025|2026|run3>.{txt,root}
C=$1; NC=${2:-8}; NY=${3:-4}; BUILD=$4
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
LOG=$(dirname "$C")/../logs; mkdir -p $LOG
read -r -a PTS <<< "$(python - "$C" <<'PY'
import sys, yaml
print(" ".join(yaml.safe_load(open(sys.argv[1]))["signal_grid"]["points"]))
PY
)"
BASE=$(python -c "import yaml,sys; print(yaml.safe_load(open('$C'))['output']['base_dir'])")
YEARS=$(python -c "import yaml; print(' '.join(str(e['year']) for e in yaml.safe_load(open('$C'))['eras']))")
if [ "$BUILD" = "--build" ]; then
  echo "== cards start $(date)"
  darkbottomline run-all --combine-config $C --era full \
    --stages merge-emu normalize-pdf datacard merge-region merge-categories merge-eras > $LOG/cards.log 2>&1 || { echo "card stages failed"; exit 1; }
fi
chunked() {   # $1 = era, $2 = number of chunks, $3 = tag for logs
  for ((k=0; k<$2; k++)); do
    sub=(); for ((i=k; i<${#PTS[@]}; i+=$2)); do sub+=("${PTS[$i]}"); done
    [ ${#sub[@]} -eq 0 ] && continue
    darkbottomline run-all --combine-config $C --era $1 --stages limit --mass-points "${sub[@]}" > $LOG/limits_$3_$k.log 2>&1 &
  done
}
echo "== limits start $(date)"
chunked full $NC combination
for Y in $YEARS; do chunked $Y $NY $Y; done
wait
echo "== collecting $(date)"
darkbottomline collect-limits --combine-config $C --input $BASE/run3/C --output $BASE/limits --name limits_bbDM_run3 > $LOG/collect_run3.log 2>&1
for Y in $YEARS; do darkbottomline collect-limits --combine-config $C --input $BASE/$Y/C --output $BASE/limits --name limits_bbDM_$Y > $LOG/collect_$Y.log 2>&1; done
# plots next to the combine output (<base_dir>/../plots); set RUN2_FILE=<Run 2 limit table> to add the Run 2 curves
PL=$(dirname $BASE)/plots; R2=${RUN2_FILE:+--run2-file $RUN2_FILE}
python scripts/limits/plot_limits.py years --limits-dir $BASE/limits --out $PL > $LOG/plots.log 2>&1
if [ -n "$RUN2_FILE" ]; then
  python scripts/limits/plot_limits.py years --limits-dir $BASE/limits --out $PL --run2 $R2 --mA 600 >> $LOG/plots.log 2>&1
  for Y in $YEARS; do python scripts/limits/plot_limits.py single --limits-dir $BASE/limits --out $PL --tag $Y $R2 >> $LOG/plots.log 2>&1; done
fi
echo "LIMITS DONE $(date)"
