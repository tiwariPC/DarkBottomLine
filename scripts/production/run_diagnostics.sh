#!/bin/bash
# Fit diagnostics of the era combination on one mass point: GoF (500 toys), impacts, pulls, pre/post-fit, as parallel jobs.
# usage: run_diagnostics.sh CONFIG [GOF_JOBS=5]       CONFIG from make_combine_config.py with --points <one point>
#  - the card stages run first (merge-emu ... merge-eras);
#  - impacts, pulls (then pre/post-fit, which needs pulls' fitDiagnostics_CRonly.root) and GoF run at the same time;
#  - the GoF toys, one serial process in combine, are split over GOF_JOBS processes with different seeds and merged
#    (the serial toys process started by run-all is stopped as soon as it appears, its command line is reused).
C=$1; K=${2:-5}; cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
LOG=$(dirname "$C")/logs; mkdir -p $LOG
RA="darkbottomline run-all --combine-config $C --era full --stages"
$RA merge-emu normalize-pdf datacard merge-region merge-categories merge-eras > $LOG/build.log 2>&1 || { echo "card stages failed"; exit 1; }
$RA impacts > $LOG/impacts.log 2>&1 &
( $RA pulls > $LOG/pulls.log 2>&1; $RA prefit-postfit > $LOG/prefit_postfit.log 2>&1 ) &
$RA gof > $LOG/gof.log 2>&1 &
GOFPID=$!
# wait for the serial toys process, take its command line and stop it
for i in $(seq 1 600); do
  TP=$(pgrep -f "combine -M GoodnessOfFit .* -n Toys" | head -n 1); [ -n "$TP" ] && break; sleep 2
done
if [ -n "$TP" ]; then
  mapfile -d '' ARGS < /proc/$TP/cmdline; CARD=$(readlink /proc/$TP/cwd); kill $TP; wait $GOFPID 2>/dev/null
  NT=$(printf '%s\n' "${ARGS[@]}" | awk 'p{print;exit} $0=="-t"{p=1}'); PER=$((NT/K))
  rm -f $CARD/higgsCombineToys*.root; TOYPIDS=()
  for ((k=0; k<K; k++)); do
    A=(); for a in "${ARGS[@]}"; do
      case "$a" in "$NT") a=$PER;; --seed=*) a="--seed=$((12345+k))";; Toys) a="Toys$k";; esac; A+=("$a"); done
    ( cd $CARD && "${A[@]}" > toys_$k.log 2>&1 ) &
    TOYPIDS+=($!)
  done
  wait "${TOYPIDS[@]}"      # only the toy jobs: a bare wait would also block on pulls/impacts
  ( cd $CARD && hadd -f higgsCombineToys.GoodnessOfFit.mH120.12345.root higgsCombineToys[0-9]*.GoodnessOfFit.mH120.*.root > hadd.log 2>&1 )
  python - "$C" "$CARD" <<'PY'
import sys, yaml
from pathlib import Path
from darkbottomline.combine_tools import CombineRunner
cfg = yaml.safe_load(open(sys.argv[1])); card = sys.argv[2]; r = CombineRunner(cfg)
mp = Path(card).name; years = [str(e["year"]) for e in cfg["eras"]]
label = years[0] if len(years) == 1 else "+".join([years[0]] + [y[-2:] for y in years[1:]])
j = r.run_collect_goodness_of_fit(f"{card}/higgsCombineObserved.GoodnessOfFit.mH120.root", card, mass_point=mp)
r.run_plot_gof(j, card, algo=cfg["fit"]["options"]["goodness_of_fit"].get("algorithm", "saturated"), title_right=f"S+B hypothesis(1b+2b {label})", mass_point=mp)
PY
fi
wait; echo "DIAGNOSTICS DONE $(date)"
