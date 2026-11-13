#!/bin/bash
# Fit diagnostics of the era combination on one mass point: GoF (500 toys, single core: the slowest), impacts, pulls, pre/post-fit.
# usage: run_diagnostics.sh CONFIG        CONFIG from make_combine_config.py with --points <one point>
# The first run builds the workspace (merge..merge-eras); impacts and pulls then run in parallel with GoF.
# pre/post-fit reads fitDiagnostics_CRonly.root written by the pulls stage, so it starts after pulls.
C=$1; cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
LOG=$(dirname "$C")/logs; mkdir -p $LOG
darkbottomline run-all --combine-config $C --era full --stages merge-emu normalize-pdf datacard merge-region merge-categories merge-eras > $LOG/build.log 2>&1 || exit 1
darkbottomline run-all --combine-config $C --era full --stages gof > $LOG/gof.log 2>&1 &
darkbottomline run-all --combine-config $C --era full --stages impacts > $LOG/impacts.log 2>&1 &
( darkbottomline run-all --combine-config $C --era full --stages pulls > $LOG/pulls.log 2>&1
  darkbottomline run-all --combine-config $C --era full --stages prefit-postfit > $LOG/prefit_postfit.log 2>&1 ) &
wait; echo "DIAGNOSTICS DONE $(date)"
