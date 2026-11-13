#!/bin/bash
# Expected limits: the era combination, then each year alone (the combination's datacards are reused).
# usage: run_limits.sh CONFIG [YEARS...]     CONFIG from make_combine_config.py; default years: 2024 2025 2026
# Stages: merge-emu normalize-pdf datacard merge-region merge-categories merge-eras limit
C=$1; shift; YEARS=${@:-2024 2025 2026}
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
LOG=$(dirname "$C")/../logs; mkdir -p $LOG
echo "== combination start $(date)"
darkbottomline run-all --combine-config $C --era full \
  --stages merge-emu normalize-pdf datacard merge-region merge-categories merge-eras limit > $LOG/limits_combination.log 2>&1
echo "== combination exit=$? $(date)"
for Y in $YEARS; do
  ( darkbottomline run-all --combine-config $C --era $Y --stages limit > $LOG/limits_$Y.log 2>&1; echo "== $Y exit=$? $(date)" ) &
done
wait; echo "LIMITS DONE $(date)"
