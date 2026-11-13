#!/bin/bash
# MET (PFMETNoMu) trigger efficiency plots per year: that year's Muon data skims vs the 2024 DY/W MC skims (same MC for all years).
# usage: run_met_trigger_plots.sh SKIM_DIR OUT_DIR [YEARS...]     SKIM_DIR holds <year>_Merged/{Muon*,DYto2L-2Jets_*,WtoLNu-2Jets_*}.root
# --json-out goes next to the plots so the committed data/corrections/*metHlt.json.gz are NOT overwritten.
M=$1; O=$2; shift 2; YEARS=${@:-2024 2025 2026}
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
for Y in $YEARS; do
  mkdir -p $O/$Y; echo "== $Y start $(date)"
  nice -n 10 python scripts/met_trigger/met_trigger_efficiency.py analyze --config configs/$Y.yaml \
    --data-skims "$M/${Y}_Merged/Muon*.root" \
    --mc-skims "$M/2024_Merged/DYto2L-2Jets_*.root" "$M/2024_Merged/WtoLNu-2Jets_*.root" \
    --outdir $O/$Y --json-out $O/$Y/metHlt_${Y}_check.json.gz > $O/$Y/analyze.log 2>&1
  echo "== $Y exit=$? $(date)"
done
echo "MET TRIG DONE"
