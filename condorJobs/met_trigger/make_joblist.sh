#!/bin/bash
# ---------------------------------------------------------------------------
# Build the condor joblist for the MET-trigger skim: one line per samplelist
# .txt, tagged data|mc. Muon* -> data; DYto2L*/WtoLNu-2Jets* -> mc.
#
# Usage:
#   condorJobs/met_trigger/make_joblist.sh <era> [joblist_out]
#   e.g. condorJobs/met_trigger/make_joblist.sh 2024
#
# Writes condorJobs/met_trigger/joblist.txt by default (referenced by submit.sub).
# ---------------------------------------------------------------------------
set -euo pipefail

ERA="${1:?usage: make_joblist.sh <era> [joblist_out]}"
OUT="${2:-condorJobs/met_trigger/joblist.txt}"

SLDIR="data/samplelist/${ERA}"
[[ -d "${SLDIR}" ]] || { echo "ERROR: no samplelist dir ${SLDIR}" >&2; exit 1; }

: > "${OUT}"

# Data: single-muon primary dataset.
for f in "${SLDIR}"/Muon*.txt; do
    [[ -e "$f" ]] || continue
    echo "data ${f}" >> "${OUT}"
done

# MC: every simulated background samplelist. Each MC sample is skimmed for BOTH
# channels and each event is routed to whichever preselection it passes, so DY
# lands mostly in Zmm, W in Wmn, and tt/single-top/diboson contaminate both.
# Exclude the data PDs (Muon/EGamma/JetMET) and the signal / VH-Hbb samples.
for f in "${SLDIR}"/*.txt; do
    [[ -e "$f" ]] || continue
    base="$(basename "$f")"
    # skip data primary datasets
    case "${base}" in
        Muon*|EGamma*|JetMET*) continue ;;
    esac
    # skip signal + SM-Higgs / VH(bb) samples (not backgrounds for this measurement)
    if echo "${base}" | grep -qiE 'BBDM|2HDMa|SMHiggs|Hto2B|-Hto2B|GluGluH|VBFH|ttH|TTH|WminusH|WplusH|ggZH|ZH-'; then
        continue
    fi
    echo "mc ${f}" >> "${OUT}"
done

N=$(wc -l < "${OUT}")
echo "Wrote ${OUT} with ${N} jobs (era ${ERA})"
echo "  data: $(grep -c '^data ' "${OUT}")  mc: $(grep -c '^mc ' "${OUT}")"
