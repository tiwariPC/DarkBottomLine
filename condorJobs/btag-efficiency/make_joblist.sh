#!/bin/bash
# ---------------------------------------------------------------------------
# Build the condor joblist for the b-tag efficiency maps: one line per MC
# samplelist .txt (data has no b-tag efficiency map: it's a MC-truth quantity).
#
# Usage:
#   condorJobs/btag-efficiency/make_joblist.sh <era> [joblist_out]
#   e.g. condorJobs/btag-efficiency/make_joblist.sh 2024
#
# Writes condorJobs/btag-efficiency/joblist.txt by default (read by submit_all.sh).
# ---------------------------------------------------------------------------
set -euo pipefail

ERA="${1:?usage: make_joblist.sh <era> [joblist_out]}"
OUT="${2:-condorJobs/btag-efficiency/joblist.txt}"

SLDIR="data/samplelist/${ERA}"
[[ -d "${SLDIR}" ]] || { echo "ERROR: no samplelist dir ${SLDIR}" >&2; exit 1; }

: > "${OUT}"

# MC only: exclude data primary datasets. Signal samples are kept (btag SFs /
# efficiency maps are needed for signal region predictions too).
for f in "${SLDIR}"/*.txt; do
    [[ -e "$f" ]] || continue
    base="$(basename "$f")"
    case "${base}" in
        Muon*|EGamma*|JetMET*) continue ;;
    esac
    echo "mc ${f}" >> "${OUT}"
done

N=$(wc -l < "${OUT}")
echo "Wrote ${OUT} with ${N} jobs (era ${ERA})"
