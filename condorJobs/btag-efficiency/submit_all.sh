#!/bin/bash
# ---------------------------------------------------------------------------
# Submit one condor CLUSTER per samplelist .txt, with one JOB per input ROOT
# file (ProcId picks the file — the slimmer is single-file-in/single-file-out,
# so unlike met_trigger there is no BATCH slicing: NJOBS = NFILES).
#
# Reads the joblist built by make_joblist.sh (lines: "mc <txtpath>"), counts the
# ROOT lines in each .txt, and calls condor_submit once per .txt, passing
# TXTFILE / NJOBS via -append.
#
# Usage:
#   condorJobs/btag-efficiency/submit_all.sh [joblist]
#   (default: joblist=condorJobs/btag-efficiency/joblist.txt)
# ---------------------------------------------------------------------------
set -euo pipefail

SUBDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBFILE="${SUBDIR}/submit.sub"
JOBLIST="${1:-${SUBDIR}/joblist.txt}"

[[ -f "${JOBLIST}" ]] || { echo "ERROR: joblist not found: ${JOBLIST}" >&2; exit 1; }

# Logs live next to these scripts (${SUBDIR}/logs), created here if missing. The
# absolute LOGDIR is passed to submit.sub so condor writes there regardless of the
# CWD condor_submit is invoked from — no hardcoded paths, no cd required.
LOGDIR="${SUBDIR}/logs"
mkdir -p "${LOGDIR}"

# Read the whole joblist into an array FIRST, then loop it. Streaming the file
# through `while read` while calling condor_submit inside the loop is fragile:
# condor_submit reads stdin and can swallow the rest of the file, so only the
# first .txt gets submitted. Slurping up front avoids any shared file descriptor.
mapfile -t JOBLINES < "${JOBLIST}"

n_clusters=0
for line in "${JOBLINES[@]}"; do
    # Split "KIND TXTFILE" (ignore blank / comment lines).
    KIND="${line%%[[:space:]]*}"
    TXTFILE="${line#*[[:space:]]}"
    [[ -z "${KIND}" || "${KIND}" == \#* ]] && continue

    if [[ ! -f "${TXTFILE}" ]]; then
        echo "!!! SKIP (${KIND}): txt NOT FOUND: ${TXTFILE}" >&2
        echo "    (resolved from CWD: $(pwd)) — check the path in the joblist" >&2
        continue
    fi
    NFILES=$(grep -v '^#' "${TXTFILE}" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
    if [[ "${NFILES}" -eq 0 ]]; then
        echo "WARNING: no ROOT files in ${TXTFILE}, skipping" >&2
        continue
    fi
    echo "Submitting ${NFILES} jobs for $(basename "${TXTFILE}")"
    condor_submit "${SUBFILE}" \
        -append "TXTFILE=${TXTFILE}" \
        -append "NJOBS=${NFILES}" \
        -append "LOGDIR=${LOGDIR}" \
        -append "USER_INITIAL=${USER:0:1}" </dev/null
    n_clusters=$((n_clusters + 1))
done

echo "Done: submitted ${n_clusters} clusters."
