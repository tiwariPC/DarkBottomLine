#!/bin/bash
# ---------------------------------------------------------------------------
# Submit one condor CLUSTER per samplelist .txt, with one JOB per BATCH-sized
# slice of ROOT files (ProcId picks the slice). So a .txt with 594 files and
# BATCH=50 -> a cluster of ceil(594/50)=12 jobs.
#
# Reads the joblist built by make_joblist.sh (lines: "<kind> <txtpath>"),
# counts the ROOT lines in each .txt, computes NJOBS=ceil(NFILES/BATCH), and calls
# condor_submit once per .txt, passing KIND / TXTFILE / BATCH / NJOBS via -append.
#
# Usage:
#   condorJobs/met_trigger/submit_all.sh [joblist] [batch]
#   env override: BATCH=100 condorJobs/met_trigger/submit_all.sh
#   (defaults: joblist=condorJobs/met_trigger/joblist.txt, batch=50)
# ---------------------------------------------------------------------------
set -euo pipefail

SUBDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBFILE="${SUBDIR}/submit.sub"
JOBLIST="${1:-${SUBDIR}/joblist.txt}"
# BATCH: env var wins, else 2nd positional arg, else default 50.
BATCH="${BATCH:-${2:-50}}"

[[ -f "${JOBLIST}" ]] || { echo "ERROR: joblist not found: ${JOBLIST}" >&2; exit 1; }
[[ "${BATCH}" -ge 1 ]] || { echo "ERROR: BATCH must be >= 1 (got ${BATCH})" >&2; exit 1; }

# Logs live next to these scripts (${SUBDIR}/logs), created here if missing. The
# absolute LOGDIR is passed to submit.sub so condor writes there regardless of the
# CWD condor_submit is invoked from — no hardcoded paths, no cd required.
LOGDIR="${SUBDIR}/logs"
mkdir -p "${LOGDIR}"
echo "Batch size: ${BATCH} files/job"

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
    # Count ROOT lines (skip comments + blanks); NJOBS = ceil(NFILES / BATCH).
    NFILES=$(grep -v '^#' "${TXTFILE}" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
    if [[ "${NFILES}" -eq 0 ]]; then
        echo "WARNING: no ROOT files in ${TXTFILE}, skipping" >&2
        continue
    fi
    NJOBS=$(( (NFILES + BATCH - 1) / BATCH ))
    echo "Submitting ${NJOBS} jobs (${NFILES} files / ${BATCH}) for ${KIND}: $(basename "${TXTFILE}")"
    condor_submit "${SUBFILE}" \
        -append "KIND=${KIND}" \
        -append "TXTFILE=${TXTFILE}" \
        -append "BATCH=${BATCH}" \
        -append "NJOBS=${NJOBS}" \
        -append "LOGDIR=${LOGDIR}" \
        -append "USER_INITIAL=${USER:0:1}" </dev/null
    n_clusters=$((n_clusters + 1))
done

echo "Done: submitted ${n_clusters} clusters."
