#!/bin/bash
# ---------------------------------------------------------------------------
# Condor executable: event selection for a slice of ROOT files from one
# samplelist .txt. Files run sequentially, one
# `darkbottomline analyze --mode event-selection` call each, so one bad file
# never costs the rest of the slice. Per file:
#   1. skip if <OUTDIR>/<YEAR>/eventsel/<sample>/<sample>_I<idx>C<ClusterId>P<ProcId>_EVENTSELECTION.root
#      already exists (a held+released retry keeps its ClusterId/ProcId; across
#      submissions submit_samples.py dedups on (sample, idx))
#   2. read via the CERN-local EOS replica when one exists, else the original
#      redirector, then INFN / FNAL fallbacks
#   3. write to worker scratch, verify it opens, xrdcp to EOS with an adler32
#      checksum (EOS FUSE writes from many jobs have silently corrupted files,
#      see condorJobs/btag-efficiency/run_slimmer.sh)
# Exits non-zero if any file failed, so condor holds + retries the job; files
# that made it to EOS are skipped on the retry.
#
# Args (from submit.sub):
#   $1  PROXY_FILE  x509 proxy filename in the job sandbox
#   $2  INDICES     colon-separated 0-based file indices into the .txt
# Env (from submit.sub): DBL_REPO_DIR DBL_CONFIG DBL_YEAR DBL_SAMPLE_TXT
#   DBL_OUTDIR DBL_DATA DBL_WORKERS DBL_CHUNK_SIZE DBL_MAX_EVENTS DBL_SLICE_ID
# ---------------------------------------------------------------------------
ulimit -s unlimited
set -u -o pipefail

PROXY_FILE="$1"
INDICES="$2"

export X509_USER_PROXY="$(pwd)/${PROXY_FILE}"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}"

REPO_DIR="${DBL_REPO_DIR}"
CONFIG="${DBL_CONFIG}"
YEAR="${DBL_YEAR}"
SAMPLE_TXT="${DBL_SAMPLE_TXT}"
OUTDIR_SAMPLE="${DBL_OUTDIR}/${YEAR}/eventsel/$(basename "${DBL_SAMPLE_TXT}" .txt)"
SAMPLE=$(basename "${SAMPLE_TXT}" .txt)
EOS_HOST="eoscms.cern.ch"

# Submission tag for output names, from the job ad HTCondor writes into the sandbox.
job_attr() { sed -n "s/^$1 = //p" "${_CONDOR_JOB_AD:-/dev/null}" 2>/dev/null | head -1; }
CLUSTER_ID=$(job_attr ClusterId)
PROC_ID=$(job_attr ProcId)
CLUSTER_ID="${CLUSTER_ID:-0}"
PROC_ID="${PROC_ID:-0}"

echo "=== event-selection job ==="
echo "host      : $(hostname)"
echo "sample    : ${SAMPLE} (${YEAR}, data=${DBL_DATA})"
echo "slice     : ${DBL_SLICE_ID} -> indices ${INDICES}"
echo "job       : cluster ${CLUSTER_ID}, proc ${PROC_ID}"
echo "outdir    : ${OUTDIR_SAMPLE}"
echo "workers   : ${DBL_WORKERS}, chunk size ${DBL_CHUNK_SIZE}"
echo "start     : $(date)"

cd "${REPO_DIR}" || { echo "✗ Error: cannot cd to ${REPO_DIR}" >&2; exit 1; }

# Environment: LCG view + the repo's .local site-packages (same as
# condorJobs/btag-efficiency/run_slimmer.sh). LCG setup.sh references unset vars.
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh"
set +u
source "${LCG_SETUP}" || { echo "✗ Error: cannot source ${LCG_SETUP}" >&2; exit 1; }
set -u
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="${REPO_DIR}/.local/lib/python${PYTHON_VERSION}/site-packages:${REPO_DIR}:${PYTHONPATH:-}"

voms-proxy-info -file "${X509_USER_PROXY}" -timeleft >/dev/null 2>&1 || \
    echo "⚠ Warning: proxy ${X509_USER_PROXY} unreadable or expired; XRootD reads may fail"

for f in "${SAMPLE_TXT}" "${CONFIG}"; do
    [ -f "${f}" ] || { echo "✗ Error: not found: ${f}" >&2; exit 1; }
done

# Same line filter as submit_samples.py (strip, drop blanks and # comments) so
# indices agree.
mapfile -t FILES < <(sed 's/^[[:space:]]*//; s/[[:space:]]*$//' "${SAMPLE_TXT}" | grep -v '^#' | grep -v '^$')

# Print candidate input URLs for one samplelist entry, best first.
candidates() {
    local url="$1"
    if [[ "${url}" != root://* ]]; then
        echo "${url}"
        return
    fi
    # root://host//store/... -> /store/...
    local lfn="${url#root://*/}"
    while [[ "${lfn}" == /* ]]; do lfn="${lfn#/}"; done
    lfn="/${lfn}"
    if [[ "${lfn}" == /store/* ]]; then
        if xrdfs "${EOS_HOST}" stat "/eos/cms${lfn}" >/dev/null 2>&1; then
            echo "root://${EOS_HOST}//eos/cms${lfn}"
        fi
        echo "${url}"
        for r in xrootd-cms.infn.it cmsxrootd.fnal.gov; do
            [[ "${url}" == root://${r}/* ]] || echo "root://${r}/${lfn}"
        done
    else
        echo "${url}"
    fi
}

# Open the file fresh and read every key's metadata: catches truncated/corrupt output.
verify_root() {
    python3 - "$1" <<'EOF'
import sys
import uproot
try:
    with uproot.open(sys.argv[1]) as f:
        if not f.keys():
            sys.exit("no keys")
        for k in f.keys():
            f[k]
except Exception as e:
    sys.exit(f"verify failed: {e}")
EOF
}

stage_out() {
    local src="$1" dest="$2" attempt
    for attempt in 1 2 3; do
        if xrdcp --nopbar --force --posc --cksum adler32 "${src}" "root://${EOS_HOST}/${dest}"; then
            return 0
        fi
        echo "⚠ xrdcp attempt ${attempt}/3 failed for ${dest}" >&2
        sleep $((attempt * 30))
    done
    return 1
}

xrdfs "${EOS_HOST}" mkdir -p "${OUTDIR_SAMPLE}" >/dev/null 2>&1 || true

N_DONE=0
N_SKIPPED=0
FAILED=()
JOB_START=$(date +%s)

for idx in ${INDICES//:/ }; do
    if [ "${idx}" -ge "${#FILES[@]}" ]; then
        echo "✗ Error: index ${idx} beyond ${#FILES[@]} files in ${SAMPLE_TXT}" >&2
        FAILED+=("${idx}")
        continue
    fi
    NAME="${SAMPLE}_I${idx}C${CLUSTER_ID}P${PROC_ID}_EVENTSELECTION.root"
    DEST="${OUTDIR_SAMPLE}/${NAME}"
    LOCAL_OUT="${SCRATCH}/${NAME}"

    if xrdfs "${EOS_HOST}" stat "${DEST}" >/dev/null 2>&1; then
        echo "→ [${idx}] already on EOS, skipping"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    echo ""
    echo "=== [${idx}] ${FILES[$idx]} ==="
    OK=0
    while read -r SRC; do
        [ -n "${SRC}" ] || continue
        echo "→ input: ${SRC}"
        CMD=(python3 -m darkbottomline.cli analyze --mode event-selection
             --config "${CONFIG}" --input "${SRC}"
             --event-selection-output "${LOCAL_OUT}"
             --executor futures --workers "${DBL_WORKERS}" --chunk-size "${DBL_CHUNK_SIZE}")
        [ -n "${DBL_MAX_EVENTS}" ] && CMD+=(--max-events "${DBL_MAX_EVENTS}")
        [ "${DBL_DATA}" = "true" ] && CMD+=(--data)

        T0=$(date +%s)
        "${CMD[@]}" </dev/null
        RC=$?
        SECONDS_USED=$(( $(date +%s) - T0 ))
        if [ "${RC}" -eq 0 ] && verify_root "${LOCAL_OUT}"; then
            OK=1
            break
        fi
        echo "⚠ [${idx}] failed with ${SRC} (exit ${RC}, ${SECONDS_USED}s)" >&2
        rm -f "${LOCAL_OUT}"
    done < <(candidates "${FILES[$idx]}")

    if [ "${OK}" -eq 1 ] && stage_out "${LOCAL_OUT}" "${DEST}"; then
        N_DONE=$((N_DONE + 1))
        # Machine-readable line parsed by pilot_report.py.
        echo "DBL_TIMING sample=${SAMPLE} year=${YEAR} idx=${idx} seconds=${SECONDS_USED} workers=${DBL_WORKERS} data=${DBL_DATA}"
        echo "✓ [${idx}] -> ${DEST} (${SECONDS_USED}s)"
    else
        echo "✗ [${idx}] FAILED" >&2
        FAILED+=("${idx}")
    fi
    rm -f "${LOCAL_OUT}"
done

echo ""
echo "=== summary: slice ${DBL_SLICE_ID} of ${SAMPLE} ==="
echo "done ${N_DONE}, skipped ${N_SKIPPED}, failed ${#FAILED[@]}, wall $(( $(date +%s) - JOB_START ))s"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "failed indices: ${FAILED[*]}" >&2
    exit 1
fi
exit 0
