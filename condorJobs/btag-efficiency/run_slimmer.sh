#!/bin/sh
# ---------------------------------------------------------------------------
# Condor executable: b-tag efficiency map for ONE ROOT file (the ProcId'th line
# of TXTFILE), written to:
#   <OUTDIR>/<txtstem>/<txtstem>_<ClusterId>_<ProcId>.root
#
# Args (from submit.sub):
#   $1  PROXY      x509 proxy filename in the job sandbox (shipped via
#                  transfer_input_files); sets X509_USER_PROXY for XRootD reads
#   $2  REPO_DIR   absolute path to the DarkBottomLine checkout (shared FS)
#   $3  CONFIG     year YAML, e.g. configs/2024.yaml
#   $4  OUTDIR     output directory for btag-efficiency ROOTs (AFS/EOS-visible)
#   $5  TXTFILE    the samplelist .txt (on shared FS; job reads line ProcId+1)
#   $6  PROCID     0-based job index -> selects the file in TXTFILE
#   $7  CLUSTERID  condor ClusterId (unique per submission, shared by all ProcIds)
# ---------------------------------------------------------------------------
ulimit -s unlimited
set -e

PROXY="$1"
REPO_DIR="$2"
CONFIG="$3"
OUTDIR="$4"
TXTFILE="$5"
PROCID="$6"
CLUSTERID="$7"

# Grid proxy shipped into the sandbox: point XRootD at it (relative to CWD, the
# sandbox, before we cd into the repo).
export X509_USER_PROXY="$(pwd)/${PROXY}"

echo "=== btag-efficiency job ==="
echo "host      : $(hostname)"
echo "proxy     : ${X509_USER_PROXY}"
echo "repo      : ${REPO_DIR}"
echo "config    : ${CONFIG}"
echo "outdir    : ${OUTDIR}"
echo "txtfile   : ${TXTFILE}"
echo "procid    : ${PROCID}"
echo "clusterid : ${CLUSTERID}"
echo "start     : $(date)"

cd "${REPO_DIR}"

# Environment — mirror condorJobs/met_trigger/run_skim.sh (the known-working
# setup): source the LCG view, then set PYTHONPATH directly to the repo's
# .local site-packages. The LCG setup.sh references an unset $COMPILER, so
# disable nounset around it.
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh"
set +u
if [ -f "${LCG_SETUP}" ]; then
    echo "Sourcing LCG environment..."
    source "${LCG_SETUP}"
else
    echo "⚠ Warning: LCG setup not found: ${LCG_SETUP}. Continuing anyway..."
fi

LOCAL_DIR="${REPO_DIR}/.local"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.9")
SITE_PACKAGES_DIR="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"
if [ -d "${SITE_PACKAGES_DIR}" ]; then
    export PYTHONPATH="${SITE_PACKAGES_DIR}:${REPO_DIR}:${PYTHONPATH}"
    echo "✓ Set PYTHONPATH: ${SITE_PACKAGES_DIR}:${REPO_DIR}"
else
    echo "⚠ Warning: .local not found at ${SITE_PACKAGES_DIR}; deps may be missing."
    export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
fi

# Confirm the shipped proxy is valid (non-fatal log; XRootD needs it).
voms-proxy-info -all -file "${X509_USER_PROXY}" 2>/dev/null || \
    echo "⚠ Warning: could not read proxy ${X509_USER_PROXY}; XRootD reads may fail"

if [ ! -f "${TXTFILE}" ]; then
    echo "✗ Error: Sample file not found: ${TXTFILE}" >&2
    exit 1
fi

INPUT=$(grep -v '^#' "${TXTFILE}" | grep -v '^[[:space:]]*$' | sed -n "$((PROCID + 1))p")
if [ -z "${INPUT}" ]; then
    TOTAL_FILES=$(grep -v '^#' "${TXTFILE}" | grep -v '^[[:space:]]*$' | wc -l)
    echo "✗ Error: ProcId ${PROCID} exceeds number of files in ${TXTFILE} (${TOTAL_FILES} files)" >&2
    exit 1
fi

TXTSTEM=$(basename "${TXTFILE}" .txt)
OUTPUT="${OUTDIR}/${TXTSTEM}/${TXTSTEM}_${CLUSTERID}_${PROCID}.root"
mkdir -p "$(dirname "${OUTPUT}")"

echo "input     : ${INPUT}"
echo "output    : ${OUTPUT}"

if [ ! -f "${CONFIG}" ]; then
    echo "✗ Error: Configuration file not found: ${CONFIG}" >&2
    exit 1
fi

CMD="python3 scripts/btag_efficiency_slimmer.py --config ${CONFIG} --input ${INPUT} --output ${OUTPUT}"
echo "command   : ${CMD}"

START_TIME=$(date +%s)
if eval "${CMD}"; then
    DURATION=$(( $(date +%s) - START_TIME ))
    echo "✓ Completed in ${DURATION}s: ${OUTPUT}"
    exit 0
else
    EXIT_CODE=$?
    DURATION=$(( $(date +%s) - START_TIME ))
    echo "✗ Failed after ${DURATION}s, exit code ${EXIT_CODE}" >&2
    exit "${EXIT_CODE}"
fi
