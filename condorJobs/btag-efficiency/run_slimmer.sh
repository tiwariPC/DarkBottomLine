#!/bin/sh
# ---------------------------------------------------------------------------
# Condor executable: b-tag efficiency map for a BATCH-sized slice of ROOT
# files (a contiguous slice of a samplelist .txt), merged into ONE output:
#   <OUTDIR>/<txtstem>/<txtstem>_<ClusterId>_<ProcId>.root
#
# Job model: one cluster per .txt, one job per BATCH-sized slice (mirrors
# condorJobs/met_trigger/run_skim.sh). ProcId selects the slice = lines
# [ProcId*BATCH+1 .. ProcId*BATCH+BATCH] of TXTFILE. The slimmer's own
# --inputs mode sums that whole slice's raw counts into one output.
#
# Args (from submit.sub):
#   $1  PROXY      x509 proxy filename in the job sandbox (shipped via
#                  transfer_input_files); sets X509_USER_PROXY for XRootD reads
#   $2  REPO_DIR   absolute path to the DarkBottomLine checkout (shared FS)
#   $3  CONFIG     year YAML, e.g. configs/2024.yaml
#   $4  OUTDIR     output directory for btag-efficiency ROOTs (AFS/EOS-visible)
#   $5  TXTFILE    the samplelist .txt (on shared FS; job reads a slice of it)
#   $6  PROCID     0-based job index -> selects the slice of TXTFILE
#   $7  BATCH      number of ROOT files per job (slice size)
#   $8  CLUSTERID  condor ClusterId (unique per submission, shared by all ProcIds)
# ---------------------------------------------------------------------------
ulimit -s unlimited
set -e

PROXY="$1"
REPO_DIR="$2"
CONFIG="$3"
OUTDIR="$4"
TXTFILE="$5"
PROCID="$6"
BATCH="$7"
CLUSTERID="$8"

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
echo "batch     : ${BATCH}"
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

if [ ! -f "${CONFIG}" ]; then
    echo "✗ Error: Configuration file not found: ${CONFIG}" >&2
    exit 1
fi

# This job's slice = ROOT lines [START..END] of TXTFILE (1-based), after
# dropping comment/blank lines (same filter as the slimmer's own .txt
# parser). sed clamps END past EOF, so the last job's slice is naturally short.
START=$((PROCID * BATCH + 1))
END=$((START + BATCH - 1))
SLICE=$(grep -v '^#' "${TXTFILE}" | grep -v '^[[:space:]]*$' | sed -n "${START},${END}p")
if [ -z "${SLICE}" ]; then
    echo "✗ Error: empty slice (lines ${START}-${END}) of ${TXTFILE}" >&2
    exit 4
fi
N_IN_SLICE=$(printf '%s\n' "${SLICE}" | grep -c .)
echo "slice     : lines ${START}-${END} (${N_IN_SLICE} files)"

TXTSTEM=$(basename "${TXTFILE}" .txt)

# Write the slice samplelist to a PRIVATE scratch dir — never the repo root.
# pwd is REPO_DIR (we cd'd there), so writing here would pollute the shared
# checkout and clash with the real data/samplelist files. Named after
# <TXTSTEM>_<ClusterId>_<ProcId> so concurrent/resubmitted jobs never collide.
SCRATCH="${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}"
SLICE_DIR="${SCRATCH}/slice_${TXTSTEM}_${CLUSTERID}_${PROCID}"
mkdir -p "${SLICE_DIR}"
SLICE_STEM="${TXTSTEM}_${CLUSTERID}_${PROCID}"
SLICE_TXT="${SLICE_DIR}/${SLICE_STEM}.txt"
printf '%s\n' "${SLICE}" > "${SLICE_TXT}"

OUTPUT="${OUTDIR}/${TXTSTEM}/${SLICE_STEM}.root"
mkdir -p "$(dirname "${OUTPUT}")"

echo "output    : ${OUTPUT}"

CMD="python3 scripts/btag_efficiency_slimmer.py --config ${CONFIG} --inputs ${SLICE_TXT} --output ${OUTPUT}"
echo "command   : ${CMD}"

# EOS (via eosxd FUSE) has been observed to silently corrupt a small fraction
# of sibling files written concurrently by many condor jobs into the same
# output directory: the slimmer's own `with uproot.recreate(...)` block exits
# cleanly (no exception, "Wrote <path>" printed), but a subset of histogram
# keys in the file on disk decompress to garbage afterwards (seen: both
# "unrecognized compression algorithm" and "invalid bit length repeat" —
# different garbage each time, ruling out a slimmer logic bug). Since the
# writing process itself can't detect this, verify by re-opening the file
# fresh (new process, forces a real read from EOS, not a page-cache echo of
# what we just wrote) and decompressing every histogram key. A bad write
# can only be fixed by regenerating it, so retry the whole slimmer run.
MAX_ATTEMPTS=3
ATTEMPT=1
while [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; do
    START_TIME=$(date +%s)
    if ! eval "${CMD}"; then
        EXIT_CODE=$?
        DURATION=$(( $(date +%s) - START_TIME ))
        echo "✗ Slimmer failed after ${DURATION}s (attempt ${ATTEMPT}/${MAX_ATTEMPTS}), exit code ${EXIT_CODE}" >&2
        ATTEMPT=$((ATTEMPT + 1))
        continue
    fi
    DURATION=$(( $(date +%s) - START_TIME ))

    if python3 -c "
import sys
import uproot
try:
    f = uproot.open('${OUTPUT}')
    keys = f.keys()
    if not keys:
        print('no keys found', file=sys.stderr)
        sys.exit(1)
    for k in keys:
        f[k].values()
except Exception as e:
    print(f'verify failed on key read: {e}', file=sys.stderr)
    sys.exit(1)
"; then
        echo "✓ Completed in ${DURATION}s, verified ${OUTPUT}"
        rm -rf "${SLICE_DIR}"
        exit 0
    else
        echo "✗ Output failed verification (attempt ${ATTEMPT}/${MAX_ATTEMPTS}, likely EOS write corruption): ${OUTPUT}" >&2
        rm -f "${OUTPUT}"
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

echo "✗ Giving up after ${MAX_ATTEMPTS} attempts: ${OUTPUT}" >&2
rm -rf "${SLICE_DIR}"
exit 1
