#!/bin/sh
# ---------------------------------------------------------------------------
# Condor executable: skim a BATCH of ROOT files (a contiguous slice of a
# samplelist .txt) into per-file skim ROOTs: <OUTDIR>/<txtstem>/<txtstem>_<rootid>.root
#
# Job model: one cluster per .txt, one job per BATCH-sized slice. ProcId selects
# the slice = lines [ProcId*BATCH+1 .. ProcId*BATCH+BATCH] of TXTFILE. The skim
# loops those files and writes one output ROOT per input file (rootid keeps each
# unique), so batching only changes how many files a single job processes.
#
# Args (from submit.sub):
#   $1  PROXY      x509 proxy filename in the job sandbox (shipped via
#                  transfer_input_files); sets X509_USER_PROXY for XRootD reads
#   $2  REPO_DIR   absolute path to the DarkBottomLine checkout (shared FS)
#   $3  CONFIG     year YAML, e.g. configs/2024.yaml
#   $4  OUTDIR     output directory for skim ROOTs (user-set, AFS/EOS-visible)
#   $5  KIND       "data" or "mc" (selects --data-files vs --mc-files)
#   $6  TXTFILE    the samplelist .txt (on shared FS; job reads a slice of it)
#   $7  PROCID     0-based job index → selects the slice of TXTFILE
#   $8  BATCH      number of ROOT files per job (slice size)
#
# Output: <OUTDIR>/<txtstem>/<txtstem>_<rootid>.root  (one per input ROOT)
# ---------------------------------------------------------------------------
ulimit -s unlimited
set -e

PROXY="$1"
REPO_DIR="$2"
CONFIG="$3"
OUTDIR="$4"
KIND="$5"
TXTFILE="$6"
PROCID="$7"
BATCH="$8"

# Grid proxy shipped into the sandbox: point XRootD at it (relative to CWD, the
# sandbox, before we cd into the repo).
export X509_USER_PROXY="$(pwd)/${PROXY}"

echo "=== met_trigger skim job ==="
echo "host    : $(hostname)"
echo "proxy   : ${X509_USER_PROXY}"
echo "repo    : ${REPO_DIR}"
echo "config  : ${CONFIG}"
echo "outdir  : ${OUTDIR}"
echo "kind    : ${KIND}"
echo "txtfile : ${TXTFILE}"
echo "procid  : ${PROCID}"
echo "batch   : ${BATCH}"
echo "start   : $(date)"

cd "${REPO_DIR}"

# Environment — mirror condorJobs/runanalysis.sh (the known-working setup):
# source the LCG view, then set PYTHONPATH directly to the repo's .local
# site-packages. Do NOT use start.sh (it re-derives the python version under a
# different shell and can point PYTHONPATH at the wrong path). The LCG setup.sh
# references an unset $COMPILER, so disable nounset around it.
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh"
set +u
if [ -f "${LCG_SETUP}" ]; then
    echo "Sourcing LCG environment..."
    source "${LCG_SETUP}"
else
    echo "⚠ Warning: LCG setup not found: ${LCG_SETUP}. Continuing anyway..."
fi
# (leave nounset off; the script does not depend on it, and ${PYTHONPATH} may be unset)

# Repo deps: .local site-packages (pip-installed) + the repo root itself (so the
# 'darkbottomline' package imports). We run python from ${REPO_DIR} below too.
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

if [ "${KIND}" = "data" ]; then
    FILE_FLAG="--data-files"
elif [ "${KIND}" = "mc" ]; then
    FILE_FLAG="--mc-files"
else
    echo "ERROR: KIND must be 'data' or 'mc', got '${KIND}'" >&2
    exit 2
fi

# This job's slice = ROOT lines [START..END] of TXTFILE (1-based), after dropping
# comment/blank lines (same filter as the skim's own .txt parser). sed clamps END
# past EOF, so the last job's slice is naturally short.
START=$((PROCID * BATCH + 1))
END=$((START + BATCH - 1))
SLICE=$(grep -v '^#' "${TXTFILE}" | grep -v '^[[:space:]]*$' | sed -n "${START},${END}p")
if [ -z "${SLICE}" ]; then
    echo "ERROR: empty slice (lines ${START}-${END}) of ${TXTFILE}" >&2
    exit 4
fi
N_IN_SLICE=$(printf '%s\n' "${SLICE}" | grep -c .)
echo "slice   : lines ${START}-${END} (${N_IN_SLICE} files)"

TXT_STEM=$(basename "${TXTFILE}" .txt)

# Write the slice samplelist to a PRIVATE scratch dir — never the repo root.
# pwd is REPO_DIR (we cd'd there), so writing here would pollute the shared checkout
# and clash with the real data/samplelist files. The temp file must be named
# <TXT_STEM>.txt so the skim derives the <txtstem>_<rootid>.root output names from
# it — isolate it in its own scratch subdirectory (per ProcId) instead of renaming.
SCRATCH="${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}"
SLICE_DIR="${SCRATCH}/slice_${TXT_STEM}_${PROCID}"
mkdir -p "${SLICE_DIR}"
SLICE_TXT="${SLICE_DIR}/${TXT_STEM}.txt"
printf '%s\n' "${SLICE}" > "${SLICE_TXT}"

# Skim into a per-samplelist subdir: <OUTDIR>/<txtstem>/. The skim loops every ROOT
# in the slice txt and writes one <txtstem>_<rootid>.root per file (rootid unique).
DEST_DIR="${OUTDIR}/${TXT_STEM}"
mkdir -p "${DEST_DIR}"
python3 scripts/met_trigger_efficiency.py skim \
    --config "${CONFIG}" \
    ${FILE_FLAG} "${SLICE_TXT}" \
    --outdir "${DEST_DIR}"

rm -rf "${SLICE_DIR}"
echo "done    : $(date)"
