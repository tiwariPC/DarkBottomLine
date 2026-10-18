#!/bin/bash
# ---------------------------------------------------------------------------
# Condor executable: region plots from the per-sample PKLs.
#   darkbottomline analyze --mode region-analysis --region-results <per_sample> ...
# Inputs are read through the /eos mount (read-only use); plots are written to
# worker scratch and xrdcp -r'd to <OUTDIR>/plots/<version>/ with no FUSE writes.
#
# Arg: $1 VERSION   Env (from plot.sub): DBL_REPO_DIR DBL_CONFIG DBL_REGIONS_CONFIG
#   DBL_XSEC_JSON DBL_XSEC_SIGNAL_JSON DBL_PLOT_CONFIG DBL_INPUT_DIR DBL_PKL_DIR
#   DBL_OUTDIR DBL_SIGNAL_SCALE
# ---------------------------------------------------------------------------
ulimit -s unlimited
set -u -o pipefail

VERSION="$1"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}"
EOS_HOST="eoscms.cern.ch"
PLOT_OUT="${SCRATCH}/plots"
DEST="${DBL_OUTDIR}/plots"

echo "=== region-plots job ==="
echo "host    : $(hostname)"
echo "version : ${VERSION}"
echo "pkls    : ${DBL_PKL_DIR} ($(ls "${DBL_PKL_DIR}" | wc -l) files)"
echo "dest    : ${DEST}"
echo "start   : $(date)"

cd "${DBL_REPO_DIR}" || { echo "✗ Error: cannot cd to ${DBL_REPO_DIR}" >&2; exit 1; }

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh"
set +u
source "${LCG_SETUP}" || { echo "✗ Error: cannot source ${LCG_SETUP}" >&2; exit 1; }
set -u
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="${DBL_REPO_DIR}/.local/lib/python${PYTHON_VERSION}/site-packages:${DBL_REPO_DIR}:${PYTHONPATH:-}"

rm -rf "${PLOT_OUT}"; mkdir -p "${PLOT_OUT}"
python3 -m darkbottomline.cli analyze --mode region-analysis \
    --config "${DBL_CONFIG}" --regions-config "${DBL_REGIONS_CONFIG}" \
    --input "${DBL_INPUT_DIR}" --region-results "${DBL_PKL_DIR}" \
    --output-dir "${PLOT_OUT}" --version "${VERSION}" \
    --xsection-json "${DBL_XSEC_JSON}" --plot-config "${DBL_PLOT_CONFIG}" \
    --make-region-plots \
    --xsection-signal-json "${DBL_XSEC_SIGNAL_JSON}" --signal-scale "${DBL_SIGNAL_SCALE}" </dev/null
RC=$?
if [ "${RC}" -ne 0 ]; then
    echo "✗ FAILED: plotting exit ${RC}" >&2
    exit 1
fi
N_OUT=$(find "${PLOT_OUT}" -type f | wc -l)
[ "${N_OUT}" -gt 0 ] || { echo "✗ FAILED: no plots written under ${PLOT_OUT}" >&2; exit 1; }
du -sh "${PLOT_OUT}"; echo "${N_OUT} files"

xrdfs "${EOS_HOST}" mkdir -p "${DEST}" >/dev/null 2>&1 || true
for attempt in 1 2 3; do
    # trailing slash on the source copies its contents, not the directory itself
    if xrdcp -r --nopbar --force --posc --cksum adler32 "${PLOT_OUT}/" "root://${EOS_HOST}/${DEST}/"; then
        echo "✓ plots -> ${DEST}"
        exit 0
    fi
    echo "⚠ xrdcp -r attempt ${attempt}/3 failed" >&2
    sleep $((attempt * 30))
done
echo "✗ FAILED: stage-out to ${DEST}" >&2
exit 1
