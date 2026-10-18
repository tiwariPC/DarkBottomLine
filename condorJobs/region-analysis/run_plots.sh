#!/bin/bash
# ---------------------------------------------------------------------------
# Condor executable: region plots from the per-sample PKLs.
#   darkbottomline analyze --mode region-analysis --region-results <per_sample> ...
# Inputs are read through the /eos mount (read-only use); plots are written to
# worker scratch and xrdcp'd to <OUTDIR>/plots/ with no FUSE writes: a background loop copies
# every file once it has been quiet for ~1 min while the plotter is still running, and a final
# pass at the end copies the rest (and fails the job if any file could not be copied).
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
OUT_HOST="${DBL_OUT_EOS_HOST:-eoscms.cern.ch}"  # plot destination (override only for tests)
SYNC_EVERY="${DBL_SYNC_EVERY:-120}"            # seconds between background copy passes
PLOT_OUT="${SCRATCH}/plots"
SRC="${PLOT_OUT}/${VERSION}"   # the plotter writes <output-dir>/<version>/{png,pdf,text,root}; we copy its contents so
                                # <OUTDIR>/plots/ holds png/ pdf/ text/ root/ directly (no version level)
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

# Copy files under ${PLOT_OUT} that are not in copied.list yet. $1 = minimum age in minutes
# (files younger than that may still be written); 0 = everything. Prints the number of files
# copied in this pass. Failed files are simply retried on the next pass.
COPIED="${SCRATCH}/copied.list"; : > "${COPIED}"
sync_pass() {
    local age="$1" all="${SCRATCH}/all.list" new="${SCRATCH}/new.list"
    if [ "${age}" -gt 0 ]; then
        (cd "${SRC}" 2>/dev/null && find . -type f -mmin +"${age}" | LC_ALL=C sort) > "${all}"
    else
        (cd "${SRC}" 2>/dev/null && find . -type f | LC_ALL=C sort) > "${all}"
    fi
    LC_ALL=C sort "${COPIED}" -o "${COPIED}"
    LC_ALL=C comm -23 "${all}" "${COPIED}" > "${new}"
    export SRC OUT_HOST DEST
    xargs -a "${new}" -d '\n' -r -P 8 -I{} bash -c '
        rel="${1#./}"
        xrdcp --nopbar --force --posc --path --cksum adler32 "${SRC}/${rel}" "root://${OUT_HOST}/${DEST}/${rel}" >/dev/null 2>&1 && echo "./${rel}"
    ' _ {} >> "${COPIED}.pass" || true
    local n; n=$(wc -l < "${COPIED}.pass" 2>/dev/null || echo 0)
    cat "${COPIED}.pass" >> "${COPIED}" 2>/dev/null; rm -f "${COPIED}.pass"
    echo "${n}"
}
sync_loop() {
    while sleep "${SYNC_EVERY}"; do
        n=$(sync_pass 1); [ "${n}" -gt 0 ] && echo "[sync $(date +%H:%M:%S)] copied ${n} file(s), $(wc -l < "${COPIED}") total"
    done
}
xrdfs "${OUT_HOST}" mkdir -p "${DEST}" >/dev/null 2>&1 || true
sync_loop &
SYNC_PID=$!

python3 -m darkbottomline.cli analyze --mode region-analysis \
    --config "${DBL_CONFIG}" --regions-config "${DBL_REGIONS_CONFIG}" \
    --input "${DBL_INPUT_DIR}" --region-results "${DBL_PKL_DIR}" \
    --output-dir "${PLOT_OUT}" --version "${VERSION}" \
    --xsection-json "${DBL_XSEC_JSON}" --plot-config "${DBL_PLOT_CONFIG}" \
    --make-region-plots \
    --xsection-signal-json "${DBL_XSEC_SIGNAL_JSON}" --signal-scale "${DBL_SIGNAL_SCALE}" </dev/null
RC=$?
kill "${SYNC_PID}" 2>/dev/null; wait "${SYNC_PID}" 2>/dev/null
if [ "${RC}" -ne 0 ]; then
    echo "✗ FAILED: plotting exit ${RC}" >&2
    # keep what was already copied; the files are partial results
    exit 1
fi
N_OUT=$(find "${SRC}" -type f 2>/dev/null | wc -l)
[ "${N_OUT}" -gt 0 ] || { echo "✗ FAILED: no plots written under ${SRC}" >&2; exit 1; }
du -sh "${SRC}"; echo "${N_OUT} files"

# Final pass: everything left, up to 3 tries; every file must be on EOS.
for attempt in 1 2 3; do
    n=$(sync_pass 0); echo "[final sync ${attempt}] copied ${n} file(s)"
    left=$(cd "${SRC}" && find . -type f | LC_ALL=C sort | LC_ALL=C comm -23 - <(LC_ALL=C sort "${COPIED}") | wc -l)
    [ "${left}" -eq 0 ] && { echo "✓ ${N_OUT} plot files -> ${DEST}"; exit 0; }
    sleep $((attempt * 30))
done
echo "✗ FAILED: ${left} file(s) could not be copied to ${DEST}" >&2
exit 1
