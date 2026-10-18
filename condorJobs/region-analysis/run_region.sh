#!/bin/bash
# ---------------------------------------------------------------------------
# Condor executable: region analysis of ONE merged <sample>_EVENTSELECTION.root.
#   1. skip if <OUTDIR>/per_sample/<stem>_DNNAPPLIED.pkl is already on EOS
#   2. read the input over XRootD (root://eoscms.cern.ch//eos/cms/...), run
#      `darkbottomline analyze --mode region-analysis --per-sample-only` (chunked
#      read, DNN scoring), write the PKL to worker scratch
#   3. verify the PKL unpickles, xrdcp it to EOS with an adler32 checksum
#      (no FUSE writes, see condorJobs/btag-efficiency/run_slimmer.sh)
# Exits non-zero on any failure so condor holds + retries; a PKL that already
# reached EOS is skipped on the retry.
#
# Args (from submit.sub):
#   $1  PROXY_FILE  x509 proxy filename in the job sandbox
#   $2  INFILE      /eos/cms/... path of the merged EVENTSELECTION.root
#   $3  IS_DATA     true|false
# Env (from submit.sub): DBL_REPO_DIR DBL_CONFIG DBL_REGIONS_CONFIG DBL_XSEC_JSON
#   DBL_DNN_MODEL DBL_DNN_CONFIG DBL_OUTDIR
# ---------------------------------------------------------------------------
ulimit -s unlimited
set -u -o pipefail

PROXY_FILE="$1"
INFILE="$2"
IS_DATA="$3"

[ -f "${PROXY_FILE}" ] && export X509_USER_PROXY="$(pwd)/${PROXY_FILE}"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}"

REPO_DIR="${DBL_REPO_DIR}"
EOS_HOST="eoscms.cern.ch"                    # inputs
OUT_HOST="${DBL_OUT_EOS_HOST:-eoscms.cern.ch}"  # PKL destination (override only for tests, e.g. eosuser.cern.ch)
STEM=$(basename "${INFILE}" .root)
DEST_DIR="${DBL_OUTDIR}/per_sample"
DEST="${DEST_DIR}/${STEM}_DNNAPPLIED.pkl"
LOCAL_OUT="${SCRATCH}/region_out"
LOCAL_PKL="${LOCAL_OUT}/per_sample/${STEM}_DNNAPPLIED.pkl"

echo "=== region-analysis job ==="
echo "host    : $(hostname)"
echo "sample  : ${STEM} (data=${IS_DATA})"
echo "input   : ${INFILE}"
echo "config  : ${DBL_CONFIG}"
echo "dest    : ${DEST}"
echo "start   : $(date)"

if xrdfs "${OUT_HOST}" stat "${DEST}" >/dev/null 2>&1; then
    echo "→ already on EOS, skipping"
    exit 0
fi

cd "${REPO_DIR}" || { echo "✗ Error: cannot cd to ${REPO_DIR}" >&2; exit 1; }

# Environment: LCG view (torch included) + the repo's .local site-packages.
# LCG setup.sh references unset vars.
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh"
set +u
source "${LCG_SETUP}" || { echo "✗ Error: cannot source ${LCG_SETUP}" >&2; exit 1; }
set -u
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="${REPO_DIR}/.local/lib/python${PYTHON_VERSION}/site-packages:${REPO_DIR}:${PYTHONPATH:-}"

for f in "${DBL_CONFIG}" "${DBL_REGIONS_CONFIG}" "${DBL_XSEC_JSON}" "${DBL_DNN_MODEL}" "${DBL_DNN_CONFIG}"; do
    [ -f "${f}" ] || { echo "✗ Error: not found: ${f}" >&2; exit 1; }
done
xrdfs "${EOS_HOST}" stat "${INFILE}" >/dev/null 2>&1 || { echo "✗ Error: input not on EOS: ${INFILE}" >&2; exit 1; }

rm -rf "${LOCAL_OUT}"
CMD=(python3 -m darkbottomline.cli analyze --mode region-analysis --per-sample-only
     --config "${DBL_CONFIG}" --regions-config "${DBL_REGIONS_CONFIG}"
     --input "root://${EOS_HOST}/${INFILE}"
     --output "${LOCAL_OUT}"
     --xsection-json "${DBL_XSEC_JSON}"
     --apply-dnn --dnn-model "${DBL_DNN_MODEL}" --dnn-config "${DBL_DNN_CONFIG}")
[ "${IS_DATA}" = "true" ] && CMD+=(--data)

T0=$(date +%s)
"${CMD[@]}" </dev/null
RC=$?
SECONDS_USED=$(( $(date +%s) - T0 ))
if [ "${RC}" -ne 0 ]; then
    echo "✗ FAILED: darkbottomline exit ${RC} after ${SECONDS_USED}s" >&2
    exit 1
fi

# The PKL must unpickle and carry what the plotter reads.
python3 - "${LOCAL_PKL}" "${STEM}" <<'PYEOF' || { echo "✗ FAILED: PKL verification" >&2; exit 1; }
import pickle, sys
path, stem = sys.argv[1:3]
with open(path, "rb") as f:
    r = pickle.load(f)
for k in ("region_histograms", "region_cutflow_steps", "metadata"):
    assert k in r, f"missing {k}"
m = r["metadata"]
assert m.get("sample") == stem, f"metadata.sample {m.get('sample')!r} != {stem!r}"
assert m.get("weighted_total_events", 0) > 0, "weighted_total_events <= 0"
print(f"PKL ok: {m['n_events_processed']} events, {len(r['region_histograms'])} regions, "
      f"wte={m['weighted_total_events']:.6g}, xsec={m.get('xsec')}")
PYEOF

xrdfs "${OUT_HOST}" mkdir -p "${DEST_DIR}" >/dev/null 2>&1 || true
STAGED=0
for attempt in 1 2 3; do
    if xrdcp --nopbar --force --posc --cksum adler32 "${LOCAL_PKL}" "root://${OUT_HOST}/${DEST}"; then
        STAGED=1
        break
    fi
    echo "⚠ xrdcp attempt ${attempt}/3 failed for ${DEST}" >&2
    sleep $((attempt * 30))
done
rm -rf "${LOCAL_OUT}"
if [ "${STAGED}" -ne 1 ]; then
    echo "✗ FAILED: stage-out to ${DEST}" >&2
    exit 1
fi
echo "DBL_TIMING sample=${STEM} seconds=${SECONDS_USED} data=${IS_DATA}"
echo "✓ ${DEST} (${SECONDS_USED}s)"
exit 0
