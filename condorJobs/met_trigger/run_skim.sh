#!/bin/bash
# ---------------------------------------------------------------------------
# Condor executable: skim ONE samplelist .txt into per-file skim ROOTs.
#
# One job = one text file. Every ROOT path listed inside that .txt is
# preselected and written as <outdir>/<txtstem>_<rootid>.root by
# scripts/met_trigger_efficiency.py skim.
#
# Args (from submit.sub):
#   $1  REPO_DIR   absolute path to the DarkBottomLine checkout (shared FS)
#   $2  CONFIG     year YAML, e.g. configs/2024.yaml
#   $3  OUTDIR     output directory for skim ROOTs (user-set, condor-visible)
#   $4  KIND       "data" or "mc" (selects --data-files vs --mc-files)
#   $5  TXTFILE    the samplelist .txt to process (absolute or repo-relative)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$1"
CONFIG="$2"
OUTDIR="$3"
KIND="$4"
TXTFILE="$5"

echo "=== met_trigger skim job ==="
echo "host      : $(hostname)"
echo "repo      : ${REPO_DIR}"
echo "config    : ${CONFIG}"
echo "outdir    : ${OUTDIR}"
echo "kind      : ${KIND}"
echo "txtfile   : ${TXTFILE}"
echo "start     : $(date)"

cd "${REPO_DIR}"

# Grid proxy for XRootD reads (submit with: -x509userproxy in submit.sub).
if [[ -n "${X509_USER_PROXY:-}" ]]; then
    voms-proxy-info -exists || echo "WARNING: no valid proxy; XRootD reads may fail"
fi

# Repo environment (conda + pip install -e per CLAUDE.md lxplus flow).
source start.sh

mkdir -p "${OUTDIR}"

if [[ "${KIND}" == "data" ]]; then
    FILE_FLAG="--data-files"
elif [[ "${KIND}" == "mc" ]]; then
    FILE_FLAG="--mc-files"
else
    echo "ERROR: KIND must be 'data' or 'mc', got '${KIND}'" >&2
    exit 2
fi

python scripts/met_trigger_efficiency.py skim \
    --config "${CONFIG}" \
    ${FILE_FLAG} "${TXTFILE}" \
    --outdir "${OUTDIR}"

echo "done      : $(date)"
