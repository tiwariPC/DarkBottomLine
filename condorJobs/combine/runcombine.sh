#!/bin/bash
ulimit -s unlimited
set -e
set -x

# Condor executable stub ONLY — sources the environment then execs
# `darkbottomline <subcommand> ...` with args from environment variables set
# at submission time. Contains no Combine pipeline logic of its own; that
# logic lives entirely in darkbottomline's CLI (run-all / make-datacard /
# run-combine / etc.), same as the framework's other CLI subcommands.

DBL_REPO_DIR="${DBL_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${DBL_REPO_DIR}"

echo "=========================================="
echo "DarkBottomLine Condor Job (combine)"
echo "=========================================="
echo "Job ID: ${1:-0}"
echo "Repository directory: ${DBL_REPO_DIR}"
echo "Date: $(date)"
echo ""

if [ -f "/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh" ]; then
    echo "Sourcing LCG environment..."
    source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
else
    echo "Warning: LCG setup script not found. Continuing anyway..."
fi

LOCAL_DIR="${DBL_REPO_DIR}/.local"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.9")
SITE_PACKAGES_DIR="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"
if [ -d "${SITE_PACKAGES_DIR}" ]; then
    export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"
fi

# DBL_COMBINE_SUBCOMMAND: which darkbottomline subcommand to run
#   (make-datacard | run-combine | merge-categories | merge-eras | run-all | ...)
# DBL_COMBINE_ARGS: the rest of that subcommand's argv, as a single string
SUBCOMMAND="${DBL_COMBINE_SUBCOMMAND:?DBL_COMBINE_SUBCOMMAND must be set}"
COMBINE_ARGS="${DBL_COMBINE_ARGS:-}"

echo "=========================================="
echo "Job Configuration"
echo "=========================================="
echo "Subcommand: ${SUBCOMMAND}"
echo "Args: ${COMBINE_ARGS}"
echo ""

CMD="python3 -m darkbottomline.cli ${SUBCOMMAND} ${COMBINE_ARGS}"

echo "Command: ${CMD}"
echo ""

START_TIME=$(date +%s)
if eval "${CMD}"; then
    END_TIME=$(date +%s)
    echo ""
    echo "Combine job completed successfully! Duration: $((END_TIME - START_TIME))s"
    exit 0
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    echo ""
    echo "Combine job failed! Exit code: ${EXIT_CODE}, Duration: $((END_TIME - START_TIME))s"
    exit ${EXIT_CODE}
fi
