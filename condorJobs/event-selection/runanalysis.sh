#!/bin/bash
ulimit -s unlimited # Set unlimited stack size
set -e  # Exit on error
set -x  # Debug mode - show commands

# Get repository directory from environment variable (set at submission time)
# This is the path from where condor jobs are submitted, not the condor cwd
DBL_REPO_DIR="${DBL_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Save condor cwd before changing directories (where transferred files are)
CONDOR_CWD="$(pwd)"

# Change to repository directory (like the working script)
cd "${DBL_REPO_DIR}"

echo "=========================================="
echo "DarkBottomLine Condor Job (event-selection)"
echo "=========================================="
echo "Job ID: ${1:-0}"
echo "Repository directory: ${DBL_REPO_DIR}"
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Source LCG environment (critical for CERN systems)
if [ -f "/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh" ]; then
    echo "Sourcing LCG environment..."
    source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
else
    echo "⚠ Warning: LCG setup script not found. Continuing anyway..."
fi

# Set up DarkBottomLine environment
echo "Setting up DarkBottomLine environment..."
LOCAL_DIR="${DBL_REPO_DIR}/.local"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.9")
SITE_PACKAGES_DIR="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"

if [ -d "${SITE_PACKAGES_DIR}" ]; then
    export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"
    echo "✓ Set PYTHONPATH: ${SITE_PACKAGES_DIR}"
else
    echo "⚠ Warning: .local directory not found at ${SITE_PACKAGES_DIR}"
    echo "  Dependencies may not be available."
fi

# Set default values from environment or use defaults
YEAR="${DBL_YEAR:-2022}"
CONFIG="${DBL_CONFIG:-configs/${YEAR}.yaml}"
EXECUTOR="${DBL_EXECUTOR:-futures}"
CHUNK_SIZE="${DBL_CHUNK_SIZE:-50000}"
WORKERS="${DBL_WORKERS:-4}"
MAX_EVENTS="${DBL_MAX_EVENTS:-}"
DATA_FLAG="${DBL_DATA:-false}"
# Workers: Automatically derived from request_cpus in submit file

PROC_ID="${1:-0}"
BKG_FILE="${DBL_BKG_FILE:-}"

if [ -n "${BKG_FILE}" ]; then
    # Sample lists live in data/samplelist/<year>/<name>.txt
    if [[ ! "${BKG_FILE}" == *"/"* ]]; then
        BKG_FILE="data/samplelist/${YEAR}/${BKG_FILE}"
    fi

    if [ ! -f "${BKG_FILE}" ]; then
        echo "✗ Error: Sample file not found: ${BKG_FILE}"
        echo "  Repository directory: ${DBL_REPO_DIR}"
        echo "  Current directory: $(pwd)"
        echo "  Files in data/samplelist/${YEAR}/:"
        ls -la "data/samplelist/${YEAR}/" 2>/dev/null || echo "    (directory not found)"
        exit 1
    fi
    echo "✓ Sample file found: ${BKG_FILE}"
    echo "  File size: $(ls -lh "${BKG_FILE}" | awk '{print $5}')"
    echo "  First few lines of file:"
    head -3 "${BKG_FILE}" | sed 's/^/    /'
    echo ""

    BKG_NAME=$(basename "${BKG_FILE}" .txt)

    # Get the specific file from the background file based on ProcId
    INPUT_LINE=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | sed -n "$((PROC_ID + 1))p")

    if [ -z "${INPUT_LINE}" ]; then
        TOTAL_FILES=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | wc -l)
        echo "✗ Error: ProcId ${PROC_ID} exceeds number of files in ${BKG_FILE} (${TOTAL_FILES} files)"
        exit 1
    fi

    INPUT="${INPUT_LINE}"
    FILE_INDEX="${PROC_ID}"
    echo "✓ Processing file ${FILE_INDEX} from ${BKG_FILE}"
    echo "  Input file: ${INPUT}"

    # Auto-detect collision data from sample name (e.g. JetMET-Run2022D-...)
    if [[ "${BKG_NAME}" =~ -Run20[0-9]{2} ]]; then
        DATA_FLAG="true"
        echo "  Detected collision data sample (name matches -Run20YY) → --data enabled"
    fi

    EVENT_SELECTION_OUTPUT="outputs/eventsel/${YEAR}/${BKG_NAME}_${FILE_INDEX}_EVENTSELECTION.root"
else
    echo "✗ Error: DBL_BKG_FILE not set. This script requires a sample file from data/samplelist/${YEAR}/."
    exit 1
fi

# Validate configuration file exists
if [ ! -f "${CONFIG}" ]; then
    echo "✗ Error: Configuration file not found: ${CONFIG}"
    exit 1
fi

# Validate input file exists (should be a ROOT file, local or xrootd)
if [ ! -f "${INPUT}" ] && [[ ! "${INPUT}" == root://* ]]; then
    echo "✗ Error: Input file not found: ${INPUT}"
    exit 1
fi

TOTAL_FILES=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | wc -l)

# Create output directory
OUTPUT_DIR=$(dirname "${EVENT_SELECTION_OUTPUT}")
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Job Configuration"
echo "=========================================="
echo "ProcId: ${1:-0}"
echo "Background File: ${BKG_FILE}"
echo "File Index: ${FILE_INDEX} of ${TOTAL_FILES}"
echo "Background Name: ${BKG_NAME}"
echo "Year: ${YEAR}"
echo "Config: ${CONFIG}"
echo "Input: ${INPUT}"
echo "Event-selection Output: ${EVENT_SELECTION_OUTPUT}"
echo "Executor: ${EXECUTOR}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Workers: ${WORKERS} (auto-derived from request_cpus)"
echo "Data: ${DATA_FLAG}"
if [ -n "${MAX_EVENTS}" ]; then
    echo "Max Events: ${MAX_EVENTS}"
fi
echo ""

# Build command: event-selection mode only (NanoAOD -> EVENTSELECTION.root)
CMD="python3 -m darkbottomline.cli analyze"
CMD="${CMD} --mode event-selection"
CMD="${CMD} --config ${CONFIG}"
CMD="${CMD} --input ${INPUT}"
CMD="${CMD} --event-selection-output ${EVENT_SELECTION_OUTPUT}"
CMD="${CMD} --executor ${EXECUTOR}"
CMD="${CMD} --workers ${WORKERS}"

if [ "${EXECUTOR}" = "futures" ] || [ "${EXECUTOR}" = "dask" ]; then
    CMD="${CMD} --chunk-size ${CHUNK_SIZE}"
fi

if [ -n "${MAX_EVENTS}" ]; then
    CMD="${CMD} --max-events ${MAX_EVENTS}"
fi

if [ "${DATA_FLAG}" = "true" ]; then
    CMD="${CMD} --data"
fi

echo "=========================================="
echo "Running Event Selection"
echo "=========================================="
echo "Command: ${CMD}"
echo ""

START_TIME=$(date +%s)
if eval "${CMD}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "Event Selection Completed Successfully!"
    echo "=========================================="
    echo "Output: ${EVENT_SELECTION_OUTPUT}"
    echo "Duration: ${DURATION} seconds"
    echo "Exit code: 0"
    exit 0
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "Event Selection Failed!"
    echo "=========================================="
    echo "Exit code: ${EXIT_CODE}"
    echo "Duration: ${DURATION} seconds"
    exit ${EXIT_CODE}
fi
