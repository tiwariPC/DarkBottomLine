#!/bin/bash
ulimit -s unlimited # Set unlimited stack size
set -e  # Exit on error
set -x  # Debug mode - show commands

# Get repository directory from environment variable (set at submission time)
# This is the path from where condor jobs are submitted, not the condor cwd
DBL_REPO_DIR="${DBL_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Save condor cwd before changing directories (where transferred files are)
CONDOR_CWD="$(pwd)"

# Change to repository directory (like the working script)
cd "${DBL_REPO_DIR}"

echo "=========================================="
echo "DarkBottomLine Condor Job"
echo "=========================================="
echo "Job ID: ${1:-0}"
echo "Repository directory: ${DBL_REPO_DIR}"
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Show directory structure after file transfer (for debugging)
echo "=========================================="
echo "Directory Structure (after file transfer)"
echo "=========================================="
echo "Current directory: $(pwd)"
echo ""
echo "Top-level files and directories:"
ls -la | head -20
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
# Use .local directory from repository location (like the working script)
LOCAL_DIR="${DBL_REPO_DIR}/.local"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.9")
SITE_PACKAGES_DIR="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"

# Set PYTHONPATH directly (like the working script)
if [ -d "${SITE_PACKAGES_DIR}" ]; then
    export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"
    echo "✓ Set PYTHONPATH: ${SITE_PACKAGES_DIR}"
else
    echo "⚠ Warning: .local directory not found at ${SITE_PACKAGES_DIR}"
    echo "  Dependencies may not be available."
fi



# Set default values from environment or use defaults
CONFIG="${DBL_CONFIG:-configs/2022.yaml}"
REGIONS_CONFIG="${DBL_REGIONS_CONFIG:-configs/regions.yaml}"
EXECUTOR="${DBL_EXECUTOR:-futures}"
CHUNK_SIZE="${DBL_CHUNK_SIZE:-50000}"
WORKERS="${DBL_WORKERS:-4}"
MAX_EVENTS="${DBL_MAX_EVENTS:-}"
# Workers: Automatically derived from request_cpus in submit file


PROC_ID="${1:-0}"
BKG_FILE="${DBL_BKG_FILE:-}"

if [ -n "${BKG_FILE}" ]; then
    # Background file is in repository: condorJobs/samplefiles/filename.txt
    # Since we're in DBL_REPO_DIR, the path is condorJobs/samplefiles/filename.txt
    if [[ ! "${BKG_FILE}" == *"/"* ]]; then
        BKG_FILE="condorJobs/samplefiles/${BKG_FILE}"
    fi

    # Check if file exists in repository directory
    if [ ! -f "${BKG_FILE}" ]; then
        echo "✗ Error: Sample file not found: ${BKG_FILE}"
        echo "  Looking for file in repository directory..."
        echo "  Repository directory: ${DBL_REPO_DIR}"
        echo "  Current directory: $(pwd)"
        echo "  Files in condorJobs/samplefiles/:"
        ls -la condorJobs/samplefiles/ 2>/dev/null || echo "    (directory not found)"
        exit 1
    fi
    # Show that the file was found
    echo "✓ Sample file found: ${BKG_FILE}"
    echo "  File size: $(ls -lh "${BKG_FILE}" | awk '{print $5}')"
    echo "  First few lines of file:"
    head -3 "${BKG_FILE}" | sed 's/^/    /'
    echo ""

    # Extract background name
    BKG_NAME=$(basename "${BKG_FILE}" .txt)

    # Get the specific file from the background file based on ProcId
    # Skip comments and empty lines
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

    # Generate output name: regions_<bkg_name>_<file_index>.pkl
    if [ -z "${DBL_OUTPUT}" ]; then
        OUTPUT="outputs/hists/regions_${BKG_NAME}_${FILE_INDEX}.pkl"
        EVENT_SELECTION_OUTPUT="outputs/hists/event_selected_${BKG_NAME}_${FILE_INDEX}.pkl"
    else
        OUTPUT="${DBL_OUTPUT}"
        # Generate event selection output based on main output path
        OUTPUT_BASE=$(dirname "${OUTPUT}")
        OUTPUT_NAME=$(basename "${OUTPUT}" .pkl)
        EVENT_SELECTION_OUTPUT="${OUTPUT_BASE}/event_selected_${OUTPUT_NAME#regions_}.pkl"
    fi

else
    # Default fallback - use single default file (for testing)
    INPUT="root://cms-xrd-global.cern.ch//store/mc/Run3Summer22NanoAODv12/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2560000/30624dd1-ba96-465e-a745-8ff472357277.root"
    BKG_NAME="default_${PROC_ID}"
    OUTPUT="outputs/hists/regions_${BKG_NAME}.pkl"
    EVENT_SELECTION_OUTPUT="outputs/hists/event_selected_${BKG_NAME}.pkl"
    echo "⚠ No DBL_BKG_FILE set, using default input file"
fi

# Validate configuration files exist
if [ ! -f "${CONFIG}" ]; then
    echo "✗ Error: Configuration file not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${REGIONS_CONFIG}" ]; then
    echo "✗ Error: Regions configuration file not found: ${REGIONS_CONFIG}"
    exit 1
fi

# Validate input file exists (should be a ROOT file)
if [ ! -f "${INPUT}" ] && [[ ! "${INPUT}" == root://* ]]; then
    echo "✗ Error: Input file not found: ${INPUT}"
    echo "  Looking for: ${INPUT}"
    exit 1
fi

# Show input information
if [ -n "${BKG_FILE}" ]; then
    TOTAL_FILES=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | wc -l)
    echo "✓ Processing file ${FILE_INDEX} of ${TOTAL_FILES} from ${BKG_FILE}"
fi

# Create output directory
OUTPUT_DIR=$(dirname "${OUTPUT}")
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Job Configuration"
echo "=========================================="
echo "ProcId: ${1:-0}"
if [ -n "${BKG_FILE}" ]; then
    echo "Background File: ${BKG_FILE}"
    echo "File Index: ${FILE_INDEX}"
fi
echo "Background Name: ${BKG_NAME}"
echo "Config: ${CONFIG}"
echo "Regions Config: ${REGIONS_CONFIG}"
echo "Input: ${INPUT}"
if [ -n "${BKG_FILE}" ]; then
    TOTAL_FILES=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | wc -l)
    echo "  → File ${FILE_INDEX} of ${TOTAL_FILES} from ${BKG_FILE}"
fi
echo "Output: ${OUTPUT}"
echo "Executor: ${EXECUTOR}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Workers: ${WORKERS} (auto-derived from request_cpus)"
if [ -n "${MAX_EVENTS}" ]; then
    echo "Max Events: ${MAX_EVENTS}"
fi
echo ""

# Build command
CMD="python3 -m darkbottomline.cli analyze"
CMD="${CMD} --config ${CONFIG}"
CMD="${CMD} --regions-config ${REGIONS_CONFIG}"
CMD="${CMD} --input ${INPUT}"
CMD="${CMD} --output ${OUTPUT}"
CMD="${CMD} --event-selection-output ${EVENT_SELECTION_OUTPUT}"
CMD="${CMD} --executor ${EXECUTOR}"
CMD="${CMD} --workers ${WORKERS}"

# Add chunk-size if executor supports it
# CHUNK_SIZE can be a number or "auto" for automatic optimization
if [ "${EXECUTOR}" = "futures" ] || [ "${EXECUTOR}" = "dask" ]; then
    CMD="${CMD} --chunk-size ${CHUNK_SIZE}"
fi

# Add max-events if specified
if [ -n "${MAX_EVENTS}" ]; then
    CMD="${CMD} --max-events ${MAX_EVENTS}"
fi

echo "=========================================="
echo "Running Analysis"
echo "=========================================="
echo "Command: ${CMD}"
echo ""

# Run the analysis
START_TIME=$(date +%s)
if eval "${CMD}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "Analysis Completed Successfully!"
    echo "=========================================="
    echo "Output: ${OUTPUT}"
    echo "Duration: ${DURATION} seconds"
    echo "Exit code: 0"
    exit 0
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "Analysis Failed!"
    echo "=========================================="
    echo "Exit code: ${EXIT_CODE}"
    echo "Duration: ${DURATION} seconds"
    exit ${EXIT_CODE}
fi
