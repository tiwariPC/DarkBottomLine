#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Only create if it doesn't already exist
if ! conda env list | grep -q "darkbottomline"; then
  echo "Creating environment..."
  mamba env create -f environment.yml
else
  echo "Environment already exists, skipping creation."
fi

conda activate darkbottomline
# Install the local package in editable mode
echo "Installing local package..."
pip install -e "$(dirname "$0")" || { echo "pip install -e failed!"; exit 1; }
echo "Environment ready! Python: $(python --version)"

# OPTIONAL: build CMS Combine (HiggsAnalysis-CombinedLimit) into this conda
# env, via its standalone CMake build. Gated behind an explicit env var so
# plain `source local_setup.sh` behaves exactly as it did before — only
# needed if you intend to run `darkbottomline run-combine`/`merge-categories`/
# `merge-eras` locally; datacard-generation does not need it.
# Usage: INSTALL_COMBINE=1 source local_setup.sh
if [ "${INSTALL_COMBINE:-0}" = "1" ]; then
  REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
  COMBINE_SRC="${REPO_DIR}/.combine_build/HiggsAnalysis/CombinedLimit"

  if command -v combine &>/dev/null; then
    echo "Combine already found on PATH — skipping build."
  else
    echo "Building CMS Combine into this environment (INSTALL_COMBINE=1)..."

    # Build toolchain (cmake, ninja) is not in environment.yml — it's only
    # needed for this optional Combine build, so install it into the active
    # env here rather than making every darkbottomline user carry it.
    if ! command -v cmake &>/dev/null || ! command -v ninja &>/dev/null; then
      echo "Installing build toolchain (cmake, ninja) into ${CONDA_PREFIX}..."
      conda install -y -n darkbottomline -c conda-forge cmake ninja \
        || { echo "Failed to install cmake/ninja!"; exit 1; }
    fi

    if [ ! -d "${COMBINE_SRC}" ]; then
      mkdir -p "$(dirname "${COMBINE_SRC}")"
      git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git "${COMBINE_SRC}" \
        || { echo "Combine git clone failed!"; exit 1; }
    fi
    PY_SITE_PACKAGES="lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
    NCPU="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
    BUILD_JOBS=$(( NCPU > 2 ? NCPU - 2 : 1 ))
    (
      cd "${COMBINE_SRC}" \
        && cmake -S . -B build -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
                 -DCMAKE_INSTALL_PYTHONDIR="${PY_SITE_PACKAGES}" -DUSE_VDT=OFF \
        && cmake --build build -j"${BUILD_JOBS}" \
        && cmake --install build
    ) || { echo "Combine build failed!"; exit 1; }
    echo "Combine installed into ${CONDA_PREFIX}"
  fi

  # diffNuisances.py (pull extraction from fitDiagnostics.root, used by the
  # Run2-parity pulls stage) lives in HiggsAnalysis-CombinedLimit's test/ dir
  # but is not installed by the CMake build above (only test/-tree scripts
  # relevant to plotting — plotImpacts.py, plotGof.py — get installed).
  # Fetch it from the same repo/branch as the build (or, if Combine was
  # already on PATH and never cloned locally, a fresh shallow clone) and drop
  # it directly into the conda env's bin/ alongside the other Combine CLI
  # tools, so `diffNuisances.py` resolves on PATH exactly like `combine`/
  # `plotGof.py` do.
  if ! command -v diffNuisances.py &>/dev/null; then
    echo "Fetching diffNuisances.py (not installed by the Combine CMake build)..."
    if [ -f "${COMBINE_SRC}/test/diffNuisances.py" ]; then
      cp "${COMBINE_SRC}/test/diffNuisances.py" "${CONDA_PREFIX}/bin/diffNuisances.py"
    else
      TMP_CLONE="$(mktemp -d)"
      git clone --depth 1 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git "${TMP_CLONE}" \
        && cp "${TMP_CLONE}/test/diffNuisances.py" "${CONDA_PREFIX}/bin/diffNuisances.py" \
        || echo "Warning: could not fetch diffNuisances.py — pulls stage will fail until it's installed manually."
      rm -rf "${TMP_CLONE}"
    fi
    if [ -f "${CONDA_PREFIX}/bin/diffNuisances.py" ]; then
      chmod +x "${CONDA_PREFIX}/bin/diffNuisances.py"
      echo "diffNuisances.py installed to ${CONDA_PREFIX}/bin/"
    fi
  fi
fi
