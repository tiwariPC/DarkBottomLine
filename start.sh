#!/bin/bash
# Script to set up DarkBottomLine environment for use
# Run this script before using DarkBottomLine in a new session
# Usage: source start.sh  (or: . start.sh)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/.local"

echo "=========================================="
echo "DarkBottomLine Environment Setup"
echo "=========================================="
echo ""

# Get Python version for site-packages path
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
if [ -z "$PYTHON_VERSION" ]; then
    echo "Error: python3 not found. Please ensure Python 3 is installed."
    return 1 2>/dev/null || exit 1
fi

SITE_PACKAGES_DIR="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"
LOCAL_BIN_DIR="${LOCAL_DIR}/bin"

# Check if .local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "⚠ Warning: .local directory not found at ${LOCAL_DIR}"
    echo "  Make sure you have run the installation first:"
    echo "    1. python3 check_requirements.py --install --local-dir ./.local"
    echo "    2. ./lxplus_setup.sh"
    echo ""
    return 1 2>/dev/null || exit 1
fi

# Setup PYTHONPATH
if [ -d "$SITE_PACKAGES_DIR" ]; then
    if [[ ":$PYTHONPATH:" != *":$SITE_PACKAGES_DIR:"* ]]; then
        export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"
        echo "✓ Added to PYTHONPATH: ${SITE_PACKAGES_DIR}"
    else
        echo "✓ PYTHONPATH already contains: ${SITE_PACKAGES_DIR}"
    fi
else
    echo "⚠ Warning: site-packages directory not found: ${SITE_PACKAGES_DIR}"
    echo "  Dependencies may not be installed. Run: python3 check_requirements.py --install --local-dir ./.local"
fi

# Setup PATH for darkbottomline command
if [ -d "$LOCAL_BIN_DIR" ]; then
    if [[ ":$PATH:" != *":$LOCAL_BIN_DIR:"* ]]; then
        export PATH="${LOCAL_BIN_DIR}:${PATH}"
        echo "✓ Added to PATH: ${LOCAL_BIN_DIR}"
    else
        echo "✓ PATH already contains: ${LOCAL_BIN_DIR}"
    fi
else
    echo "⚠ Warning: bin directory not found: ${LOCAL_BIN_DIR}"
    echo "  The darkbottomline command may not be available."
fi

echo ""
echo "=========================================="
echo "Environment Setup Complete!"
echo "=========================================="
echo ""
echo "You can now use DarkBottomLine:"
echo "  darkbottomline --help"
echo "  # Or:"
echo "  python3 -m darkbottomline.cli --help"
echo ""
echo "To verify installation:"
echo "  python3 -c \"from darkbottomline import DarkBottomLineProcessor; print('✓ Import successful')\""
echo ""
