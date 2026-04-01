#!/bin/bash
# Script to install DarkBottomLine on lxplus (CERN computing environment)
# Creator: ptiwari (main creator)
#
# This script installs the DarkBottomLine repository itself.
# Dependencies should be installed first using: python3 check_requirements.py --install --local-dir ./.local

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/.local"

# Get Python version for site-packages path
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES_DIR="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"

echo "=========================================="
echo "DarkBottomLine Installation for lxplus"
echo "=========================================="
echo ""
echo "Note: Make sure dependencies are installed first:"
echo "  python3 check_requirements.py --install --local-dir ./.local"
echo ""

# Add .local site-packages to PYTHONPATH if it exists (packages installed by check_requirements.py)
# When using --user with PYTHONUSERBASE, packages go to lib/pythonX.X/site-packages
if [ -d "$SITE_PACKAGES_DIR" ]; then
    if [[ ":$PYTHONPATH:" != *":$SITE_PACKAGES_DIR:"* ]]; then
        export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"
        echo "✓ Found .local site-packages directory, added to PYTHONPATH"
    fi
elif [ -d "$LOCAL_DIR" ]; then
    # Fallback to base directory if site-packages doesn't exist yet
    if [[ ":$PYTHONPATH:" != *":${LOCAL_DIR}:"* ]]; then
        export PYTHONPATH="${LOCAL_DIR}:${PYTHONPATH}"
        echo "✓ Found .local directory, added to PYTHONPATH"
    fi
fi
# Install the package in development mode
echo "Installing DarkBottomLine repository..."
echo "----------------------------------------"
echo "Running: pip3 install -e . --prefix ${LOCAL_DIR}"
echo ""

# Temporarily remove .local from PYTHONPATH to avoid conflicts during installation
# We'll add it back after
OLD_PYTHONPATH="$PYTHONPATH"
if [[ "$PYTHONPATH" == *"${SITE_PACKAGES_DIR}"* ]]; then
    # Remove site-packages from PYTHONPATH temporarily
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^${SITE_PACKAGES_DIR}$" | tr '\n' ':' | sed 's/:$//')
elif [[ "$PYTHONPATH" == *"${LOCAL_DIR}"* ]]; then
    # Remove .local from PYTHONPATH temporarily
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^${LOCAL_DIR}$" | tr '\n' ':' | sed 's/:$//')
fi

# Ensure .local directory exists
mkdir -p "${LOCAL_DIR}"

# Install using --prefix to force installation to cwd .local
# Note: --prefix installs to prefix/lib/pythonX.X/site-packages, so we need to adjust
# For editable installs, we need to set PYTHONUSERBASE instead
export PYTHONUSERBASE="${LOCAL_DIR}"
unset PYTHONHOME  # Prevent interference

# if pip3 install -e . --user --no-cache-dir; then
if pip3 install -e . --prefix ${LOCAL_DIR} --no-cache-dir --no-deps; then
    echo "✓ DarkBottomLine repository installed successfully!"
    echo ""

    # Restore PYTHONPATH
    export PYTHONPATH="$OLD_PYTHONPATH"

    # Make sure site-packages directory is in PYTHONPATH
    if [ -d "$SITE_PACKAGES_DIR" ]; then
        if [[ ":$PYTHONPATH:" != *":$SITE_PACKAGES_DIR:"* ]]; then
            export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"
        fi
    fi

    unset PYTHONUSERBASE

    # Add .local/bin to PATH if not already there (for darkbottomline command)
    LOCAL_BIN_DIR="${LOCAL_DIR}/bin"
    if [ -d "$LOCAL_BIN_DIR" ] && [[ ":$PATH:" != *":$LOCAL_BIN_DIR:"* ]]; then
        export PATH="$LOCAL_BIN_DIR:$PATH"
        echo "Added ${LOCAL_BIN_DIR} to PATH for this session"
        echo "To make it permanent, add to your ~/.bashrc:"
        echo "  export PATH=\"${LOCAL_BIN_DIR}:\$PATH\""
        echo ""
    fi

    # Verify installation
    echo "Verifying installation..."
    if python3 -c "from darkbottomline import DarkBottomLineProcessor; print('✓ Import successful')" 2>/dev/null; then
        echo "✓ Package can be imported"
    else
        echo "⚠ Package installed but import test failed"
        echo "  Make sure .local is in your PYTHONPATH:"
        echo "  export PYTHONPATH=\"${LOCAL_DIR}:\$PYTHONPATH\""
        echo "  export PYTHONPATH=\"${SITE_PACKAGES_DIR}:\$PYTHONPATH\""
    fi

    # Check if command is available
    if command -v darkbottomline &> /dev/null; then
        echo "✓ darkbottomline command is available"
    else
        echo "⚠ darkbottomline command not found in PATH"
        echo "  Make sure ${LOCAL_BIN_DIR} is in your PATH:"
        echo "  export PATH=\"${LOCAL_BIN_DIR}:\$PATH\""
        echo "  Or run: python3 -m darkbottomline.cli --help"
    fi

    echo ""
    echo "=========================================="
    echo "Installation Complete!"
    echo "=========================================="
    echo ""

    # Export paths for current session
    echo "Setting up environment for current session..."
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    SITE_PACKAGES_PATH="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"

    # Export PYTHONPATH
    if [ -d "$SITE_PACKAGES_PATH" ]; then
        if [[ ":$PYTHONPATH:" != *":$SITE_PACKAGES_PATH:"* ]]; then
            export PYTHONPATH="${SITE_PACKAGES_PATH}:${PYTHONPATH}"
            echo "✓ Exported PYTHONPATH=${SITE_PACKAGES_PATH}:${PYTHONPATH}"
        fi
    fi

    # Export PATH for darkbottomline command
    if [ -d "$LOCAL_BIN_DIR" ] && [[ ":$PATH:" != *":$LOCAL_BIN_DIR:"* ]]; then
        export PATH="${LOCAL_BIN_DIR}:${PATH}"
        echo "✓ Exported PATH=${LOCAL_BIN_DIR}:${PATH}"
    fi

    echo ""
    echo "To use DarkBottomLine in this session:"
    echo "  darkbottomline --help"
    echo "  # Or:"
    echo "  python3 -m darkbottomline.cli --help"
    echo ""
    echo "To use DarkBottomLine in future sessions, export these paths:"
    echo "  PYTHON_VERSION=\$(python3 -c \"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')\")"
    echo "  export PYTHONPATH=\"${LOCAL_DIR}/lib/python/${PYTHON_VERSION}/site-packages:\$PYTHONPATH\""
    echo "  export PATH=\"${LOCAL_BIN_DIR}:\$PATH\""
    echo ""
    echo "Or add to your ~/.bashrc or ~/.bash_profile:"
    echo "  # DarkBottomLine paths"
    echo "  PYTHON_VERSION=\$(python3 -c \"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')\")"
    echo "  export PYTHONPATH=\"${LOCAL_DIR}/lib/python/${PYTHON_VERSION}/site-packages:\$PYTHONPATH\""
    echo "  export PATH=\"${LOCAL_BIN_DIR}:\$PATH\""
    echo ""
else
    # Restore PYTHONPATH and unset PYTHONUSERBASE
    export PYTHONPATH="$OLD_PYTHONPATH"
    unset PYTHONUSERBASE

    echo "✗ Installation failed"
    echo ""
    echo "If you see pkg_resources errors, try:"
    echo "  pip3 install --target ${LOCAL_DIR} setuptools jaraco.text"
    echo "  export PYTHONUSERBASE=\"${LOCAL_DIR}\""
    echo "  pip3 install -e . --user"
    exit 1
fi
