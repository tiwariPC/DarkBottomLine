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
pip install -e "$(dirname "$0")" || { echo "pip install -e failed!"; return 1; }
echo "Environment ready! Python: $(python --version)"
