#!/bin/bash

# Exit script on error
set -e

echo "Building Python package using pyproject.toml..."

# Ensure the build module is installed
python3 -m pip install --upgrade build

# Build the package
python3 -m build

echo "\rPackage built successfully."

exit 0
