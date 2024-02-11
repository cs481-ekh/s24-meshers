#!/bin/bash

# Default python command is python3
pyname="python3"

# In case one wants to run python, py, ... instead of python3
if [ ! -z "$1" ]; then
    pyname="$1"
fi

# Exit script on error
set -e

echo "Building Python package using pyproject.toml..."

# Ensure the build module is installed
"$pyname" -m pip install --upgrade build

# Build the package
"$pyname" -m build

echo -e "\rPackage built successfully."

exit 0
