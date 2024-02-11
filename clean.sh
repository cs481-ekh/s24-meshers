#!/bin/bash

# Exit script on error
set -e

echo "Cleaning package build directory..."

# Check if the directories exist and remove them
[[ -d "src/pymgm_test.egg-info" ]] && rm -r "src/pymgm_test.egg-info"
[[ -d "dist" ]] && rm -r "dist"

echo -e "\rDone."

exit 0
