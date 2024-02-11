#!/bin/bash

# Exit script on error
set -e

# Lint with flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Set PYTHONPATH for pytest
export PYTHONPATH=./src:$PYTHONPATH

echo -e "Running tests ..."

# Test with pytest
pytest ./test

echo -e "\rTests OK."

exit 0