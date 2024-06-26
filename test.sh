#!/bin/bash
g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/pymgm_test/utils/bindings.cpp src/pymgm_test/utils/PcCoarsen2D.cpp -o src/pymgm_test/utils/PcCoarsen`python3-config --extension-suffix`
ls
# Exit script on error
set -e
# Compile 'elim' library



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
