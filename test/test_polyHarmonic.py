import pytest
import numpy as np
from src.pymgm_test.utils.polyHarmonic import polyHarmonic

@pytest.mark.parametrize(
    "r2, ell, k, ex",
    [
        (25, 0, 1, 5.0),
        (16, 1, 1, 64.0),
        (9, 3, 2, 1701.0),
        (1, 4, 2, 9.0),
        (16, 4, 3, 64512.0),
        (9, 2, 3, 45.0)
    ]
)

# The test cases below ensure that the matlab files arei beng loaded in properly 
# (containing specific data about the matrix and its determinant) 
# r2, ell, k as defined for polyHarmonic
# ex is the expected value
def test_polyHarmonic(r2, ell, k, ex):
    phi = polyHarmonic(r2, ell, k)
    print(phi)
    assert np.isclose(phi, ex)
