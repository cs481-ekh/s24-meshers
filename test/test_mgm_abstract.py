import numpy as np
import pytest
from src.pymgm_test.mgm import mgm

class TestMGMImplementation(mgm):
    pass

@pytest.fixture
def mgmStruct():
    uh = np.array([1, 2])  # Example input vector
    return {'uh': uh}

def test_afun_square_matrix(mgmStruct):
    Lh = np.array([[1, 2], [3, 4]])  # Example square matrix representing discretization

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(mgmStruct['uh'], Lh)

    # Perform manual matrix-vector multiplication for validation
    expected_result = np.array([5, 11])

    # Assert that the result matches the expected result
    assert np.array_equal(result, expected_result)

def test_afun_non_square_matrix(mgmStruct):
    Lh = np.array([[1, 2, 3], [4, 5, 6]])  # Example non-square matrix representing discretization

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(mgmStruct['uh'], Lh)

    # Perform manual matrix-vector multiplication for validation
    expected_result = np.array([9, 12, 15])

    # Assert that the result matches the expected result
    assert np.array_equal(result, expected_result)

def test_afun_empty_matrix(mgmStruct):
    Lh = np.empty((0, 0))  # Empty matrix

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(mgmStruct['uh'], Lh)

    # Assert that the result is an empty array
    assert result.size == 0

def test_afun_empty_vector():
    Lh = np.array([[1, 2], [3, 4]])  # Example square matrix representing discretization
    uh = np.empty((0,))  # Empty input vector
    mgmStruct = {'uh': uh}

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(uh, Lh)

    # Assert that the result is an empty array
    assert result.size == 0
