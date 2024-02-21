import numpy as np
import pytest
from src.pymgm_test.mgm.mgm import  mgm

class TestMGMImplementation(mgm):
    pass

@pytest.fixture
def mgmStruct():
    uh = np.array([1, 2])  # Example input vector
    return {'uh': uh}

#--------------------------------------------------------------------------------
#                          AFUN TESTS
#--------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------
#                          MULTILEVEL TESTS
#--------------------------------------------------------------------------------
@pytest.fixture
def example_input():
    levelsData = [
        {"Mhf": np.array([[2, -1], [-1, 2]]), "Nhf": np.eye(2), "Lh": np.eye(2), "R": np.eye(2), "DLh": np.eye(2),
         "I": np.eye(2)},
        {"Mhf": np.array([[2, -1], [-1, 2]]), "Nhf": np.eye(2), "Lh": np.eye(2), "R": np.eye(2), "DLh": np.eye(2),
         "I": np.eye(2)}
    ]
    smooths = [2, 2]
    fh = np.array([1, 1])
    uh = np.zeros(2)
    expected_result = np.array([0.5, 0.5])
    return levelsData, smooths, fh, uh, expected_result


def test_multilevel_solution_accuracy(example_input):
    levelsData, smooths, fh, uh, expected_result = example_input

    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    result = mgm_obj.multilevel(fh, levelsData, smooths, uh)

    assert np.allclose(expected_result, result)


def test_multilevel_empty_input():
    # Test when the input arrays are empty
    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    with pytest.raises(ValueError):
        mgm_obj.multilevel(np.array([]), [], [], np.array([]))


def test_multilevel_non_matching_dimensions():
    # Test when input arrays have non-matching dimensions
    levelsData = [
        {"Mhf": np.array([[2, -1], [-1, 2]]), "Nhf": np.eye(2), "Lh": np.eye(2), "R": np.eye(2), "DLh": np.eye(2),
         "I": np.eye(2)},
        {"Mhf": np.array([[2, -1], [-1, 2]]), "Nhf": np.eye(2), "Lh": np.eye(2), "R": np.eye(2), "DLh": np.eye(2),
         "I": np.eye(2)}
    ]
    smooths = [2, 2]
    fh = np.array([1, 1, 1])  # Incorrect size
    uh = np.zeros(3)  # Incorrect size

    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    with pytest.raises(ValueError):
        mgm_obj.multilevel(fh, levelsData, smooths, uh)

#--------------------------------------------------------------------------------
#                          SOlVE TESTS
#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------
#                          STANDALONE TESTS
#--------------------------------------------------------------------------------