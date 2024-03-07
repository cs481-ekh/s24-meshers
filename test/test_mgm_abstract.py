import numpy as np
import pytest
import scipy as sp
from src.pymgm_test.mgm.mgm import  mgm
from sqrpoisson import squarepoissond

class TestMGMImplementation(mgm):
    pass
    def buildInterOp(self, fineLevelStruct, coarseLevelStruct):
        pass


class Struct:
    def __init__(self, Lh):
        self.Lh = Lh

@pytest.fixture
def mgmStruct():
    uh = np.array([1, 2])  # Example input vector
    return {'uh': uh}  # Wrap Lh array with key 'lh'

#--------------------------------------------------------------------------------
#                          AFUN TESTS
#--------------------------------------------------------------------------------
def test_afun_square_matrix(mgmStruct):
    Lh = np.array([[1, 2], [3, 4]])  # Example square matrix representing discretization

    # Create instances of the Struct class with different Lh arrays
    struct1 = Struct(Lh)
    struct2 = Struct(Lh)

    # Create a list of the struct instances
    mgmobj = [struct1, struct2]

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(mgmStruct['uh'], mgmobj)

    # Perform manual matrix-vector multiplication for validation
    expected_result = np.array([5, 11])

    # Assert that the result matches the expected result
    assert np.array_equal(result, expected_result)

def test_afun_non_square_matrix(mgmStruct):
    Lh = np.array([[1, 2], [3, 4], [5, 6]])  # Adjusted Lh to 3x2 matrix

    # Create instances of the Struct class with different Lh arrays
    struct1 = Struct(Lh)
    struct2 = Struct(Lh)

    # Create a list of the struct instances
    mgmobj = [struct1, struct2]

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(mgmStruct['uh'], mgmobj)

    # Perform manual matrix-vector multiplication for validation
    expected_result = np.array([5, 11, 17])  # Adjusted expected result for the 3x2 matrix

    # Assert that the result matches the expected result
    assert np.array_equal(result, expected_result)

def test_afun_empty_matrix(mgmStruct):
    Lh = np.empty((0, 0))  # Empty matrix

    struct1 = Struct(Lh)
    struct2 = Struct(Lh)

    # Create a list of the struct instances
    mgmobj = [struct1, struct2]

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Assert that calling afun with an empty matrix raises an exception
    with pytest.raises(Exception):
        mgm_obj.afun(mgmStruct['uh'], mgmobj)

def test_afun_empty_vector():
    Lh = np.array([[1, 2], [3, 4]])  # Example square matrix representing discretization
    uh = np.empty((0,))  # Empty input vector

    struct1 = Struct(Lh)
    struct2 = Struct(Lh)

    # Create a list of the struct instances
    mgmobj = [struct1, struct2]

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Assert that calling afun with an empty matrix raises an exception
    with pytest.raises(Exception):
        mgm_obj.afun(mgmStruct['uh'], mgmobj)


# #--------------------------------------------------------------------------------
# #                          MULTILEVEL TESTS
# #--------------------------------------------------------------------------------
@pytest.fixture
def example_input():
    # Load the .mat file

    mat_contents = sp.io.loadmat('multievel_data/mgmStruct_before.mat')
    keys = list(mat_contents.keys())
    print( mat_contents['mgmStruct_before'].shape)
    # Initialize an empty list to store the levelsData
    levelsData = []

    # Iterate over the data from the .mat file
    for i in range(len(mat_contents['mgmStruct_before'])):
        level_data = {
            'nodes': mat_contents['mgmStruct_before'][i][0]['nodes'],
            'stencilSize': mat_contents['mgmStruct_before'][i][0]['stencilSize'],
            'rbfOrder': mat_contents['mgmStruct_before'][i][0]['rbfOrder'],
            'rbfPolyDeg': mat_contents['mgmStruct_before'][i][0]['rbfPolyDeg'],
            'rbf': mat_contents['mgmStruct_before'][i][0]['rbf'],
            # Function handle to the polyharmonic spline kernel function needs polyHarmonic implementation
            'idx': mat_contents['mgmStruct_before'][i][0]['idx'],
            'Lh': mat_contents['mgmStruct_before'][i][0]['Lh'],
            'DLh': mat_contents['mgmStruct_before'][i][0]['DLh'],
            'I': mat_contents['mgmStruct_before'][i][0]['I'],
            'R': mat_contents['mgmStruct_before'][i][0]['R'],
            'Mhf': mat_contents['mgmStruct_before'][i][0]['Mhf'],
            'Nhf': mat_contents['mgmStruct_before'][i][0]['Nhf'],
            'Mhb': mat_contents['mgmStruct_before'][i][0]['Mhb'],
            'Nhb': mat_contents['mgmStruct_before'][i][0]['Nhb'],
            'preSmooth': mat_contents['mgmStruct_before'][i][0]['preSmooth'],
            'postSmooth': mat_contents['mgmStruct_before'][i][0]['postSmooth'],
            'Ihat': mat_contents['mgmStruct_before'][i][0]['Ihat'],
            'Rhat': mat_contents['mgmStruct_before'][i][0]['Rhat'],
            'w': mat_contents['mgmStruct_before'][i][0]['w'],
            'Qh': mat_contents['mgmStruct_before'][i][0]['Qh']
        }
        levelsData.append(level_data)
    return levelsData


def test_multilevel_solution_accuracy(example_input):

    smooths = [example_input[0]['preSmooth'][0][0], example_input[0]['postSmooth'][0][0]]
    fh = sp.io.loadmat('multievel_data/fh_before_multi.mat')
    fh = fh['fh']
    expected_result = sp.io.loadmat('multievel_data/uh_after_multi.mat')
    expected_result = expected_result['uh'].reshape(-1,1)
    uh = sp.io.loadmat('multievel_data/uh_before_multi.mat')
    uh = uh['uh']
    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    result = mgm_obj.multilevel(fh, example_input, smooths, uh).reshape(-1,1)


    assert np.allclose(expected_result, result)


def test_multilevel_empty_input():
    # Test when the input arrays are empty
    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    with pytest.raises(IndexError):
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

# # #--------------------------------------------------------------------------------
# # #                          SOlVE TESTS
# # #--------------------------------------------------------------------------------
# @pytest.fixture
# def mgmStruct1():
#     nodes = np.array([[0, 0], [1, 0], [0, 1]])  # Example nodes
#     levelsData = [{"nodes": nodes}]  # Example levelsData
#     return levelsData
# def test_solve_no_acceleration_default_parameters(mgmStruct1):
#     fh = np.array([1, 1, 1])  # Example right-hand side
#     tol = 1e-8  # Default tolerance
#     maxIters = 100  # Default maximum number of iterations
#     accel = 'none'
#
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call solve method with no acceleration and default parameters
#     uh, flag, relres, iters, resvec = mgm_obj.solve(mgmStruct1,fh, tol, accel, maxIters )
#
#     # Assert the result dimensions
#     assert uh.shape == fh.shape
#     assert isinstance(flag, int)
#     assert isinstance(relres, float)
#     assert isinstance(iters, int)
#     assert isinstance(resvec, np.ndarray)
#
#
# def test_solve_no_acceleration_custom_parameters(mgmStruct1):
#     # Define input parameters
#     fh = np.array([1, 1, 1])  # Example right-hand side
#     tol = 1e-5  # Custom tolerance
#     maxIters = 50  # Custom maximum number of iterations
#     accel = 'none'  # No acceleration
#
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call solve method with no acceleration and custom parameters
#     uh, flag, relres, iters, resvec = mgm_obj.solve(mgmStruct1, fh, tol, accel, maxIters)
#
#     # Assert the result dimensions
#     assert uh.shape == fh.shape
#     assert isinstance(flag, int)
#     assert isinstance(relres, float)
#     assert isinstance(iters, int)
#     assert isinstance(resvec, np.ndarray)
#
# def test_solve_no_acceleration_large_system(mgmStruct1):
#     # Test with a larger system
#     fh = np.ones(100)  # Example right-hand side for a larger system
#     tol = 1e-8  # Default tolerance
#     maxIters = 100  # Default maximum number of iterations
#     accel = 'none'  # No acceleration
#
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call solve method with no acceleration and default parameters for a larger system
#     uh, flag, relres, iters, resvec = mgm_obj.solve(mgmStruct1, fh, tol, accel, maxIters)
#
#     # Assert the result dimensions
#     assert uh.shape == fh.shape
#     assert isinstance(flag, int)
#     assert isinstance(relres, float)
#     assert isinstance(iters, int)
#     assert isinstance(resvec, np.ndarray)
#
#
#
# #--------------------------------------------------------------------------------
# #                          STANDALONE TESTS
# #--------------------------------------------------------------------------------
# def test_standalone_convergence(example_input):
#     # Define input parameters
#
#  # Example levelsData
#
#   # Example right-hand side
#     fh = sp.io.loadmat('standalone_data/fh_before_multi.mat')
#     fh = fh['fh']
#     tol = 1e-10  # Tolerance
#     max_iters = 100  # Maximum number of iterations
#     uh = mgmStruct['uh']  # Input vector
#     smooths = [1, 1]  # Example smooths
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call standalone method
#     uh, flag, relres, iters, resvec = mgm_obj.standalone(example_input, fh, tol, max_iters, uh, smooths)
#     expected_uh = sp.io.loadmat('standalone_data/uh_after_multi.mat')['uh']
#     expected_resvec = sp.io.loadmat('standalone_data/resvec_after_multi.mat')['resvec']
#     # Assert the result
#     assert flag == 0  # Convergence flag
#     assert relres <= tol  # Relative residual within tolerance
#     assert iters <= max_iters  # Number of iterations within maximum
#     assert np.allclose(expected_uh, uh)
#     assert np.allclose(expected_resvec, resvec)
#
#
# def test_standalone_non_convergence(mgmStruct):
#     # Define input parameters
#     levelsData = [
#         {
#             'nodes': np.array([[0, 0], [1, 0], [0, 1]]),
#             'stencilSize': 4,
#             'rbfOrder': 0,
#             'rbfPolyDeg': 0,
#             'rbf': 0,  # Function handle to the polyharmonic spline kernel function needs polyHarmonic implementation
#             'idx': [],
#             'Lh': [],
#             'DLh': [],
#             'I': [],
#             'R': [],
#             'Mhf': [],
#             'Nhf': [],
#             'Mhb': [],
#             'Nhb': [],
#             'preSmooth': 2,
#             'postSmooth': 2,
#             'Ihat': 1,
#             'Rhat': 1,
#             'w': [],
#             'Qh': 0
#         },
#         # Add more dictionaries for additional levels if needed
#     ]
#     # Example levelsData
#     fh = np.array([[1, 2], [3,4]])  # Example right-hand side
#   # Example right-hand side
#     tol = 1e-8  # Tolerance
#     max_iters = 10  # Maximum number of iterations (set intentionally low for non-convergence)
#     uh = mgmStruct['uh']  # Input vector
#     smooths = [2, 2]  # Example smooths
#
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call standalone method
#     uh, flag, relres, iters, resvec = mgm_obj.standalone(levelsData, fh, tol, max_iters, uh, smooths)
#
#     # Assert the result
#     assert flag == 1  # Non-convergence flag
#     assert relres > tol  # Relative residual exceeds tolerance
#     assert iters == max_iters  # Maximum iterations reached
#
# def test_standalone_empty_input(mgmStruct):
#     # Define empty input parameters
#     levelsData = [
#         {
#             'nodes': np.array([[0, 0], [1, 0], [0, 1]]),
#             'stencilSize': 4,
#             'rbfOrder': 0,
#             'rbfPolyDeg': 0,
#             'rbf': 0,  # Function handle to the polyharmonic spline kernel function needs polyHarmonic implementation
#             'idx': [],
#             'Lh': [],
#             'DLh': [],
#             'I': [],
#             'R': [],
#             'Mhf': [],
#             'Nhf': [],
#             'Mhb': [],
#             'Nhb': [],
#             'preSmooth': 2,
#             'postSmooth': 2,
#             'Ihat': 1,
#             'Rhat': 1,
#             'w': [],
#             'Qh': 0
#         },
#         # Add more dictionaries for additional levels if needed
#     ] # Empty levelsData
#     fh = np.array([])  # Empty right-hand side
#     tol = 1e-8  # Tolerance
#     max_iters = 100  # Maximum number of iterations
#     uh = np.array([])  # Empty input vector
#     smooths = [2, 2]  # Example smooths
#
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call standalone method
#     uh, flag, relres, iters, resvec = mgm_obj.standalone(levelsData, fh, tol, max_iters, uh, smooths)
#
#     # Assert the result
#     assert flag == 0  # Empty input should converge immediately
#     assert relres == 0  # Relative residual should be zero
#     assert iters == 0  # No iterations needed
#
# def test_standalone_zero_tolerance(mgmStruct):
#     # Define input parameters with zero tolerance
#     levelsData = [
#         {
#             'nodes': np.array([[0, 0], [1, 0], [0, 1]]),
#             'stencilSize': 4,
#             'rbfOrder': 0,
#             'rbfPolyDeg': 0,
#             'rbf': 0,  # Function handle to the polyharmonic spline kernel function needs polyHarmonic implementation
#             'idx': [],
#             'Lh': [],
#             'DLh': [],
#             'I': [],
#             'R': [],
#             'Mhf': [],
#             'Nhf': [],
#             'Mhb': [],
#             'Nhb': [],
#             'preSmooth': 2,
#             'postSmooth': 2,
#             'Ihat': 1,
#             'Rhat': 1,
#             'w': [],
#             'Qh': 0
#         },
#         # Add more dictionaries for additional levels if needed
#     ]  # Example levelsData
#     fh = np.array([1, 1, 1])  # Example right-hand side
#     tol = 0  # Zero tolerance
#     max_iters = 100  # Maximum number of iterations
#     uh = mgmStruct['uh']  # Input vector
#     smooths = [2, 2]  # Example smooths
#
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call standalone method
#     uh, flag, relres, iters, resvec = mgm_obj.standalone(levelsData, fh, tol, max_iters, uh, smooths)
#
#     # Assert the result
#     assert flag == 0  # Convergence flag
#     assert relres == 0  # Relative residual should be zero regardless of tolerance
#     assert iters == 0  # No iterations needed with zero tolerance
#
# def test_standalone_small_system(mgmStruct):
#     # Define input parameters for a very small system
#     levelsData = [
#         {
#             'nodes': np.array([[0, 0], [1, 0], [0, 1]]),
#             'stencilSize': 4,
#             'rbfOrder': 0,
#             'rbfPolyDeg': 0,
#             'rbf': 0,  # Function handle to the polyharmonic spline kernel function needs polyHarmonic implementation
#             'idx': [],
#             'Lh': [],
#             'DLh': [],
#             'I': [],
#             'R': [],
#             'Mhf': [],
#             'Nhf': [],
#             'Mhb': [],
#             'Nhb': [],
#             'preSmooth': 2,
#             'postSmooth': 2,
#             'Ihat': 1,
#             'Rhat': 1,
#             'w': [],
#             'Qh': 0
#         },
#         # Add more dictionaries for additional levels if needed
#     ]  # Single-element levelsData
#     fh = np.array([1])  # Single-element right-hand side
#     tol = 1e-8  # Tolerance
#     max_iters = 100  # Maximum number of iterations
#     uh = np.array([1])  # Single-element input vector
#     smooths = [2, 2]  # Example smooths
#
#     # Instantiate TestMGMImplementation object
#     mgm_obj = TestMGMImplementation()
#
#     # Call standalone method
#     uh, flag, relres, iters, resvec = mgm_obj.standalone(levelsData, fh, tol, max_iters, uh, smooths)
#
#     # Assert the result
#     assert flag == 0  # Convergence flag
#     assert relres == 0  # Relative residual should be zero for such a small system
#     assert iters == 0  # No iterations needed for such a small system
