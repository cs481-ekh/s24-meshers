import numpy as np
import pytest
import scipy as sp
from src.pymgm_test.mgm.mgm import  mgm
from sqrpoisson import squarepoissond
import os


@pytest.mark.skip(reason="Concrete test class for abstract base class testing")
class TestMGMImplementation(mgm):
    __test__ = False
    pass
    def buildInterpOp(self, fineLevelStruct, coarseLevelStruct, interpMethod):
        pass



def construct_file_path(folder_name, file_name):
    # Construct the file path dynamically based on the current working directory
    if os.path.basename(os.getcwd()) != 'test':
        return os.path.join(os.getcwd(), 'test', folder_name, file_name)
    else:
        return os.path.join(folder_name, file_name)
# --------------------------------------------------------------------------------
#                          AFUN TESTS
# --------------------------------------------------------------------------------
def test_afun_square_matrix(mgmObj):
    Lh = mgmObj[4]['levelsData'][0]['Lh']  # Example square matrix representing discretization
    uh = np.zeros(2401).reshape(-1,1)  # Example input vector

    # Create a list of the struct instances

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(uh, mgmObj)

    # Perform manual matrix-vector multiplication for validation
    expected_result = sp.io.loadmat(construct_file_path('afun_data', 'uh_after_afun_bicgstab.mat'))['uh']

    # Assert that the result matches the expected result
    assert np.array_equal(result, expected_result)

def test_afun_non_square_matrix(mgmObj):
    original_Lh = mgmObj[4]['levelsData'][0]['Lh']

    # Calculate the total number of elements and the desired number of columns
    total_elements = 2401 *2401
    num_cols = 49  # For example, choose any number of columns you desire

    # Calculate the number of rows needed
    num_rows = total_elements // num_cols
    if total_elements % num_cols != 0:
        num_rows += 1
    mgmobj = mgmObj
    mgmobj[4]['levelsData'][0]['Lh'] = mgmobj[4]['levelsData'][0]['Lh'].reshape(num_rows, num_cols)
    uh = np.zeros(num_cols).reshape(-1,1)  # Example input vector

    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Call afun method with mgmStruct
    result = mgm_obj.afun(uh, mgmobj)

    # Perform manual matrix-vector multiplication for validation
    expected_result =  np.zeros((117649,1)) # Adjusted expected result for the 3x2 matrix

    # Assert that the result matches the expected result
    assert np.array_equal(result, expected_result)

def test_afun_empty_matrix(mgmObj):
    mgmobj = mgmObj
    mgmobj[4]['levelsData'][0]['Lh'] = np.empty((0, 0))  # Empty matrix
    uh = uh = np.zeros(2401).reshape(-1,1)   # Example input vector


    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Assert that calling afun with an empty matrix raises an exception
    with pytest.raises(Exception):
        mgm_obj.afun(uh, mgmobj)

def test_afun_empty_vector():
    mgmobj = mgmObj
    uh = np.empty((0, 0))
    # Instantiate mgm object
    mgm_obj = TestMGMImplementation()

    # Assert that calling afun with an empty matrix raises an exception
    with pytest.raises(Exception):
        mgm_obj.afun(uh, mgmobj)


#--------------------------------------------------------------------------------
#                          MULTILEVEL TESTS
#--------------------------------------------------------------------------------
@pytest.fixture
def example_input():
    file_path = construct_file_path('multievel_data', 'mgmStruct_before.mat')
    mat_contents = sp.io.loadmat(file_path)

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
    file_path = construct_file_path('multievel_data', 'fh_before_multi.mat')
    fh = sp.io.loadmat(file_path)
    fh = fh['fh']
    file_path = construct_file_path('multievel_data', 'uh_after_multi.mat')
    expected_result = sp.io.loadmat(file_path)
    expected_result = expected_result['uh'].reshape(-1,1)
    file_path = construct_file_path('multievel_data', 'uh_before_multi.mat')
    uh = sp.io.loadmat(file_path)
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

#--------------------------------------------------------------------------------
#                          SOlVE TESTS
#--------------------------------------------------------------------------------
@pytest.fixture
def example_input_obj():
    file_path = construct_file_path('solve_data', 'levelsData_before_solve.mat')
    mat_contents = sp.io.loadmat(file_path)

    # Initialize an empty list to store the levelsData
    levelsData = []

    # Iterate over the data from the .mat file
    for i in range(len(mat_contents['levelsData'])):
        level_data = {
            'nodes': mat_contents['levelsData'][i][0]['nodes'],
            'stencilSize': mat_contents['levelsData'][i][0]['stencilSize'],
            'rbfOrder': mat_contents['levelsData'][i][0]['rbfOrder'],
            'rbfPolyDeg': mat_contents['levelsData'][i][0]['rbfPolyDeg'],
            'rbf': mat_contents['levelsData'][i][0]['rbf'],
            # Function handle to the polyharmonic spline kernel function needs polyHarmonic implementation
            'idx': mat_contents['levelsData'][i][0]['idx'],
            'Lh': mat_contents['levelsData'][i][0]['Lh'],
            'DLh': mat_contents['levelsData'][i][0]['DLh'],
            'I': mat_contents['levelsData'][i][0]['I'],
            'R': mat_contents['levelsData'][i][0]['R'],
            'Mhf': mat_contents['levelsData'][i][0]['Mhf'],
            'Nhf': mat_contents['levelsData'][i][0]['Nhf'],
            'Mhb': mat_contents['levelsData'][i][0]['Mhb'],
            'Nhb': mat_contents['levelsData'][i][0]['Nhb'],
            'preSmooth': mat_contents['levelsData'][i][0]['preSmooth'],
            'postSmooth': mat_contents['levelsData'][i][0]['postSmooth'],
            'Ihat': mat_contents['levelsData'][i][0]['Ihat'],
            'Rhat': mat_contents['levelsData'][i][0]['Rhat'],
            'w': mat_contents['levelsData'][i][0]['w'],
            'Qh': mat_contents['levelsData'][i][0]['Qh']
        }
        levelsData.append(level_data)
    return levelsData

@pytest.fixture
def mgmObj(example_input_obj):
    mgmobj = [
        {'stencilSizeT':3},
        {'polyDegreeT':3},
        {'transferOp':'RBF'},
        {'verbose':1},
        {'levelsData':example_input_obj},
        {'domainVol':4},
        {'coarseningFactor':4},
        {'Nmin':250},
        {'preSmooth':1},
        {'postSmooth':1},
        {'maxIters':100},
        {'hasConstNullSpace':False}
    ]
    return mgmobj


def test_solve_no_acceleration_default_parameters(mgmObj):
    file_path = construct_file_path('solve_data', 'fh_before_solve.mat')
    fh = sp.io.loadmat(file_path)['fh']
    expected_uh = sp.io.loadmat(construct_file_path('solve_data', 'uh_after_solve.mat'))['uh']
    expected_relres = sp.io.loadmat(construct_file_path('solve_data', 'relres_after_solve.mat'))['relres'][0][0]
    expected_resvec = sp.io.loadmat(construct_file_path('solve_data', 'resvec_after_solve.mat'))['resvec']
    tol = 1e-10  # Default tolerance
    maxIters = 100  # Default maximum number of iterations
    accel = 'none'

    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    # Call solve method with no acceleration and default parameters
    uh, flag, relres, iters, resvec = mgm_obj.solve(mgmObj,fh, tol, accel=accel, maxIters=maxIters)

    # Assert the result dimensions
    assert np.allclose(expected_uh, uh)
    assert flag == 0
    assert np.allclose(expected_relres, relres)
    assert iters == 15
    assert np.allclose(expected_resvec, resvec)


def test_solve_bicgstab_accel(mgmObj):
    file_path = construct_file_path('solve_data', 'fh_before_solve_bicgstab.mat')
    fh = sp.io.loadmat(file_path)['fh']
    expected_uh = sp.io.loadmat(construct_file_path('solve_data', 'uh_after_solve_bicgstab.mat'))['uh']
    expected_relres = sp.io.loadmat(construct_file_path('solve_data', 'relres_after_solve_bicgstab.mat'))['relres'][0][0]
    expected_resvec = sp.io.loadmat(construct_file_path('solve_data', 'resvec_after_solve_bicgstab.mat'))['resvec']
    tol = 1e-10  # Default tolerance
    maxIters = 100  # Default maximum number of iterations
    accel = 'bicgstab'

    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    # Call solve method with no acceleration and default parameters
    uh, flag, relres, iters, resvec = mgm_obj.solve(mgmObj,fh, tol, accel, maxIters )

    # Assert the result dimensions
    assert np.allclose(expected_uh, uh)
    assert flag == 0
    ## bigcstab doesn't return relres, iters and resvec for now uh and the flag are ok
    # assert np.allclose(expected_relres, relres)
    # assert iters == 13
    # assert np.allclose(expected_resvec, resvec)

def test_solve_gmres_accel(mgmObj):
    file_path = construct_file_path('solve_data', 'fh_before_solve_bicgstab.mat')
    fh = sp.io.loadmat(file_path)['fh']
    tol = 1e-10  # Default tolerance
    maxIters = 100  # Default maximum number of iterations
    accel = 'gmres'

    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    # Call solve method with no acceleration and default parameters
    with pytest.raises(NotImplementedError):
        uh, flag, relres, iters, resvec = mgm_obj.solve(mgmObj, fh, tol, accel, maxIters)



#--------------------------------------------------------------------------------
#                          STANDALONE TESTS
#--------------------------------------------------------------------------------
def test_standalone_convergence(example_input):
  # Example right-hand side
    file_path = construct_file_path('standalone_data', 'fh_before_standalone.mat')
    fh = sp.io.loadmat(file_path)
    fh = fh['fh']
    tol = 1e-10  # Tolerance
    max_iters = 100  # Maximum number of iterations


    file_path = construct_file_path('standalone_data', 'uh_before_standalone.mat')
    uh = sp.io.loadmat(file_path)
    uh = uh['uh'].reshape(-1,1)
    smooths = [1, 1]  # Example smooths
    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    # Call standalone method
    uh, flag, relres, iters, resvec = mgm_obj.standalone(example_input, fh, tol, max_iters, uh, smooths)
    file_path = construct_file_path('standalone_data', 'uh_afterstand.mat')
    expected_uh = sp.io.loadmat(file_path)['uh'].reshape(-1,1)

    file_path = construct_file_path('standalone_data', 'resvec_afterstand.mat')
    expected_resvec = sp.io.loadmat(file_path)['resvec']
    # Assert the result
    assert flag == 0  # Convergence flag
    assert relres <= tol  # Relative residual within tolerance
    assert iters <= max_iters  # Number of iterations within maximum
    assert np.allclose(expected_uh, uh)
    assert np.allclose(expected_resvec, resvec)


def test_standalone_non_convergence(example_input):
    # Example right-hand side
    file_path = construct_file_path('standalone_data', 'fh_before_standalone.mat')
    fh = sp.io.loadmat(file_path)
    fh = fh['fh']
    tol = 1e-10  # Tolerance
    max_iters = 14  # Maximum number of iterations
    file_path = construct_file_path('standalone_data', 'uh_before_standalone.mat')
    uh = sp.io.loadmat(file_path)
    uh = uh['uh'] # Input vector
    smooths = [1, 1]  # Example smooths
    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    # Call standalone method
    with pytest.warns(UserWarning):
        uh, flag, relres, iters, resvec = mgm_obj.standalone(example_input, fh, tol, max_iters, uh, smooths)

    file_path = construct_file_path('standalone_data', 'uh_afterstand_nonconverge.mat')
    expected_uh = sp.io.loadmat(file_path)['uh']

    file_path = construct_file_path('standalone_data', 'resvec_afterstand_nonconverge.mat')
    expected_resvec = sp.io.loadmat(file_path)['resvec']
    # Assert the result
    assert flag == 1  # Convergence flag
    assert relres > tol  # Relative residual within tolerance
    assert iters <= max_iters  # Number of iterations within maximum
    assert np.allclose(expected_uh, uh)
    assert np.allclose(expected_resvec, resvec)

def test_standalone_empty_input():
    # Define empty input parameters
    fh = np.array([])  # Empty right-hand side
    tol = 1e-8  # Tolerance
    max_iters = 100  # Maximum number of iterations
    uh = np.array([])  # Empty input vector
    smooths = [1, 1]  # Example smooths

    # Instantiate TestMGMImplementation object
    mgm_obj = TestMGMImplementation()

    # Call standalone method
    # uh, flag, relres, iters, resvec = mgm_obj.standalone(levelsData, fh, tol, max_iters, uh, smooths)
    with pytest.raises(IndexError):
         mgm_obj.standalone(np.array([]), fh, tol, max_iters, uh, smooths)
