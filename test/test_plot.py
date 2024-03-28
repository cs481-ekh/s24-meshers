import os
import scipy as sp
from src.pymgm_test.mgm2d.mgm2d import mgm2D
import pytest

def construct_file_path(folder_name, file_name):
    # Construct the file path dynamically based on the current working directory
    if os.path.basename(os.getcwd()) != 'test':
        return os.path.join(os.getcwd(), 'test', folder_name, file_name)
    else:
        return os.path.join(folder_name, file_name)

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

def test_plot(mgmObj):
    mgm_obj = mgm2D()
    h = mgm_obj.plot(mgmObj)

    assert True

