from sqrpoisson import squarepoissond
import os
import numpy as np
import scipy as sp
import pytest

def construct_file_path(folder_name, file_name):
    # Construct the file path dynamically based on the current working directory
    if os.path.basename(os.getcwd()) != 'test':
        return os.path.join(os.getcwd(), 'test', folder_name, file_name)
    else:
        return os.path.join(folder_name, file_name)

def test_squarepoisson():
     expected_Lh = sp.io.loadmat(construct_file_path('sqrpoisson_data', 'sqrpoisson_Lh.mat'))['Lh'].toarray()
     expected_x = sp.io.loadmat(construct_file_path('sqrpoisson_data', 'sqrpoisson_x.mat'))['x']
     expected_vol = sp.io.loadmat(construct_file_path('sqrpoisson_data', 'sqrpoisson_vol.mat'))['vol']
     expected_uexact = sp.io.loadmat(construct_file_path('sqrpoisson_data', 'sqrpoisson_uexact.mat'))['uexact']
     Lh, x, vol, fh, uexact = squarepoissond(50)

     assert np.allclose(Lh.toarray(), expected_Lh)
     assert np.allclose(x, expected_x)
     assert np.allclose(vol, expected_vol)
     assert np.allclose(uexact, expected_uexact)