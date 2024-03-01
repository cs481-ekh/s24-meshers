import pytest

from src.pymgm_test.utils.gallery import Gallery_m

# The test cases below ensure that the matlab files arei beng loaded in properly 
# (containing specific data about the matrix and its determinant) 

def test_load_matrix(filename, det):
    g = Gallery_m()
    g.load_mat_test(filename)
    print(g._det__str__())
    assert g._det__str__() == str(det) # The determinant

#print(os.getcwd())

# 3x3 matrix
#[[1, -3, 4]; [-6, -6, 7]; [3, 2, 1]]
test_load_matrix("src/pymgm_test/utils/matrices/matfile.mat", -77)

# 6x6 matrix
#[[2, -6, 5, 7, 10, 11]; [2, -16, 5, 10, 11, 3]; [7, -7, 6, 14, 12, 21]; 
#[3, 0, -1, 8, 2, 9]; [-1, -10, -5, 16, 20, 15]; [13, -18, 12, 5, -9, -11]]
test_load_matrix("src/pymgm_test/utils/matrices/matfile_2.mat", -45512) 

# 4x4 matrix
#[[5,1,2,4]; [4,5,3,2]; [7,-8,6,3]; [-1,1,4,5]]
test_load_matrix("src/pymgm_test/utils/matrices/matfile_3.mat", 1261)

# 4x4 matrix
#[[1,0,3,0]; [2,0,0,3]; [1,0,0,0]; [0,0,0,1]]
# -0.0 has to be used otherwise the test won't run 
test_load_matrix("src/pymgm_test/utils/matrices/matfile_4.mat", -0.0)
