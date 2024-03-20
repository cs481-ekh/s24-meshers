import pytest
from src.pymgm_test.utils.gallery import Gallery_m

@pytest.mark.parametrize(
    "filenm, det",
    [
        ("src/pymgm_test/utils/matrices/matfile.mat", -77),
        ("src/pymgm_test/utils/matrices/matfile_2.mat", -45512),
        ("src/pymgm_test/utils/matrices/matfile_3.mat", 1261),
        ("src/pymgm_test/utils/matrices/matfile_4.mat", -0.0)
    ]
)

# The test cases below ensure that the matlab files arei beng loaded in properly 
# (containing specific data about the matrix and its determinant) 
def test_load_matrix(filenm, det):
    g = Gallery_m()
    g.load_mat_test(filenm)
    print(g._det__str__())
    assert g._det__str__() == str(det) # The determinant
