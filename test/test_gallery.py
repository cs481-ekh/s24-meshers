import pytest

from src.pymgm_test.utils.gallery import Gallery_m

@pytest.fixture

def test_load_matrix():
    g = Gallery_m()
    g.load_mat_test('../src/pymgm_test/utils/matricies/matfile.mat')
    print(g._det__str__())

test_load_matrix()