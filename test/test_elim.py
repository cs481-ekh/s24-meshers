#import pytest
import numpy as np
# from ..src.pymgm_test.utils.cyCodeBase  import elim
import elim # library for testing which links directly to wse class. 

def poisson_gen_base(n):
    h = 2 / (n+1)
    m = n
    yy, xx = np.meshgrid(-1 + np.arange(n+2) * h, -1 + np.arange(n+2) * h)
    xx = xx[1:n+1, 1:n+1]
    yy = yy[1:n+1, 1:n+1]
    x = np.column_stack([xx.flatten(), yy.flatten()])
    return x

n=8
x = poisson_gen_base(8) #  generate 64 points
base_case_lvl_1 = x

x1 = np.array([-0.77777778, -0.33333333, -0.77777778, -0.77777778, -0.11111111, -0.77777778, 0.11111111, 0.77777778, 0.77777778, -0.33333333, 0.33333333, -0.33333333, 0.11111111, 0.77777778, 0.33333333, 0.77777778])
x2 = np.array([-0.77777778, -0.77777778, -0.33333333, 0.77777778, -0.33333333, 0.33333333, -0.77777778, -0.33333333, -0.77777778, 0.11111111, -0.11111111, 0.77777778, 0.33333333, 0.33333333, 0.77777778, 0.77777778])

# expected coarsening in matlab with same input points
# coarsening factor 4 on 64 => 16 points
base_case_lvl_2 = np.column_stack((x1, x2))       
obj = elim.WeightedSampleElimination()  # constructor with no arguments, build with defaults
assert obj is not None


def test_build_default_wse():
    obj = elim.WeightedSampleElimination()  # constructor with no arguments, build with defaults
    assert obj is not None
    
def test_wse_is_tiling():
    obj = elim.WeightedSampleElimination()  # constructor with no arguments, build with defaults
    assert obj.IsTiling() == False

def test_elim_base_case():
    wse = elim.WeightedSampleElimination(Point2d, double, 2, int)  # constructor with 2D points, double type, 2 dimensions, int
    wse.Eliminate()
    t = wse.outputPoints # after eliminate(), output points should be lvl 2 points. 
    assert np.array_equal(base_case_lvl_2, t)
