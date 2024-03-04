import pytest
import numpy as np
#from ..src.pymgm_test.utils.cyCodeBase import PCCoarsen

import PcCoarsen # Main libary links all cyCodeBase for 2D


def test_PcCoarsen_2D

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