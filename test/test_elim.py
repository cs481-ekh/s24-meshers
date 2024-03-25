import pytest
import numpy as np
from src.pymgm_test.utils import PcCoarsen


# Create an instance of the Elim class
PcCoarsen_2d = PcCoarsen.PcCoarsen2D()
wse = PcCoarsen.WeightedSampleElimination()  # constructor with no arguments, build with defaults


def test_build_default_wse():
    wse = PcCoarsen.WeightedSampleElimination()  
    assert wse is not None
    
def test_wse_is_tiling():
    wse = PcCoarsen.WeightedSampleElimination()  
    assert wse.IsTiling() == False
    
def test_wse_set_tiling():
    wse = PcCoarsen.WeightedSampleElimination()  
    wse.SetTiling(True)
    assert wse.IsTiling() == True
    
def test_wse_is_limiting():
    assert wse.IsWeightLimiting() == True   
    
def test_wse_set_limiting():
    wse = PcCoarsen.WeightedSampleElimination() 
    wse.SetWeightLimiting(False)
    assert wse.IsWeightLimiting() == False   
    
def test_get_param_alpha():
    wse = PcCoarsen.WeightedSampleElimination() 
    assert wse.GetParamAlpha() == 8.0   
        
def test_get_max_poisson_radius():
    wse = PcCoarsen.WeightedSampleElimination() 
    v=wse.GetMaxPoissonDiskRadius(2,16,0) 
    assert v == 0.1343212  
    


