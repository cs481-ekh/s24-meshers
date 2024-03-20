import pytest
import numpy as np
from src.pymgm_test.utils.cyCodeBase import  elim
wse = elim.WeightedSampleElimination()  # constructor with no arguments, build with defaults


def test_build_default_wse():
    wse = elim.WeightedSampleElimination()  
    assert wse is not None
    
def test_wse_is_tiling():
    wse = elim.WeightedSampleElimination()  
    assert wse.IsTiling() == False
    
def test_wse_set_tiling():
    wse = elim.WeightedSampleElimination()  
    wse.SetTiling(True)
    assert wse.IsTiling() == True
    
def test_wse_is_limiting():
    assert wse.IsWeightLimiting() == True   
    
def test_wse_set_limiting():
    wse = elim.WeightedSampleElimination() 
    wse.SetWeightLimiting(False)
    assert wse.IsWeightLimiting() == False   
    
def test_get_param_alpha():
    wse = elim.WeightedSampleElimination() 
    assert wse.GetParamAlpha() == 8.0   
        
def test_get_max_poisson_radius():
    wse = elim.WeightedSampleElimination() 
    v=wse.GetMaxPoissonDiskRadius(2,16,0) 
    assert v == 0.1343212  
    


