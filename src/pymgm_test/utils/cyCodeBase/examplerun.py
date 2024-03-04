# Example demonstrates how to compile SampleElim, import it, run and create WSE object, and call its member functions from python.

# To compile 'elim' library: 
# g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` bindings.cpp cySampleElim.h -o elim`python3-config --extension-suffix`  
import sys
print(sys.path)

import elim

wse = elim.WeightedSampleElimination()

print("hello after wse")
t=wse.IsTiling()
print("isTiling:", t)

wse.SetTiling(True)
st=wse.IsTiling()
print("after set(true), isTiling:", st)

w=wse.IsWeightLimiting()
print("IsWeightLimiting():",w) # on by default

wse.SetWeightLimiting(False)# set to off
wf=wse.IsWeightLimiting()
print("after set to false IsWeightLimiting():",wf)# check again



u=wse.GetParamAlpha()
print("param alpha:", u)


# get radius on 2 dimensions, with 16 samples, and size 0 triggers default behavior 
v=wse.GetMaxPoissonDiskRadius(2,16,0) 
print("max poisson radius:", v)


