# To compile 'elim' library: 
# g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` bindings.cpp cySampleElim.h -o elim`python3-config --extension-suffix`  


import elim

obj = elim.WeightedSampleElimination()

print("hello after obj")
t=obj.IsTiling()
print(t)
print("hello after isTiling")
