from abc import ABC, abstractmethod

class mgm(ABC):
    def __init__(self):
        pass

    def afun(self, uh,mgmobj):
        raise NotImplementedError("Method not yet implemented")

    def solve(self,mgmobj,fh,tol,accel,maxIters):
        raise NotImplementedError("Method not yet implemented")

    def multilevel(self,fh,multilevelStruct,smooths,uh):
        raise NotImplementedError("Method not yet implemented")

    def standalone(self,levelsData,fh,tol,max_iter,uh,smooths):
        raise NotImplementedError("Method not yet implemented")
