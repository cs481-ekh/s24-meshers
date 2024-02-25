from abc import ABC, abstractmethod, abstractstaticmethod


class mgm(ABC):
    def __init__(self):
        self.levels_data = None
        self.domain_vol = None
        self.coarsening_factor = None
        self.Nmin = 250
        self.pre_smooth = 1
        self.post_smooth = 1
        self.max_iters = 100
        self.has_const_nullspace = False

    from ._afun import afun
    def solve(self, mgmobj, fh, tol, accel, maxIters):
        pass


    def multilevel(self, fh, multilevelStruct, smooths, uh):
        pass


    def standalone(self, levelsData, fh, tol, max_iter, uh, smooths):
        pass
