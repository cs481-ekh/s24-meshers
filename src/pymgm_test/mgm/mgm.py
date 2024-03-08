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
    from ._solve import solve
    from ._multilevel import multilevel
    from ._standalone import standalone

    def afuncon(self):
        pass
    def multilevelcon(self):
        pass
    def standalonecon(self):
        pass
    @abstractmethod
    def buildInterOp(self, fineLevelStruct, coarseLevelStruct):
        pass

