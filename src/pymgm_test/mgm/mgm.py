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
    from .solve import solve
    from ._multilevel import multilevel
    from ._standalone import standalone
    from ._multilevelcon import multilevelcon
    from ._standalonecon import standalonecon
    from ._afuncon import afuncon
    @abstractmethod
    def buildInterOp(self, fineLevelStruct, coarseLevelStruct):
        pass

