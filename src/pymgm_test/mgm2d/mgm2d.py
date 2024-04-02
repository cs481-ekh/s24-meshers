import matplotlib.pyplot as plt
import numpy as np
import math
from src.pymgm_test.mgm.mgm import mgm
from src.pymgm_test.utils import PcCoarsen


class mgm2D(mgm):
    def __init__(self, Lh=None, x=None, domArea=None, hasConstNullspace=None, verbose=None):
        super().__init__()

        self.stencilSizeT = 3
        self.polyDegT = 0
        self.transferOp = 'RBF'
        self.verbose = 1

        if Lh is not None and x is not None and domArea is not None and hasConstNullspace is not None and verbose is not None:
            self.obj = self.constructor(Lh, x, domArea, hasConstNullspace, verbose)
        else:
            self.obj = None

    def plot(self, mgmobj):
        """
        Visualize the coarse levels of an MGM2D object

        Parameters:
            mgmobj : object
                MGM2D object

        Returns:
            h : list
                List of figure handles
        """
        if mgmobj is None:
            return []

        num_levels = len(mgmobj[4]['levelsData'])
        h = []

        for j in range(num_levels):
            plt.figure()
            nodes = mgmobj[4]['levelsData'][j]['nodes']
            plt.plot(nodes[:, 0], nodes[:, 1], '.')
            plt.title('Nodes on level {}, total={}'.format(j, len(nodes)))
            plt.xlabel('x')
            plt.ylabel('y')
            h.append(plt.gca())
        plt.show()
        return h

    def constructor(self, Lh, x, domArea, hasConstNullspace, verbose):
        # Actual construction here
        #Compute number of levels
        p = math.floor(math.log(N / Nmin) / math.log(coarseningFactor));

        levelsData = []

        # Create level data for each level
        for _ in range(p + 1):
            level_data = {
                'nodes': [],
                'stencilSize': self.stencilSizeT,
                'rbfOrder': self.rbfOrderT,
                'rbfPolyDeg': self.polyDegT,
                'rbf': self.rbf,
                'idx': [],
                'Lh': [],
                'DLh': [],
                'I': [],
                'R': [],
                'Mhf': [],
                'Nhf': [],
                'Mhb': [],
                'Nhb': [],
                'preSmooth': self.preSmooth,
                'postSmooth': self.postSmooth,
                'Ihat': 1,
                'Rhat': 1,
                'w': [],
                'Qh': 0
            }
            levelsData.append(level_data)



        obj = {}
        obj['coarseningFactor'] = 4

        return obj


    def buildInterpOp(self,fineLevelStruct, coarseLevelStruct, interpMethod):
        """
        Build interpolation operator for MGM2D.

        Parameters:
            fineLevelStruct : dict
                Fine level structure
            coarseLevelStruct : dict
                Coarse level structure
            interpMethod : str
                Interpolation method ('RBF' or 'GMLS')

        Returns:
            fineLevelStruct : dict
                Fine level structure with interpolation operator added
        """
        # Implementation of buildInterpOp method
        # This is just a placeholder, you need to implement the actual logic
        # based on your MATLAB implementation

        # For example, let's just return a random interpolation operator
        fineLevelStruct['interpolationOperator'] = np.random.rand(5, 5)
        return fineLevelStruct
