import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial import cKDTree
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

# Simple implementation for demo skips copies to locals, and skips range checking
    def constructor(self, Lh, x, domArea, hasConstNullspace, verbose):
        polyDim = (self.polyDegT+1)*(self.polyDegT+2)/2
        minStencil = math.ceil(max(1.5*polyDim,3));  # Heuristic
        computeDomArea = true
        self.hasConstNullspace = false
        verbose = false
        rbfOrderT = 0
        rbf = polyHarmonic
        interpMethod = 0

        N = x.shape[0] # get number of rows ( shape gives dimensions, then access row dim)
        p = math.floor(math.log(N / self.Nmin) / math.log(self.coarseningFactor)) # #Compute number of levels

        kdtree = cKDTree(x) # Build a cKDTree
        distances, indices = kdtree.query(x, k=2) # Perform nearest neighbor search
        domArea = x.shape[0] * np.mean(distances[:, 1])**2  # Compute domain area
        self.domainVol = domArea

        xc = np.empty(p + 1, dtype=object)
        Nc = np.zeros(p + 1)
        Nc[0] = N
        xc[0] = x

        #Build coarsened levels
        for j in range(2, p + 2):
            Nc[j] = math.floor(N / coarseningFactor ** (j - 1))
            xc[j] = PcCoarsen2D(xc[j - 1], Nc[j], domArea)  # starting a j=2, this fills in the next level (first level keep same)
            Nc[j] = xc[j].shape[0]

        levelsData = []

        # Initialize level data for each level
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

        levelsData[1]['nodes'] = x
        levelsData[0]['Lh'] = Lh
        levelsData[0]['w'] = np.ones(N)


        obj = {}
        obj['coarseningFactor'] = 4
        obj['levelsData'] = levelsData

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
