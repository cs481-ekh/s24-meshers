import matplotlib.pyplot as plt
import numpy as np
import math
from src.pymgm_test.mgm.mgm import mgm
from scipy.spatial import cKDTree
from src.pymgm_test.utils.polyHarmonic import polyHarmonic
import time
from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm
from src.pymgm_test.mgm2d.buildInterpOp import buildInterpOp
from src.pymgm_test.utils import PcCoarsen
import scipy.sparse as sp
# from src.pymgm_test.utils import PcCoarsen


class mgm2D(mgm):
    def __init__(self, Lh=None, x=None, domArea=None, hasConstNullspace=None, verbose=None):
        super().__init__()

        self.stencilSizeT = 3
        self.polyDegT = 0
        self.transferOp = 'RBF'
        self.verbose = 1
        self.coarsening_factor = 4

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


    def constructor(self, Lh, x, domArea=None, hasConstNullspace=False, verbose=True):
        # Size of the stencil for the interpolation operator
        stencilSizeT = self.stencilSizeT
        # Polynomial degree of precision of the interpolation operator
        polyDegT = self.polyDegT
        # Type of transfer operator
        transferOp = self.transferOp
        # Coarsening factor for the coarser levels
        coarseningFactor = self.coarsening_factor
        # Minimum size of the coarsest level
        Nmin = self.Nmin
        # Number of pre and post smooths
        preSmooth = self.pre_smooth
        postSmooth = self.post_smooth

        # Do some range checking on the stencil sizes for the interpolation operator
        if stencilSizeT < 1:
            print(
                'Stencil size for the interpolation operator must be greater than or equal to 1. Changing this value to 1.')
            stencilSizeT = 1

        if polyDegT < 0:
            print(
                'Polynomial precision for the interpolation operator must be greater than zero. Changing this value to 0.')
            polyDegT = 0

        # Min stencil size for the interpolation operator has to be larger than the
        # dimension of the space of bivariate polynomials of degree polyDegT
        polyDim = (polyDegT + 1) * (polyDegT + 2) / 2
        minStencil = int(np.ceil(max(1.5 * polyDim, 3)))  # Heuristic
        if stencilSizeT <= polyDim:
            print('Interpolation operator stencil must be larger than %d. Changing this value to %d.' % (
            polyDim, minStencil))
            stencilSizeT = minStencil

            # Compute the domain area
        if domArea:
            computeDomArea = True
        else:
            computeDomArea = False

        if not hasConstNullspace:
            hasConstNullspace = False
        else:
            hasConstNullspace = True

        self.hasConstNullspace = hasConstNullspace

        if verbose:
            verbose = True

        rbfOrderT = 0
        rbf = polyHarmonic

        if polyDegT > 2:
            raise ValueError(
                'Polynomial degree for transfer operator not supported.  Only constant (deg=0) and linear (deg=1) are presently supported.')

        # Assuming transferOp is a string variable containing the transfer operator type

        if transferOp.lower() == 'rbf':
            interpMethod = 0
        elif transferOp.lower() == 'gmls':
            interpMethod = 1
        else:
            raise ValueError('Invalid transfer operator, choices are RBF or GMLS')

        N = x.shape[0]
        p = int(np.floor(np.log(N / Nmin) / np.log(coarseningFactor)))

        if verbose:
            print('Building coarse node sets, N=%d, levels=%d...\n', N, p + 1)

        if computeDomArea:
            tree = cKDTree(x)
            d, _ = tree.query(x, k=2)
            domArea = N * np.mean(d[:, 1]) ** 2

        self.domainVol = domArea

        stime = time.time()
        xc = np.empty(p + 1, dtype=object)
        Nc = np.zeros(p + 1)

        Nc[0] = N
        xc[0] = x

        for j in range(1, p + 1):
            Nc[j] = int(N / (coarseningFactor ** (j)))
            if verbose:
                print('Building coarse node set Nc=%d' % Nc[j])
            xc[j] = np.array(PcCoarsen.PcCoarsen2D().Coarsen(xc[j - 1], int(Nc[j]), float(domArea)))

            Nc[j] = len(xc[j])

        if verbose:
            etime = time.time() - stime
            print('Done building coarse node sets. Construction time = %1.2e\n' % etime)

        # Initialize the structure for the various levels
        levelsData = []
        for j in range(p + 1):
            levelsData.append({
                'nodes': xc[j],
                'stencilSize': stencilSizeT,
                'rbfOrder': 0,
                'rbfPolyDeg': polyDegT,
                'rbf': rbf,  # Function handle to the polyharmonic spline kernel function
                'idx': [],
                'Lh': [],
                'DLh': [],
                'I': [],
                'R': [],
                'Mhf': [],
                'Nhf': [],
                'Mhb': [],
                'Nhb': [],
                'preSmooth': preSmooth,
                'postSmooth': postSmooth,
                'Ihat': 1,
                'Rhat': 1,
                'w': [],
                'Qh': 0
            })

        # Intergrid transfer operators
        if verbose:
            print('Building Interp/Restriction operators, #levels=%d...' % (p + 1))

        stime = time.time()
        levelsData[0]['Lh'] = Lh
        levelsData[0]['w'] = np.ones(N)

        for j in range(1, p + 1):
            levelsData[j]['nodes'] = xc[j]

            # Build interpolation operator
            levelsData[j - 1] = buildInterpOp(levelsData[j - 1], levelsData[j], interpMethod)

            # Restriction is transpose of interpolation
            levelsData[j]['R'] = levelsData[j - 1]['I'].T

            # Galerkin coarse level operator
            levelsData[j]['Lh'] = levelsData[j]['R'] @ levelsData[j - 1]['Lh'] @ levelsData[j - 1]['I']

            # Convert 'Lh' to CSC format
            levelsData[j]['Lh'] = sp.csc_matrix(levelsData[j]['Lh'])

            # Re-order nodes to get nice banded structure in the operators
            id = rcm(levelsData[j]['Lh'])
            levelsData[j]['Lh'] = levelsData[j]['Lh'][id][:, id]
            levelsData[j - 1]['I'] = levelsData[j - 1]['I'][:, id]
            levelsData[j]['R'] = levelsData[j]['R'][id, :]
            levelsData[j]['nodes'] = levelsData[j]['nodes'][id, :]

            # Smoother - Gauss-Seidel
            # Forward GS
            # Forward GS
            levelsData[j - 1]['Mhf'] = sp.tril(levelsData[j - 1]['Lh'], k=0).tocsr()
            levelsData[j - 1]['Nhf'] = -sp.triu(levelsData[j - 1]['Lh'], k=1).tocsr()

            # Backward GS
            levelsData[j - 1]['Mhb'] = sp.triu(levelsData[j - 1]['Lh'], k=0).tocsr()
            levelsData[j - 1]['Nhb'] = -sp.tril(levelsData[j - 1]['Lh'], k=-1).tocsr()

            # Constraint for Poisson problem from the Galerkin operator
            levelsData[j]['w'] = levelsData[j]['R'] @ levelsData[j - 1]['w']

            if verbose:
                # Print diagnostics
                # Print diagnostics
                sparsity = 1 - np.count_nonzero(levelsData[j]['Lh'].toarray()) / np.size(levelsData[j]['Lh'].toarray())
                print('level={}, unknowns={}, non-zeros={}, sparsity={:.3f}'.format(j - 1, Nc[j],
                                                                                    np.count_nonzero(
                                                                                        levelsData[j]['Lh'].toarray()),
                                                                                    sparsity))

        if hasConstNullspace:
            # Add the constraint to the coarse-level solver
            levelsData[p + 1]['Lh'] = np.block([[levelsData[p + 1]['Lh'], levelsData[p + 1]['w'][..., None]],
                                                [levelsData[p + 1]['w'], 0]])

        levelsData[p]['DLh'] = levelsData[p]['Lh']

        if verbose:
            etime = time.time() - stime
            print('Done. Transfer operator build time %1.2e\n' % etime)

        obj = [
            {'stencilSizeT': stencilSizeT},
            {'polyDegreeT': polyDegT},
            {'transferOp': transferOp},
            {'verbose': verbose},
            {'levelsData': levelsData},
            {'domainVol': self.domainVol},
            {'coarseningFactor': coarseningFactor},
            {'Nmin': self.Nmin},
            {'preSmooth': preSmooth},
            {'postSmooth': postSmooth},
            {'maxIters': self.max_iters},
            {'hasConstNullSpace': hasConstNullspace},
        ]
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
