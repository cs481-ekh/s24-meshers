import numpy as np

from scipy.spatial import cKDTree
from src.pymgm_test.utils.polyHarmonic import polyHarmonic  # You need to import util module or provide implementation for it
import time

def constructor(self,Lh, x, domArea=None, hasConstNullspace=False, verbose=True):
    # Size of the stencil for the interpolation operator
    stencilSizeT = self.stencilSizeT
    # Polynomial degree of precision of the interpolation operator
    polyDegT = self.polyDegT
    # Type of transfer operator
    transferOp = self.transferOp
    # Coarsening factor for the coarser levels
    coarseningFactor = self.coarseningFactor
    # Minimum size of the coarsest level
    Nmin = self.Nmin
    # Number of pre and post smooths
    preSmooth = self.preSmooth
    postSmooth = self.postSmooth



    # Do some range checking on the stencil sizes for the interpolation operator
    if stencilSizeT < 1:
        print('Stencil size for the interpolation operator must be greater than or equal to 1. Changing this value to 1.')
        stencilSizeT = 1

    if polyDegT < 0:
        print('Polynomial precision for the interpolation operator must be greater than zero. Changing this value to 0.')
        polyDegT = 0

    # Min stencil size for the interpolation operator has to be larger than the
    # dimension of the space of bivariate polynomials of degree polyDegT
    polyDim = (polyDegT + 1) * (polyDegT + 2) / 2
    minStencil = int(np.ceil(max(1.5 * polyDim, 3)))  # Heuristic
    if stencilSizeT <= polyDim:
        print('Interpolation operator stencil must be larger than %d. Changing this value to %d.' % (polyDim, minStencil))
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

    rbfOrderT =  0
    rbf = polyHarmonic



    if polyDegT > 2:
        raise ValueError('Polynomial degree for transfer operator not supported.  Only constant (deg=0) and linear (deg=1) are presently supported.')

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
        print('Building coarse node sets, N=%d, levels=%d...\n',N,p+1)

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
        Nc[j] = int(N / coarseningFactor ** (j - 1))
        if verbose:
            print('Building coarse node set Nc=%d' % Nc[j])
        # xc[j] = PcCoarsen2D(xc[j - 1], Nc[j], domArea)
        Nc[j] = xc[j].shape[0]

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
            'rbf': polyHarmonic,  # Function handle to the polyharmonic spline kernel function
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
        levelsData[j]['Lh'] = np.dot(np.dot(levelsData[j]['R'], levelsData[j - 1]['Lh']), levelsData[j - 1]['I'])
        levelsData[j]['w'] = np.dot(levelsData[j]['R'], levelsData[j - 1]['w'])

    if verbose:
        etime = time.time() - stime
        print('Done. Transfer operator build time %1.2e\n' % etime)

    obj =  [
        {'stencilSizeT':stencilSizeT},
        {'polyDegreeT':polyDegT},
        {'transferOp':transferOp},
        {'verbose':verbose},
        {'levelsData':levelsData},
        {'domainVol':self.domainVol},
        {'coarseningFactor':coarseningFactor},
        {'Nmin':self.Nmin},
        {'preSmooth':preSmooth},
        {'postSmooth':postSmooth},
        {'maxIters':self.maxIters},
        {'hasConstNullSpace':hasConstNullspace},
    ]
    return obj
