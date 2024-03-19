import numpy as np
from scipy.spatial import KDTree
from src.pymgm_test.utils.polynomialBasis2D import poly_basis
# fineLevelStruct is LevelsData
# coarseLevelStruct is LevelsData 
# interp is Boolean   
def buildInterpOp(fineLevelStruct, coarseLevelStruct, interp):
    fnodes = fineLevelStruct['nodes']
    cnodes = coarseLevelStruct['nodes']

    nf = len(fnodes)
    nc = len(cnodes)

    nd = coarseLevelStruct['stencilSize']

    row_index = np.array([np.arange(nf)] * nd)
    col_index = row_index
    interp_wghts = row_index

    tree = KDTree(cnodes)
    # use Faiss if dataset becomes too large where classic CPU computation is impractical

    # _ is the distance between nearest neighbors (not used in the Matlab code)
    # stencilSize is (hard-coded as) 3, though subject to change (just a note for myself)
    _, idx = tree.query(fnodes, k=nd)
    fineLevelStruct['nodes'] = fnodes
    fineLevelStruct['idx'] = idx

    rbfOrder = coarseLevelStruct['rbfOrder']
    rbfPolyDeg = coarseLevelStruct['rbfPolyDeg']
    rbf = coarseLevelStruct['rbf']

    dimPoly = (rbfPolyDeg+1)*(rbfPolyDeg+2)/2
    dimPoly = int(dimPoly)

    ZM = np.zeros((dimPoly,dimPoly))
    pe = np.zeros((dimPoly,1))
    pe[1][0] = 1

    Wf = 2

    nf = 1
    for i in range(nf):
        xe = fnodes[i]
        #print(xe)
        j = idx[i]
        x = cnodes[j,:] # selects rows with indicies in j (e.g. if j=[2 3] it will select 2nd and 3rd rows)
        xx = x[:,0]
        yy = x[:,1]

        #print("Tokat")
        #print(xx)
        #print(yy)
        xxt = xx.T  # Transpose
        yyt = yy.T

        rd2 = (xx - xxt) ** 2 + (yy - yyt) ** 2

        diffxe = x - xe[None, :]
        re2 = np.sum(diffxe**2, axis=1)

        stencilRad = 1
        diffxe / stencilRad
        P, _ = poly_basis(diffxe / stencilRad,rbfPolyDeg)
        print(P)

        #print(x)
        #print()
        #print(xe)
        #print()
        #print(xx)
        #print()
        #print(yy)
        #print()
        #print(xxt)
        #print()
        #print(yyt)
        #print()
        #print(rd2)
        #print()
        #print(diffxe)
        #print()
        #print(re2)
        #print()

    return

#Example
fineLevelStruct = {}
coarseLevelStruct = {}

coarseLevelStruct['nodes'] = np.array([
    [1,1],[0,1],[2,2],[2,1],[4,3],[3,1]
])
coarseLevelStruct['idx'] = None

coarseLevelStruct['rbfOrder'] = 3
coarseLevelStruct['rbfPolyDeg'] = 3
coarseLevelStruct['rbf'] = 3

#print(coarseLevelStruct['nodes'].shape)
fineLevelStruct['stencilSize'] = 3

fineLevelStruct['nodes'] = np.array([
    [1,1],[0,1],[2,2],[2,1],[4,3],[3,1],[3,5],[2,0],[4,1],[0,4],[0,3],[2,3]
])
fineLevelStruct['idx'] = None

#print(fineLevelStruct['nodes'].shape)
coarseLevelStruct['stencilSize'] = 3

buildInterpOp(fineLevelStruct, coarseLevelStruct, True)

#print(np.array([np.arange(6)] * 13).shape)