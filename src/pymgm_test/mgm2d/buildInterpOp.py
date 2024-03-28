import numpy as np
from scipy.spatial import KDTree
from src.pymgm_test.utils.polynomialBasis2D import poly_basis
from scipy.io import loadmat
from src.pymgm_test.utils.polyHarmonic import polyHarmonic
from scipy.sparse import coo_matrix # sparse() matlab equivalent, for generating sparse matrices

# fineLevelStruct is LevelsData
# coarseLevelStruct is LevelsData 
# interp is Boolean   
def buildInterpOp(fineLevelStruct, coarseLevelStruct, interp):
    fnodes = fineLevelStruct['nodes']
    cnodes = coarseLevelStruct['nodes']

    nf = len(fnodes)
    nc = len(cnodes)

    nd = coarseLevelStruct['stencilSize']

    row_index = np.array([np.arange(nf)] * nd, dtype=float)
    col_index = np.array(row_index, dtype=float)
    interp_wghts = row_index

    tree = KDTree(cnodes)
    # use Faiss if dataset becomes too large where classic CPU computation is impractical

    # _ is the distance between nearest neighbors (not used in the Matlab code)
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
    pe[0][0] = 1

    Wf = 2

    nf = 1
    for i in range(nf):
        xe = fnodes[i]
        #print(xe)
        j = idx[i]
        x = cnodes[j,:] # selects rows with indicies in j (e.g. if j=[2 3] it will select 2nd and 3rd rows)
        xx = x[:,0]

        xx = xx.reshape(-1,1)
        yy = x[:,1]
        yy = yy.reshape(-1,1)

        xxt = xx.T  # Transpose
        yyt = yy.T

        rd2 = (xx - xxt)**2 + (yy - yyt)**2

        diffxe = x - xe[None, :]
        re2 = np.sum(diffxe**2, axis=1)

        stencilRad = 1
        diffxe / stencilRad
        P, _ = poly_basis(diffxe / stencilRad,rbfPolyDeg)
        wghts = None

        # interp = 1
        if interp:
            W = np.exp((-Wf*re2)/(1 + re2))**2
            W = W.reshape(-1,1)
            Q, R = np.linalg.qr(W * P)
            r_pe = np.linalg.solve(R, pe)
            wghts = W * Q * r_pe

        # interp = 0
        else:
            A = rbf(rd2,rbfOrder,1)
            P = P.reshape(1,-1)
            A = np.concatenate((A, P), axis = 0)
            P = P.reshape(-1, 1)
            last_col = np.concatenate((P, ZM), axis = 0)
            A = np.concatenate((A, last_col), axis = 1)
            b = (rbf(re2,rbfOrder,1)).reshape(-1, 1)
            b = np.concatenate((b, pe), axis = 0)
            wghts = np.linalg.solve(A, b)
        
        interp_wghts[:,i] = wghts[0:nd].flatten()
        col_index[:,i] = j

    fineLevelStruct['I'] = coo_matrix(interp_wghts(row_index, col_index), shape=(nf, nc))
    return fineLevelStruct

#Example parameters 
fineLevelStruct = {}
coarseLevelStruct = {}

coarseLevelStruct['nodes'] = loadmat('src/pymgm_test/mgm2d/coarseparams.mat')['cond'] 
coarseLevelStruct['idx'] = None
coarseLevelStruct['rbfOrder'] = 0
coarseLevelStruct['rbfPolyDeg'] = 0
coarseLevelStruct['rbf'] = polyHarmonic
#print(fineLevelStruct['nodes'].shape)
coarseLevelStruct['stencilSize'] = 3


#print(coarseLevelStruct['nodes'].shape)
fineLevelStruct['stencilSize'] = 3
fineLevelStruct['nodes'] = loadmat('src/pymgm_test/mgm2d/fineparams.mat')['find'] 
#np.array([[1,1],[0,1],[2,2],[2,1],[4,3],[3,1],[3,5],[2,0],[4,1],[0,4],[0,3],[2,3]])
fineLevelStruct['idx'] = None
fineLevelStruct['I'] = None

buildInterpOp(fineLevelStruct, coarseLevelStruct, True)

#print(np.array([np.arange(6)] * 13).shape)