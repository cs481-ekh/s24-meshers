import numpy as np
from scipy.spatial import KDTree
from src.pymgm_test.utils.polynomialBasis2D import poly_basis
from src.pymgm_test.utils.polyHarmonic import polyHarmonic
from scipy.sparse import csr_matrix

# not necessary
from tqdm import tqdm # used for checking progress (wrap it around a range())


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
    col_index = np.array(row_index)
    interp_wghts = np.array(row_index, dtype=float)

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
            
        interp_wghts[:,i] = wghts[0:nd].flatten() # turns to 1d array
        col_index[:,i] = j
        #print(col_index)

    row_index1d = row_index.reshape(-1).flatten()
    col_index1d = col_index.reshape(-1).flatten()
    interp_wghts1d = interp_wghts.reshape(1,-1).flatten()

    # Sparse matrix
    fineLevelStruct['I'] = csr_matrix((interp_wghts1d, (row_index1d, col_index1d)), shape=(nf, nc), dtype=np.double)
    print(csr_matrix)

    return fineLevelStruct
