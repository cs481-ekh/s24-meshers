import numpy as np
from ._multilevel import multilevel


def standalone(self,mgmStruct, fh, tol, max_iter, uh, smooths):
    iter_count = 0
    residual = np.inf
    resvec = np.ones((max_iter + 1, 1))
    nrmrhs = np.linalg.norm(fh)
    resvec[0] = nrmrhs
    reltol = tol * nrmrhs
    Lh = mgmStruct[0]['Lh']

    while residual > reltol and iter_count < max_iter:
        uh = multilevel(fh, mgmStruct, smooths, uh)
        rh = fh - np.dot(Lh , uh)
        iter_count += 1
        residual = np.linalg.norm(rh)
        resvec[iter_count] = residual

    relres = residual / nrmrhs
    resvec = resvec[:iter_count + 1]
    flag = 0

    if iter_count >= max_iter:
        flag = 1
        # Uncomment the line below to raise a warning if needed
        # warnings.warn('MGM failed to converge in {} iterations, relative residual={:.3e}'.format(max_iter, relres))

    return uh, flag, relres, iter_count, resvec

