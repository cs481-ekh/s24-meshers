import numpy as np
from scipy.sparse.linalg import bicgstab, LinearOperator, gmres
from scipy.sparse import csc_matrix

def solve(self,mgmobj, fh, tol=1e-8, accel='none', maxIters=None):
    levelsData = mgmobj[4]['levelsData']
    N = levelsData[0]['nodes'].shape[0]

    if maxIters is None:
        maxIters = mgmobj[10]['maxIters']

    if mgmobj[11]['hasConstNullSpace']:
        fh = np.append(fh, 0)
        uh0 = np.zeros(N + 1)
        mgmOp = self.multilevelcon
        matvecOp = self.afuncon
        mgmMethod = self.standalonecon
    else:
        uh0 = np.zeros(N).reshape(-1, 1)
        mgmOp = self.multilevel
        matvecOp = lambda x: self.afun(x, mgmobj)
        mgmMethod = self.standalone

    smooths = [mgmobj[8]['preSmooth'], mgmobj[9]['postSmooth']]

    if accel.lower() == 'gmres':
        A = csc_matrix(matvecOp(levelsData, np.eye(N)))
        b = fh
        uh, flag = gmres(A, b, tol=tol, maxiter=maxIters)
        iters = flag  # Note: scipy gmres returns iteration count or convergence flag
        relres = np.linalg.norm(b - A @ uh) / np.linalg.norm(b)
        if flag != 0:
            print(f"GMRES did not converge to a tolerance of {tol} in {maxIters} iterations")
    elif accel.lower() == 'bicgstab':
        A_operator = LinearOperator(shape=(N, N), matvec=matvecOp)
        b = fh
        uh, flag = bicgstab(A_operator, b, tol=tol, maxiter=maxIters)
        iters = flag[1]  # Note: scipy bicgstab returns iteration count and convergence flag
        relres = np.linalg.norm(b - A @ uh) / np.linalg.norm(b)
        if flag[1] != 0:
            print(f"BiCGStab did not converge to a tolerance of {tol} in {maxIters} iterations")
    elif accel.lower() == 'none':
        uh, flag, relres, iters, resvec = mgmMethod(levelsData, fh, tol, maxIters, uh0, smooths)
        if flag != 0:
            print(f"MGM did not converge to a tolerance of {tol} in {maxIters} iterations")
    else:
        raise ValueError(f"Unknown acceleration option {accel}. Choices are none, gmres, or bicgstab")

    return uh, flag, relres, iters, resvec


# Example usage:
# uh, flag, relres, iters = solve(mgmobj, fh, tol=1e-8, accel='bicgstab', maxIters=100)
