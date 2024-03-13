from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np
from scipy.sparse.linalg import spsolve,  bicgstab, LinearOperator
import warnings

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



    def afun(self, uh, mgmobj):
        Lh = mgmobj[4]['levelsData'][0]['Lh']
        res = Lh.dot(uh)
        return res  # Assuming mgmobj is an array-like object with Lh attribute



    def solve(self, mgmobj, fh, tol=1e-8, accel='none', maxIters=None):
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
            # A = csc_matrix(matvecOp(levelsData, np.eye(N)))
            # b = fh
            # uh, flag = gmres(A, b, tol=tol, maxiter=maxIters) #python gmres proabaly not what we need
            # iters = flag  # Note: scipy gmres returns iteration count or convergence flag
            # relres = np.linalg.norm(b - A @ uh) / np.linalg.norm(b)
            # if flag != 0:
            #     print(f"GMRES did not converge to a tolerance of {tol} in {maxIters} iterations")
            raise NotImplementedError
        elif accel.lower() == 'bicgstab':
            A_operator = LinearOperator(shape=(N, N), matvec=matvecOp)
            b = fh
            uh, flag = bicgstab(A_operator, b, tol=tol, maxiter=maxIters)
            uh = uh.reshape(-1, 1)
            # resvec = b - A_operator.dot(uh)
            # relres = np.linalg.norm(resvec) / np.linalg.norm(b)
            resvec = None
            relres = None
            iters = None
            if flag != 0:
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

    def multilevel(self, fh, levelsData, smooths=None, uh=None):
        num_vcycles = 1
        if smooths is None or len(smooths) != 2:
            pre_smooth = 1
            post_smooth = 1
        else:
            pre_smooth = smooths[0]
            post_smooth = smooths[1]
        if uh is None:
            uh = np.zeros_like(fh)  # Use zero initial guess

        num_levels = len(levelsData) - 1

        rH = [None] * (num_levels + 1)
        deltaH = [None] * (num_levels + 1)

        rH[0] = fh
        deltaH[0] = uh

        for j in range(num_vcycles):

            for lvl in range(num_levels):
                uh = deltaH[lvl]
                fh = rH[lvl]
                # Smooth

                for k in range(pre_smooth):
                    tmpres = levelsData[lvl]['Nhf'].dot(uh) + fh
                    solution = spsolve(levelsData[lvl]['Mhf'], tmpres).reshape(-1, 1)
                    # solution, residuals, rank, s = np.linalg.lstsq(levelsData[lvl]['Mhf'], tmpres, rcond=None)
                    uh = solution

                deltaH[lvl] = uh
                # Defect
                # defect = fh - np.dot(levelsData[lvl]['Lh'], uh)

                defect = fh - levelsData[lvl]['Lh'].dot(uh)
                # Restrict
                # rH[lvl + 1] = np.dot(levelsData[lvl + 1]['R'], defect)
                rH[lvl + 1] = levelsData[lvl + 1]['R'].dot(defect)
                deltaH[lvl + 1] = np.zeros_like(rH[lvl + 1])

            # Coarse solve
            # rhs = np.dot(levelsData[lvl].Nhf, uh) + fh
            solution = spsolve(levelsData[num_levels]['DLh'], rH[num_levels])

            deltaH[-1] = solution

            for lvl in range(num_levels, 0, -1):
                # uh = deltaH[lvl] + np.dot(levelsData[lvl]['I'], deltaH[lvl + 1])
                # deltah1 =deltaH[lvl].reshape(-1,1)
                # lvlsdatlvl = levelsData[lvl-1];
                # tmp = levelsData[lvl-1]['I'].dot( deltaH[lvl].reshape(-1,1))

                uh = deltaH[lvl - 1].reshape(-1, 1) + levelsData[lvl - 1]['I'].dot(deltaH[lvl].reshape(-1, 1))
                fh = rH[lvl - 1]
                # Smooth
                for k in range(post_smooth):
                    # tmpres = np.dot(levelsData[lvl]['Nhf'], uh)
                    # tmpres = tmpres + fh
                    # solution, residuals, rank, s = np.linalg.lstsq(levelsData[lvl]['Mhf'], tmpres, rcond=None)
                    nhf = levelsData[lvl - 1]['Nhf'];
                    nuh = nhf.dot(uh);
                    nuhf = nuh + fh;
                    tmpres = levelsData[lvl - 1]['Nhf'].dot(uh) + fh
                    solution = spsolve(levelsData[lvl - 1]['Mhf'], tmpres).reshape(-1, 1)
                    uh = solution
                    # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
                    # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                    # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                    # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
                deltaH[lvl - 1] = uh

            # Check residual
            # rh = fh - np.dot(levelsData[lvl]['Lh'], uh)
            # residual[j] = np.linalg.norm(rh) / np.linalg.norm(fh)
            # print(f'iter={j}, ||r||={residual[j]:.4e}')
        return uh



    def standalone(self, mgmStruct, fh, tol, max_iter, uh, smooths):
        iter_count = 0
        residual = np.inf
        resvec = np.ones((max_iter + 1, 1))
        nrmrhs = np.linalg.norm(fh)
        resvec[0] = nrmrhs
        reltol = tol * nrmrhs
        Lh = mgmStruct[0]['Lh']

        while residual > reltol and iter_count < max_iter:
            uh = self.multilevel( fh, mgmStruct, smooths, uh)
            rh = fh - (Lh.dot(uh))
            iter_count += 1
            residual = np.linalg.norm(rh)
            resvec[iter_count] = residual

        relres = residual / nrmrhs
        resvec = resvec[:iter_count + 1]
        flag = 0

        if iter_count >= max_iter:
            flag = 1
            # Uncomment the line below to raise a warning if needed
            warnings.warn('MGM failed to converge in {} iterations, relative residual={:.3e}'.format(max_iter, relres))

        return uh, flag, relres, iter_count, resvec



    from ._multilevelcon import multilevelcon
    from ._standalonecon import standalonecon
    from ._afuncon import afuncon
    @abstractmethod
    def buildInterOp(self, fineLevelStruct, coarseLevelStruct):
        pass

