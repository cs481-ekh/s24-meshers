import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def multilevel(self,fh, levelsData, smooths=None, uh=None):
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

        for lvl in range(num_levels , 0, -1):
            # uh = deltaH[lvl] + np.dot(levelsData[lvl]['I'], deltaH[lvl + 1])
            # deltah1 =deltaH[lvl].reshape(-1,1)
            # lvlsdatlvl = levelsData[lvl-1];
            # tmp = levelsData[lvl-1]['I'].dot( deltaH[lvl].reshape(-1,1))

            uh = deltaH[lvl-1].reshape(-1,1) + levelsData[lvl-1]['I'].dot( deltaH[lvl].reshape(-1,1))
            fh = rH[lvl-1]
            # Smooth
            for k in range(post_smooth):
                # tmpres = np.dot(levelsData[lvl]['Nhf'], uh)
                # tmpres = tmpres + fh
                # solution, residuals, rank, s = np.linalg.lstsq(levelsData[lvl]['Mhf'], tmpres, rcond=None)
                nhf = levelsData[lvl-1]['Nhf'];
                nuh = nhf.dot(uh);
                nuhf = nuh + fh;
                tmpres = levelsData[lvl-1]['Nhf'].dot(uh) + fh
                solution = spsolve(levelsData[lvl-1]['Mhf'], tmpres)
                uh = solution
                # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
            deltaH[lvl-1] = uh

        # Check residual
        # rh = fh - np.dot(levelsData[lvl]['Lh'], uh)
        # residual[j] = np.linalg.norm(rh) / np.linalg.norm(fh)
        # print(f'iter={j}, ||r||={residual[j]:.4e}')
    return uh


