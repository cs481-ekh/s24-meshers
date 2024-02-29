import numpy as np


def multilevel(self,fh, levelsData, smooths, uh):
    num_vcycles = 1
    if smooths is None:
        pre_smooth = 1
        post_smooth = 1
    else:
        pre_smooth = smooths[0]
        post_smooth = smooths[1]
    if uh is None:
        uh = np.zeros_like(fh)  # Use zero initial guess


    num_levels = len(levelsData) - 2

    rH = [None] * (num_levels + 2)
    deltaH = [None] * (num_levels + 2)

    rH[0] = fh
    deltaH[0] = uh

    for j in range(num_vcycles):

        for lvl in range(num_levels):
            uh = deltaH[lvl]
            fh = rH[lvl]
            # Smooth

            for k in range(pre_smooth):
                tmpres = np.dot(levelsData[lvl]['Nhf'], uh)
                tmpres = tmpres + fh
                solution, residuals, rank, s = np.linalg.lstsq(levelsData[lvl]['Mhf'], tmpres, rcond=None)
                uh = solution

            deltaH[lvl] = uh
            # Defect
            defect = fh - np.dot(levelsData[lvl]['Lh'], uh)
            # Restrict
            rH[lvl + 1] = np.dot(levelsData[lvl + 1]['R'], defect)
            deltaH[lvl + 1] = np.zeros_like(rH[lvl + 1])



        # Coarse solve
        # rhs = np.dot(levelsData[lvl].Nhf, uh) + fh
        #
        # # Use numpy.linalg.lstsq to solve the least squares problem
        solution, residuals, rank, s = np.linalg.lstsq(levelsData[num_levels+1]['DLh'], rH[num_levels+1], rcond=None)
        deltaH[-1] = solution
         # deltaH[-1] = np.dot(levelsData[num_levels + 1]['DLh'], np.linalg.inv(rH[num_levels + 1]))

        #deltaH[lvl_counter ] = np.linalg.solve(levelsData[lvl_counter ]['DLh'], rH[lvl_counter ])

        for lvl in range(num_levels, 0, -1):
            uh = deltaH[lvl] + np.dot(levelsData[lvl]['I'], deltaH[lvl + 1])
            fh = rH[lvl]
            # Smooth
            for k in range(post_smooth):
                tmpres = np.dot(levelsData[lvl]['Nhf'], uh)
                tmpres = tmpres + fh
                solution, residuals, rank, s = np.linalg.lstsq(levelsData[lvl]['Mhf'], tmpres, rcond=None)
                uh = solution
                # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
            deltaH[lvl] = uh

        # Check residual
        # rh = fh - np.dot(levelsData[lvl]['Lh'], uh)
        # residual[j] = np.linalg.norm(rh) / np.linalg.norm(fh)
        # print(f'iter={j}, ||r||={residual[j]:.4e}')
    return deltaH[0]


