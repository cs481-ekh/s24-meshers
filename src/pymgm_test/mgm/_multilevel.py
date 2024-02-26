import numpy as np


def multilevel(fh, levelsData, smooths, uh):
    num_vcycles = 1
    if smooths is None:
        pre_smooth = 1
        post_smooth = 1
    else:
        pre_smooth = smooths[0]
        post_smooth = smooths[1]
    if uh is None:
        uh = np.zeros_like(fh)  # Use zero initial guess

    pre_smooth, post_smooth = smooths

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
                uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
            deltaH[lvl] = uh
            # Defect
            defect = fh - np.dot(levelsData[lvl]['Lh'], uh)
            # Restrict
            rH[lvl + 1] = np.dot(levelsData[lvl + 1]['R'], defect)
            deltaH[lvl + 1] = np.zeros_like(rH[lvl + 1])

        # Coarse solve
        deltaH[lvl + 1] = np.linalg.solve(levelsData[lvl + 1]['DLh'], rH[lvl + 1])

        for lvl in range(num_levels, 0, -1):
            uh = deltaH[lvl] + np.dot(levelsData[lvl]['I'], deltaH[lvl + 1])
            fh = rH[lvl]
            # Smooth
            for k in range(post_smooth):
                uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhb'], np.dot(levelsData[lvl]['Nhb'], uh) + fh)
                # uh = np.linalg.solve(levelsData[lvl]['Mhf'], np.dot(levelsData[lvl]['Nhf'], uh) + fh)
            deltaH[lvl] = uh

        # Check residual
        # rh = fh - np.dot(levelsData[lvl]['Lh'], uh)
        # residual[j] = np.linalg.norm(rh) / np.linalg.norm(fh)
        # print(f'iter={j}, ||r||={residual[j]:.4e}')



