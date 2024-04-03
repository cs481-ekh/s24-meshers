
import math
def polyHarmonic(r2, ell, k):
    """
    Compute the polyharmonic spline kernel function
    :param rs: distance matrix
    :param ell: shape parameter
    :param k: order of the kernel
    :return: polyharmonic spline kernel function
    """
    if k == 1:
        phi = (r2 ** ell) * math.sqrt(r2)

    elif k == 2:
        phi = (2 * ell + 1) * (r2 ** (ell - 1)) * math.sqrt(r2)
    elif k == 3:
        phi = ((2 * ell + 1) * (2 * ell - 1)) * (r2 ** (ell - 2)) * math.sqrt(r2)
    else:
        raise NotImplementedError("Polynomial harmonic of order {} not implemented.".format(k))
    return phi
