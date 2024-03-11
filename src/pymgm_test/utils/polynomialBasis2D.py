import numpy as np

# Polynomial basis for 2d array
# Pts is a set of points to obtain a polynomial basis for (numpy array)
# Pts should be an n by 2 array (since a 2d point is a 2-tuple) where n is the number of points
# l is the maximum degree of the polynomial basis
# Returns a tuple (Pbasis, lapP) of the polynomial basis (Pbasis) and Laplacian matrix (lapP) [both are numpy arrays]
def poly_basis(Pts, l):
    # invalid degree (must be non-negative)
    if l < 0: return (None, None)

    terms = (l+1)*(l+2) // 2
    Pbasis = np.zeros((len(Pts), terms))
    lapP = np.zeros((len(Pts), terms))

    # fill the PolyBasis array (P) with the different terms (evaluated at the points)
    # e.g. for l = 1 would simply be a linear basis [1, x, y]
    # for l = 2 would include terms of up to degree 2 [1, x, y, x^2, x*y, y^2]

    for ind in range(len(Pts)):
        t = 0
        for d in range(l+1):
            for i in range(d+1):
                Pbasis[ind][t] = Pts[ind][0]**(d-i) * Pts[ind][1]**(i)
                if (d-i >= 2):
                    lapP[ind][t] += (d-i)*(d-i-1) * Pts[ind][0]**(d-i-2) * Pts[ind][1]**(i)
                if (i >= 2):
                    lapP[ind][t] += (i)*(i-1) * Pts[ind][0]**(d-i) * Pts[ind][1]**(i-2)
                t += 1

    return (Pbasis, lapP)
