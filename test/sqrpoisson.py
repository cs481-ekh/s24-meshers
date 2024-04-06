import numpy as np
import scipy.sparse as sp


def squarepoissond(n):
    h = 2 / (n)
    m = n-1
    # Set up the grid of values over the square [-1,1] x [-1,1]
    xx, yy = np.meshgrid(-1 + np.arange(n + 2) * h, -1 + np.arange(n + 2) * h)

    # Remove the values on the boundary from xx and yy
    xx = xx[1:n , 1:n]
    yy = yy[1:n , 1:n]


    # Form the second derivative operator
    Lh = sp.diags([1, -2, 1], [-1, 0, 1], shape=(m, m)).toarray()
    Lh = (sp.kron(sp.eye(m), Lh) + sp.kron(Lh, sp.eye(m))) / h ** 2

    # Reshape the points to a (m*m)-by-2 array corresponding to (x,y) values
    x = np.column_stack([yy.flatten(),xx.flatten() ])

    # Make up a solution
    uexact = np.sin(np.pi * x[:, 0] ** 2) * np.cos(np.pi * x[:, 1] ** 2)
    uexact = uexact.reshape((-1, 1))
    # Compute the right hand side for this solution
    fh = Lh.dot(uexact)
    fh = fh.reshape((-1, 1))

    # The domain is a square with side length 2 so the area (or volume) is 4
    vol = 4

    return Lh, x, vol, fh, uexact