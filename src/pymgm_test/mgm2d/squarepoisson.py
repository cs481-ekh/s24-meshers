import numpy as np
import scipy.sparse as sp

def squarepoissond(n):
    h = 2 / (n+1)
    m = n
    # Set up the grid of values over the square [-1,1] x [-1,1]
    xx, yy = np.meshgrid(-1 + np.arange(n+2) * h, -1 + np.arange(n+2) * h)
   
    # Remove the values on the boundary from xx and yy
    xx = xx[1:n+1, 1:n+1]
    yy = yy[1:n+1, 1:n+1]

    # Form the second derivative operator
    Lh = sp.diags([1, -2, 1], [-1, 0, 1], shape=(m, m)).toarray()
    Lh = (sp.kron(sp.eye(m), Lh) + sp.kron(Lh, sp.eye(m)))/ h**2

    # Reshape the points to a (m*m)-by-2 array corresponding to (x,y) values
    x = np.column_stack([xx.flatten(), yy.flatten()])

    # Make up a solution
    uexact = np.sin(np.pi*x[:,0]**2)*np.cos(np.pi*x[:,1]**2)
    # Compute the right hand side for this solution
    fh = Lh @ uexact
   
    # The domain is a square with side length 2 so the area (or volume) is 4
    vol = 4

    return Lh, x, vol, fh, uexact


# Size of the system is n**2
n = 40
# Compute the sparse matrix Lh, point values x, volume of the domain vol, right hand side fh, exact solution uexact
Lh, x, vol, fh, uexact = squarepoissond(n)
# Solve the linear system using the sparse solver in SciPy.  This is equivalent to the backslash command in matlab
# You should be able to use MGM with this Lh, fh, vol, and x to get a solution similar to what the spsolve gives.
u = sp.linalg.spsolve(Lh,fh)
# Check that the solution is compute correctly
err = np.linalg.norm(u-uexact)
print("Error in the solution is:", err)