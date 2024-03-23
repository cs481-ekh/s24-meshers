
import PcCoarsen
import numpy as np

def poisson_gen_base(n):
    h = 2 / (n+1)
    m = n
    yy, xx = np.meshgrid(-1 + np.arange(n+2) * h, -1 + np.arange(n+2) * h)
    xx = xx[1:n+1, 1:n+1]
    yy = yy[1:n+1, 1:n+1]
    x = np.column_stack([xx.flatten(), yy.flatten()])
    return x

N=8
x = poisson_gen_base(N) #  generate 64 points
vol=4

# next level N, divide by coarsening factor
NC = len(x) / 4

# Create an instance of the PcCoarsen2D class
PcCoarsen_2d = PcCoarsen.PcCoarsen2D()

# Call the Coarsen method
# xc the next coarsened next level
# xc is doubles
# NC is double
# vol is double
xc = PcCoarsen_2d.Coarsen(x,NC,vol)
print("xc is:")
print(xc)

