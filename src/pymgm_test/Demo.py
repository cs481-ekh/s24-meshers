from src.pymgm_test.mgm2d.mgm2d import mgm2D
from src.pymgm_test.utils.sqrpoisson import squarepoissond
import matplotlib.pyplot as plt
import numpy as np
# Create the square poisson problem
Lh, x, vol, fh, uexact = squarepoissond(50)

# Create the MGM object
mgm = mgm2D(Lh,x,vol,False,1)

# Plot the coarse levels
mgm.plot(mgm.obj)
# Run the MGM solver
uh,flag,relres,iters,resvec = mgm.solve(mgm.obj,fh,1e-10,'none',100)
# Scatter plot of the solution
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=uh, cmap='viridis', s=20)
plt.colorbar(label='Solution')
plt.title('Scatter plot of the solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# np.set_printoptions(threshold=np.inf)
print("solution: ",uh)
print("\n")
print("convergence flag: ",flag)
print("\n")
print("relres: ",relres)
print("\n")
print("iterations: ",iters)
print("\n")
print("residual vector: \n",resvec)
print("\n")