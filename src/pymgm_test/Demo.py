from src.pymgm_test.mgm2d.mgm2d import mgm2D
from src.pymgm_test.utils.sqrpoisson import squarepoissond
# Create the square poisson problem
Lh, x, vol, fh, uexact = squarepoissond(50)

# Create the MGM object
mgm = mgm2D(Lh,x,vol,False,1)

# Plot the coarse levels
mgm.plot(mgm.obj)

# Run the MGM solver
uh,flag,relres,iters,resvec = mgm.solve(mgm.obj,fh,1e-10,'none',100)

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