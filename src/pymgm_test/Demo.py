from src.pymgm_test.mgm2d.mgm2d import mgm2D
from src.pymgm_test.mgm2d.squarepoisson import squarepoissond
# Create the square poisson problem
Lh, x, vol, fh, uexact = squarepoissond(50)

# Create the MGM object
mgm = mgm2D(Lh,x,vol,False,1)

# Plot the coarse levels
mgm.plot(mgm.obj)

# Run the MGM solver
uh,flag,relres,iters,resvec = mgm.solve(mgm.obj,fh,1e-10,'none',100)

print(uh)
print("/n")
print(flag)
print("/n")
print(relres)
print("/n")
print(iters)
print("/n")
print(resvec)
print("/n")