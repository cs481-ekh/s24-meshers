import numpy as np

def polyHarmonic(r2,ell,k):
    match k:
        case 1:
            phi = r2**(ell)*np.sqrt(r2)
        case 2:
            phi = (2*ell+1)*(r2**(ell-1)*np.sqrt(r2))
        case 3:
            phi = ((2*ell+1)*(2*ell-1))*(r2**(ell-2)*np.sqrt(r2))
    return phi 

# Works
#print(polyHarmonic(np.array([[0.29222, 0, 0.029333], [0, 0.29222, 0.377140], [0.377140, 0.029333, 0]]), 0, 1))