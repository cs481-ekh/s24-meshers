import numpy as np



def afun(self,uh, mgmobj):
    lh=mgmobj[0].Lh
    res = np.dot(lh, uh)
    return res # Assuming mgmobj is an array-like object with Lh attribute