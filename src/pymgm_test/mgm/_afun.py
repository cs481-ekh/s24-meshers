import numpy as np



def afun(self,uh, mgmobj):
    Lh = mgmobj[4]['levelsData'][0]['Lh']
    res = Lh.dot(uh)
    return res # Assuming mgmobj is an array-like object with Lh attribute
