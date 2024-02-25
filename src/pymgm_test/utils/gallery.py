import numpy as np
from scipy.io import loadmat

class Gallery_m:
    # Constructor for processed information (Lh, x, fh, uexact)
    # Default values set to None in case loading from a mat file (load_mat)
    def __init__(self, Lh=None, x=None, fh=None, uexact=None):
        self.Lh = Lh
        self.x = x
        self.fh = fh
        self.uexact = uexact
    
    def __str__(self):
        return "(" + str(self.Lh) + "," + str(self.x) + "," + str(self.fh) + "," + str(self.uexact) + ")"

    def load_mat(self, filename):
        mat_dict = loadmat(filename)
        self.Lh = mat_dict['Lh']
        self.x = mat_dict['x']
        self.fh = np.transpose(mat_dict['fh'])
        self.uexact = np.transpose(mat_dict['uexact'])

    def _Lh__str__(self):
        return str(self.Lh)

    def _x__str__(self):
        return str(self.x)

    def _fh__str__(self):
        return str(self.fh)

    def _uexact__str__(self):
        return str(self.uexact)

