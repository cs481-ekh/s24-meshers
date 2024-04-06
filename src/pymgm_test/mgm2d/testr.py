import numpy as np
from scipy.sparse import coo_matrix # sparse() matlab equivalent, for generating sparse matrices

row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
df = coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
print(df)