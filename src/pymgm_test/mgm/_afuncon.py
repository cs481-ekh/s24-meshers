import numpy as np
def afuncon(self,uh, mgmStruct):
    # Matrix vector multiply with a Lagrange multiplier constraint to account for
    # a one dimensional nullspace.

    temp = uh[:-1]

    # Part 1: Matrix-vector multiplication
    part1 = np.dot(mgmStruct[0]['Lh'], temp)

    # Part 2: Vector-scalar multiplication
    part2 = mgmStruct[0]['w'] * uh[-1]

    add = part1 + part2

    # Part 3: Transpose of vector and matrix-vector multiplication
    part3 = np.dot(mgmStruct[0]['w'].T, temp)

    # Combine parts 1, 2, and 3 vertically to form the final result
    uh = np.vstack((part1 + part2, part3))

    uh = [mgmStruct[0]['Lh'].dot(temp) + mgmStruct[0]['w'] * uh[-1], mgmStruct[0]['w'].dot(temp)]
    return uh
