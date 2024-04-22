
import PcCoarsen
import numpy as np

# Generate base case for poisson
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

# base case is first level
base_case_lvl_1 = x
# print("x is: ")
# print(x)
# print (" ")
x1 = np.array([-0.77777778, -0.33333333, -0.77777778, -0.77777778, -0.11111111, -0.77777778, 0.11111111, 0.77777778, 0.77777778, -0.33333333, 0.33333333, -0.33333333, 0.11111111, 0.77777778, 0.33333333, 0.77777778])
y1 = np.array([-0.77777778, -0.77777778, -0.33333333, 0.77777778, -0.33333333, 0.33333333, -0.77777778, -0.33333333, -0.77777778, 0.11111111, -0.11111111, 0.77777778, 0.33333333, 0.33333333, 0.77777778, 0.77777778])

# expected coarsening in matlab with same input points
# coarsening factor 4 on 64 => 16 points
#expected level 2 after one coarsen
base_case_lvl_2 = np.column_stack((x1, y1))


# Create an instance of the PcCoarsen2D class
PcCoarsen_2d = PcCoarsen.PcCoarsen2D()

# xc (doubles) is the next coarsened level
# NC is double
# vol is double
NC_int = int(NC) # coarsen from 64 to 16
vol_float = float(vol)
xc = PcCoarsen_2d.Coarsen(x,NC_int,vol_float)
# print("xc is: ")
# print(xc)
#xc_array = np.array(xc)
# Transpose the array to swap rows and columns
xy_column_vectors = np.reshape(xc, (-1, 2))

xcx = xy_column_vectors[:, 0]  # First column (x)
xcy = xy_column_vectors[:, 1]  # Second column (y)
# print("xcx is: ")
# print(xcx)
# print (" ")

xc_reshaped = np.column_stack((xcx, xcy))

# print("xc_reshaped is: ")
# print(xc_reshaped)
# print (" ")


# Sort both arrays


xc_sorted2_indices = np.lexsort((xc_reshaped[:, 1], xc_reshaped[:, 0]))
xc_sorted = xc_reshaped[xc_sorted2_indices]
print("xc sorted (python)  lvl2")
print(xc_sorted)
#

xc_lvl_2_expect_indices = np.lexsort((base_case_lvl_2[:, 1], base_case_lvl_2[:, 0]))
xc_lvl_2_sorted = base_case_lvl_2[xc_lvl_2_expect_indices]
print("xc_expect_sorted (matlab) (lvl2)")
print(xc_lvl_2_sorted)
tolerance = 1e-3  # Set your desired tolerance here

# Check if the arrays are approximately equal with custom tolerance
x_equal = np.allclose(xc_sorted[:,0], xc_lvl_2_sorted[:,0], atol=tolerance)
y_equal = np.allclose(xc_sorted[:,1], xc_lvl_2_sorted[:,1], atol=tolerance)

print("x_equal is: ",x_equal)


#expected (matlab) level 3 after two coarsen
# x3 = np.array([0.777777777777778,-0.777777777777778,-0.777777777777778,0.777777777777778])
# y3 = np.array([-0.777777777777778,0.777777777777778,-0.777777777777778,0.777777777777778])
# base_case_lvl_3 = np.column_stack((x3, y3))