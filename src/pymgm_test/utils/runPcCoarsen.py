
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
print("x is: ")
print(x)
print (" ")
x1 = np.array([-0.77777778, -0.33333333, -0.77777778, -0.77777778, -0.11111111, -0.77777778, 0.11111111, 0.77777778, 0.77777778, -0.33333333, 0.33333333, -0.33333333, 0.11111111, 0.77777778, 0.33333333, 0.77777778])
y1 = np.array([-0.77777778, -0.77777778, -0.33333333, 0.77777778, -0.33333333, 0.33333333, -0.77777778, -0.33333333, -0.77777778, 0.11111111, -0.11111111, 0.77777778, 0.33333333, 0.33333333, 0.77777778, 0.77777778])
x3 = np.array([0.777777777777778,-0.777777777777778,-0.777777777777778,0.777777777777778])
y3 = np.array([-0.777777777777778,0.777777777777778,-0.777777777777778,0.777777777777778])
# expected coarsening in matlab with same input points
# coarsening factor 4 on 64 => 16 points
#expected level 2 after one coarsen
base_case_lvl_2 = np.column_stack((x1, y1))
#expected level 3 after one coarsen
base_case_lvl_3 = np.column_stack((x3, y3))

# Create an instance of the PcCoarsen2D class
PcCoarsen_2d = PcCoarsen.PcCoarsen2D()

# xc (doubles) is the next coarsened level
# NC is double
# vol is double
NC_int = int(NC) # coarsen from 64 to 16
vol_float = float(vol)
xc = PcCoarsen_2d.Coarsen(x,NC_int,vol_float)
print("xc is: ")
print(xc)
#xc_array = np.array(xc)
# Transpose the array to swap rows and columns
xy_column_vectors = np.reshape(xc, (-1, 2))

xcx = xy_column_vectors[:, 0]  # First column (x)
xcy = xy_column_vectors[:, 1]  # Second column (y)
print("xcx is: ")
print(xcx)
print (" ")

xc_reshaped = np.column_stack((xcx, xcy))

print("xc_reshaped is: ")
print(xc_reshaped)
print (" ")


# Sort both arrays


xc_sorted2_indices = np.lexsort((xc_reshaped[:, 1], xc_reshaped[:, 0]))
xc_sorted2 = xc_reshaped[xc_sorted2_indices]
print("xc sorted2  lvl2")
print(xc_sorted2)
#
# xc_expect_sorted = base_case_lvl_2[base_case_lvl_2[:, 0].argsort()]
# print("xc_expect_sorted (lvl2)")
# print(xc_expect_sorted)
# # Compare if sorted arrays are equal
# are_equal = np.allclose(xc_sorted, xc_expect_sorted)
# print("Are arrays equal (lvl2):", are_equal)
#
# # Coarsen again (lvl 3)
# # expected level 3 after coarsen
# NC3 = len(base_case_lvl_2) / 4
# NC3_int = int(NC3) # coarsen from 16 to 4
# xc3 = PcCoarsen_2d.Coarsen(xc,NC3_int,vol_float)
# xc3x = xc3[:, 0]  # First column (x)
# xc3y = xc3[:, 1]  # Second column (y)
# xc3_reshaped = [xc3x, xc3y];
# #xc3_reshaped = np.reshape(xc3, (-1, 2))
#
#
# # Sort both arrays (lvl 3)
# xc3_sorted = xc3_reshaped[xc3_reshaped[:, 0].argsort()]
# xc3_expect_sorted = base_case_lvl_3[base_case_lvl_3[:, 0].argsort()]
# print("xc sorted (lvl3)")
# print(xc3_sorted)
#
# print("xc3_expect_sorted (lvl3)")
# print(xc3_expect_sorted)
# # Compare if sorted arrays are equal
# are_equal3 = np.allclose(xc3_sorted, xc3_expect_sorted)
# print("Are arrays equal (lvl3):", are_equal3)
#
# # quick tests in runPcCorasen for now Tracking down anomalies
# # Test more discrete or larger cases
# # keyholepoissond, diskpoisson with variuus sizes
# # Then Build all this out to /test/test_PC_coarsen.py