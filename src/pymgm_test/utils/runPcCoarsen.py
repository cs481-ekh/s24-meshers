
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
xc_reshaped = np.reshape(xc, (-1, 2))


# Sort both arrays
xc_sorted = np.sort(xc_reshaped, axis=1)
xc_expect_sorted = np.sort(base_case_lvl_2, axis=1)
print("xc sorted (lvl2)")
print(xc_sorted)

print("xc_expect_sorted (lvl2)")
print(xc_expect_sorted)
# Compare if sorted arrays are equal
are_equal = np.allclose(xc_sorted, xc_expect_sorted)
print("Are arrays equal (lvl2):", are_equal)

# Coarsen again (lvl 3)
# expected level 3 after coarsen
NC3 = len(base_case_lvl_2) / 4
NC3_int = int(NC3) # coarsen from 16 to 4
xc3 = PcCoarsen_2d.Coarsen(xc,NC3_int,vol_float)
xc3_reshaped = np.reshape(xc3, (-1, 2))


# Sort both arrays (lvl 3)
xc3_sorted = np.sort(xc3_reshaped, axis=1)
xc3_expect_sorted = np.sort(base_case_lvl_3, axis=1)
print("xc sorted (lvl3)")
print(xc3_sorted)

print("xc3_expect_sorted (lvl3)")
print(xc3_expect_sorted)
# Compare if sorted arrays are equal
are_equal3 = np.allclose(xc3_sorted, xc3_expect_sorted)
print("Are arrays equal (lvl3):", are_equal3)

# quick tests in runPcCorasen for now Tracking down anomalies
# Test more discrete or larger cases
# keyholepoissond, diskpoisson with variuus sizes
# Then Build all this out to /test/test_PC_coarsen.py