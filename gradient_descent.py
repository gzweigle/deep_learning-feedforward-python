# Gradient Descent stage
#
# To Do: Don't hardcode scale scaling
#        Don't pass so many values.
#
import numpy as np

# Update the network matrices based on the backpropagated error values.
# Use the gradient descent search method.
# This code needs to be fast.
def gradient_descent(ina, z2a, z3a, d2a, d3a, d4a, A2, A3, A4, b2, b3, b4,
    minibatch_size, use_mnist):

    # For MNIST data, convergence requires a smaller step size
    # than for the fake test data.
    if use_mnist == True:
        scale = 0.01 / minibatch_size
    else:
        scale = 0.1 / minibatch_size

    gradient_matrix(A2, d2a, ina, scale)
    gradient_matrix(A3, d3a, z2a, scale)
    gradient_matrix(A4, d4a, z3a, scale)

    b2[:] = b2 - scale * d2a.sum(axis=0)
    b3[:] = b3 - scale * d3a.sum(axis=0)
    b4[:] = b4 - scale * d4a.sum(axis=0)

# Average over the batch and then update the network matrices.
# Separate as a function for readability.
# As an example, let d = d00 d01 d02   and z = z00 z01 z02 z03 z04
#                        d10 d11 d12           z10 z11 z12 z13 z14
#                        d20 d21 d22           z20 z21 z22 z23 z24
# Then:
# A00 = d00 * z00 + d10 * z10 + d20 * z20
# A01 = d00 * z01 + d10 * z11 + d20 * z21
# A02 = d00 * z02 + d10 * z12 + d20 * z22
# A03 = d00 * z03 + d10 * z13 + d20 * z23
# A04 = d00 * z04 + d10 * z14 + d20 * z24
# A10 = d01 * z00 + d11 * z10 + d21 * z20
# ...
# A24 = d02 * z04 + d12 * z14 + d22 * z24
def gradient_matrix(A, d, z, scale):

    for j in range(d.shape[0]):
        A[:] = A - scale * np.outer(d[j,:],z[j,:])