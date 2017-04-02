# Update the network matrices based on the backpropagated error values.
# Use the gradient descent search method.
#
# by Greg C. Zweigle
#
import numpy as np

def gradient_descent(z_minibatch, d_minibatch, A, b, step_size):

    for k in range(len(A)):
        gradient_matrix(A[k], d_minibatch[k], z_minibatch[k], step_size)
        b[k] = b[k] - step_size * d_minibatch[k].sum(axis=0)

# Separate the A matrix update into a function because it helps
# me with readability and to understand the Python code better.
#
# As an example, let d = d00 d01 d02   and z = z00 z01 z02 z03 z04
#                        d10 d11 d12           z10 z11 z12 z13 z14
#                        d20 d21 d22           z20 z21 z22 z23 z24
# Each row of d and z are for a new minibatch while the columns
# are the data for network layer k.
#
# Then:
# A00 -= step_size * (d00 * z00 + d10 * z10 + d20 * z20)
# A01 -= step_size * (d00 * z01 + d10 * z11 + d20 * z21)
# A02 -= step_size * (d00 * z02 + d10 * z12 + d20 * z22)
# A03 -= step_size * (d00 * z03 + d10 * z13 + d20 * z23)
# A04 -= step_size * (d00 * z04 + d10 * z14 + d20 * z24)
# A10 -= step_size * (d01 * z00 + d11 * z10 + d21 * z20)
# ...
# A24 -= step_size * (d02 * z04 + d12 * z14 + d22 * z24)
#
# It all simplfies to be an outer product.
def gradient_matrix(A, d, z, step_size):

    for j in range(d.shape[0]):
        A[:] = A - step_size * np.outer(d[j,:],z[j,:])