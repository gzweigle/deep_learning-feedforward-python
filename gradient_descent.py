# Gradient Descent stage
#
# by Greg C. Zweigle
#
import numpy as np

# Update the network matrices based on the backpropagated error values.
# Use the gradient descent search method.
def gradient_descent(ina, layers, z_minibatch, d_minibatch, A, b, \
                     minibatch_size, step_size):
            
    # Input layer.
    gradient_matrix(A[0,0:layers[1],0:layers[0]], \
    d_minibatch[0,:,0:layers[1]], ina, step_size)
    
    # Hidden layers - update the weights.
    for k in range(1, len(layers)-1):
        gradient_matrix(A[k,0:layers[k+1],0:layers[k]], \
        d_minibatch[k,:,0:layers[k+1]], \
        z_minibatch[k-1,:,0:layers[k]], step_size)

    # Hidden layers and layers - update the weights.
    for k in range(len(layers)-1):
        b[k,0:layers[k+1]] = b[k,0:layers[k+1]] - \
        step_size * d_minibatch[k,:,0:layers[k+1]].sum(axis=0)

# The gradient_matrix function averages over the batch and then updates
# the network matrices.  Separate as a function for readability.
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
def gradient_matrix(A, d, z, step_size):

    for j in range(d.shape[0]):
        A[:] = A - step_size * np.outer(d[j,:],z[j,:])