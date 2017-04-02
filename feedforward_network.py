# One pass through the network, computing the following at each stage:
# z = nonlinearity(Ax+b)
#
# by Greg C. Zweigle
#
import numpy as np
import nonlinearities as nl

def feedforward_network(inval, layers, A, b):

    z = [np.empty((x,)) for x in layers]

    # Treat the input to the network as the output from a fictitious
    # identity input stage. So, set z[0] equal to the input.
    # The benefit of doing this is avoiding propagation of input values
    # separately to the gradient descent stage.  The downside of doing
    # this is that the z array indices are one larger than the backpropagation
    # (d) array indices.
    z[0] = inval

    # Hidden layers use the RLU nonlinearity.
    for k in range(0, len(A)-1):
        z[k+1] = nl.rlu(A[k].dot(z[k]) + b[k])

    # Output stage uses the sigmoid as it keeps outputs in the range [0,1]
    z[len(A)] = nl.sigmoid(A[len(A)-1].dot(z[len(A)-1]) + b[len(A)-1])

    return z