# Backpropagation stage
#
# To Do: Don't hardcode the network depth
#

# External modules.
import numpy as np

# Modules specific to this program.
import nonlinearities as nl


def backpropagate(zval, A3, A4, y):
    
    # Calculate the initial error, using cross entropy.

    # Get the first error term
    d4 = nl.sigmoid_derivative(zval[2])
    for i in range(len(d4)):
        d4[i] = error[i] * d4[i]

    # Back propagate the error.
    A4 = np.transpose(A4)
    d3 = np.dot(A4,d4)
    z3n = nl.nonlinearity_derivative(zval[1])
    for i in range(len(d3)):
        d3[i] = d3[i] * z3n[i]
    
    # Back propagate the error.
    A3 = np.transpose(A3)
    d2 = np.dot(A3,d3)
    z2n = nl.nonlinearity_derivative(zval[0])
    for i in range(len(d2)):
        d2[i] = d2[i] * z2n[i]
        
    return d2, d3, d4
