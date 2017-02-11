# Backpropagation stage
#
# To Do: Don't hardcode the network depth.
#        Passing 4 parameters isn't clean.
#

# External modules.
import numpy as np

# Modules specific to this program.
import nonlinearities as nl


def backpropagate(zval, A3, A4, y):
    
    # Calculate the initial error, using cross entropy.
    error = np.empty((len(y),1,))
    for i in range(len(zval[2])):
        # Catch any possible inf cases.
        if zval[2][i] == 0:
            error[i] = 1000000
        elif zval[2][i] == 1:
            error[i] = -1000000
        else:
            # The normal path.
            error[i] = -0.5 * (y[i]/zval[2][i] - (1-y[i])/(1-zval[2][i]))

    # Calculate the first error term.
    d4 = nl.sigmoid_derivative(zval[2])
    for i in range(len(d4)):
        d4[i] = error[i] * d4[i]

    # Back propagate the error from the output stage N to stage N-1.
    A4 = np.transpose(A4)
    d3 = np.dot(A4,d4)
    z3n = nl.nonlinearity_derivative(zval[1])
    for i in range(len(d3)):
        d3[i] = d3[i] * z3n[i]
    
    # Back propagate the error again, to stage N-2.
    A3 = np.transpose(A3)
    d2 = np.dot(A3,d3)
    z2n = nl.nonlinearity_derivative(zval[0])
    for i in range(len(d2)):
        d2[i] = d2[i] * z2n[i]
        
    return d2, d3, d4