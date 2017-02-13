# Backpropagation stage
#
# To Do: Don't hardcode the network depth.
#        Passing 6 parameters isn't clean.
#

# External modules.
import numpy as np

# Modules specific to this program.
import nonlinearities as nl


def backpropagate(z2, z3, z4, A3, A4, target_values):
    
    # Calculate the initial error, using cross entropy.
    error_value = cross_entropy(target_values, z4)

    # Initial backpropagation, from the output towards the previous network stage.
    d4 = nl.sigmoid_derivative(z4)
    for i in range(len(d4)):
        d4[i] = error_value[i] * d4[i]

    # Backpropagate the error to the input.
    d3 = backpropagate_onestage(A4, d4, z3)
    d2 = backpropagate_onestage(A3, d3, z2)
        
    return d2, d3, d4


def backpropagate_onestage(A, d, z):
    
    A = np.transpose(A)
    d_out = np.dot(A,d)
    z_derivative = nl.rlu_derivative(z)
    for i in range(len(d_out)):
        d_out[i] = d_out[i] * z_derivative[i]
    return d_out


def cross_entropy(target_values, output_values):

    LARGE_ERROR = 100000

    error_value = np.empty((len(target_values),1,))
    for i in range(len(output_values)):

        # First, catch any possible infinity cases.
        if output_values[i] == 0:
            error_value[i] = LARGE_ERROR
        elif output_values[i] == 1:
            error_value[i] = -LARGE_ERROR

        else:
            # The normal path.
            error_value[i] = -0.5 * (target_values[i]/output_values[i] - 
                                    (1-target_values[i])/(1-output_values[i]))

    return error_value