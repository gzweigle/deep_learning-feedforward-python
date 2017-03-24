# Backpropagation stage
#
# To Do: Don't hardcode the network depth.
#        Passing 6 parameters isn't clean.
#
import numpy as np
import nonlinearities as nl

# Class backpropagation algorithm.
def backpropagate(z2, z3, z4, A3, A4, target_values):

    # Calculate the initial error, using cross entropy.
    error_value = cross_entropy(target_values, z4)

    # Initial backpropagation, from the output towards the
    # previous network stage. Use a different nonlinearity
    # for the output stage to keep in range [0,1]
    d4 = error_value * nl.sigmoid_derivative(z4)

    # Now backpropagate the error to the input, with RLU nonlinearities.
    d3 = np.dot(A4.T,d4) * nl.rlu_derivative(z3)
    d2 = np.dot(A3.T,d3) * nl.rlu_derivative(z2)

    return d2, d3, d4

# Measuring a distance between output values and expected output values.
def cross_entropy(target_values, output_values):

    # To avoid a divide-by-zero, keep output values away from 0 and 1.
    # The exact small value chosen (1e-8) is not critical.
    output_values[output_values == 0] = 1e-8
    output_values[output_values == 1] = 1-1e-8
    return -0.5 * (target_values/output_values -
           (1-target_values)/(1-output_values))