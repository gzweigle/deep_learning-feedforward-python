# Standard backpropagation algorithm
#
# by Greg C. Zweigle
#
import numpy as np
import nonlinearities as nl

def backpropagate(target_values, layers, A, z):

    d = [np.empty((x,)) for x in layers]
    last_layer_index = len(layers) - 2

    # Calculate the initial error, using cross entropy.
    # Add 1 to z's index because z included the input as z[0]
    # and so all of its indices are one larger than d.
    error_value = cross_entropy(target_values, z[last_layer_index+1])

    # Initial backpropagation, from the output towards the
    # previous network stage. Use a sigmoid nonlinearity
    # for the output stage to keep output values in range [0,1]
    d[last_layer_index] = error_value * nl.sigmoid_derivative(
        z[last_layer_index+1])

    # Now backpropagate the error to the input, with RLU nonlinearities.
    for k in reversed(range(last_layer_index)):
        d[k] = A[k+1].transpose().dot(d[k+1]) * nl.rlu_derivative(z[k+1])

    return d

# Measuring a distance between output values and expected output values.
def cross_entropy(target_values, output_values):

    # To avoid any divide-by-zero possibility, keep output values away
    # from 0 and 1.  The exact small value chosen (1e-8) is not critical.
    output_values[output_values == 0] = 1e-8
    output_values[output_values == 1] = 1-1e-8

    return -0.5 * (target_values/output_values -
           (1-target_values)/(1-output_values))