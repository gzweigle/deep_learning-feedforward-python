# Backpropagation stage
#
# by Greg C. Zweigle
#
import numpy as np
import nonlinearities as nl

# Classic backpropagation algorithm.
def backpropagate(layers, A, z, target_values):

    d = np.empty((len(layers)-1, max(layers)))
    last_layer = len(layers) - 1
    
    # Calculate the initial error, using cross entropy.
    error_value = cross_entropy(target_values, \
        z[last_layer-1,0:layers[last_layer]])

    # Initial backpropagation, from the output towards the
    # previous network stage. Use a sigmoid nonlinearity
    # for the output stage to keep in range [0,1]
    d[last_layer-1,0:layers[last_layer]] = error_value * \
        nl.sigmoid_derivative(z[last_layer-1,0:layers[last_layer]])

    # Now backpropagate the error to the input, with RLU nonlinearities.
    for k in reversed(range(len(layers)-2)):
        d[k,0:layers[k+1]] = np.dot(A[k+1,0:layers[k+2],0:layers[k+1]].T, \
        d[k+1,0:layers[k+2]]) * nl.rlu_derivative(z[k,0:layers[k+1]])

    return d

# Measuring a distance between output values and expected output values.
def cross_entropy(target_values, output_values):

    # To avoid a divide-by-zero, keep output values away from 0 and 1.
    # The exact small value chosen (1e-8) is not critical.
    output_values[output_values == 0] = 1e-8
    output_values[output_values == 1] = 1-1e-8

    # Calculate and return the cross entropy.
    return -0.5 * (target_values/output_values -
           (1-target_values)/(1-output_values))