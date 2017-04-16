""" Standard backpropagation algorithm. """
#
# by Greg C. Zweigle
#
import numpy as np
import nonlinearities as nl

def backpropagate(output_vs_correct, a_matrix, linear_output,
                  dropout, parameters):
    """ Compute the change in cost vs output for each layer and return as delta.
    """

    delta = [np.empty((k, parameters.minibatch_size))
             for k in parameters.layers[1:]]

    last_layer_index = len(parameters.layers) - 2

    # The last-stage nonlinearity, and associated cost metric is either:
    # sigmoid + cross entropy,  or
    # softmax + log likelihood.
    # In both cases, mathematically the first delta term is the same.
    delta[last_layer_index] = output_vs_correct

    for k in reversed(range(last_layer_index)):
        delta[k] = (a_matrix[k+1].transpose().dot(delta[k+1]) *
                    nl.rlu_derivative(linear_output[k], dropout[k]))

    return delta
