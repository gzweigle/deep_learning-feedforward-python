"""Initialize the network weights and biases."""
#
# Greg C. Zweigle
#
import numpy as np

def init_network(parameters):
    """Return randomly initialized network matrices."""

    a_matrix = [np.random.random((x, y)) - 0.5
                for x, y in zip(parameters.layers[1:], parameters.layers[:-1])]

    # (x,) required a .transpose() when tiling in training_iteration().
    # (x,1) didn't, but did require the [:,0] in gradient_descent().
    # Which is faster? They seem about the same wrt performance.
    b_vector = [np.random.random((x, 1)) - 0.5 for x in parameters.layers[1:]]

    return a_matrix, b_vector
