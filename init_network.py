# Initialize the network weights and biases.
#
# Greg C. Zweigle
#
import numpy as np

def init_network(layers):

    # The list axis in A and b corresponds to the network layer.
    A = [np.random.random((x, y))-0.5 for x, y in zip(layers[1:], layers[:-1])]
    b = [np.random.random((x,))-0.5 for x in layers[1:]]

    return A, b