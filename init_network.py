# Initialize the network weights and biases.
#
# Greg C. Zweigle
#
import numpy as np

def init_network(layers):

    # This is a simple initialization algorithm:
    # A consists of random numbers in range [-0.5,0.5]
    # b consists of random numbers in range [-0.5,0.5]

    # The first axis in A and b corresponds to the network layer.
    # Layer 0 connects to input data.
    # Layer len(layers) - 1 connects to output data.

    # The remain axis in A and b correspond to the network weights and biases
    # of each layer. Using the max(layers) in order to size the vector to
    # handle the largest possible width.
    
    A = np.random.random((len(layers)-1,max(layers),max(layers))) - 0.5
    b = np.random.random((len(layers)-1,max(layers))) - 0.5

    return A, b