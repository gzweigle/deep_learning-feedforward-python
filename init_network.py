# Initialize the feedforward network
#
# To Do: More sophisticated algorithm
#

# External modules.
import numpy as np


def init_network(indim, outdim):

    # M* is in range [0.5,0.5]
    # b* is in range [0,1]

    # The approach to selecting intermediate dimensions is just something
    # simple for now (indim-1 and outdim+1).  These and other hyperparameters
    # can be selected with more sophistication eventually.

    # Input dimension:  indim
    # Output dimension: indim - 1
    A2 = 2*np.random.random((indim-1,indim)) - 1
    b2 = np.random.random((indim-1,))

    # Input dimension:  indim  - 1
    # Output dimension: outdim + 1
    A3 = 2*np.random.random((outdim+1,indim-1)) - 1
    b3 = np.random.random((outdim+1,))

    # Input dimension:  outdim + 1
    # Output dimension: outdim
    A4 = 2*np.random.random((outdim,outdim+1)) - 1
    b4 = np.random.random((outdim,))

    return A2, A3, A4, b2, b3, b4