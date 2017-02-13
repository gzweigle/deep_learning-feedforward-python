# Initialize the feedforward network
#
# To Do: There are better ways to init the network.
#        Don't hardcode the hyperparameters.
#

# External modules.
import numpy as np


def init_network(inwidth, outwidth):

    # M* is in range [0.5,0.5]
    # b* is in range [0,1]

    # The approach to selecting intermediate dimensions is just something
    # simple for now (inwidth-1 and outwidth+1).  These and other hyperparameters
    # can be selected with more sophistication eventually.

    # Input dimension:  inwidth
    # Output dimension: inwidth - 1
    A2 = 2*np.random.random((inwidth-1,inwidth)) - 1
    b2 = np.random.random((inwidth-1,))

    # Input dimension:  inwidth  - 1
    # Output dimension: outwidth + 1
    A3 = 2*np.random.random((outwidth+1,inwidth-1)) - 1
    b3 = np.random.random((outwidth+1,))

    # Input dimension:  outwidth + 1
    # Output dimension: outwidth
    A4 = 2*np.random.random((outwidth,outwidth+1)) - 1
    b4 = np.random.random((outwidth,))

    return A2, b2, A3, b3, A4, b4