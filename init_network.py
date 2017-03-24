# Initialize the feedforward network
#
# To Do: There are better ways to init the network.
#        Don't hardcode the hyperparameters.
#
import numpy as np

def init_network(inwidth, outwidth, use_mnist):

    # A* is in range [-1,1]
    # b* is in range [-1,1]

    # The approach to selecting intermediate dimensions is just something
    # simple for now (inwidth-1 and outwidth+1).
    # These and other hyperparameters
    # can be selected with more sophistication eventually.

    if use_mnist == False:

        # Input dimension:  inwidth
        # Output dimension: inwidth - 1
        A2 = 2*np.random.random((inwidth-1,inwidth)) - 1
        b2 = 2*np.random.random((inwidth-1,)) - 1

        # Input dimension:  inwidth  - 1
        # Output dimension: outwidth + 1
        A3 = 2*np.random.random((outwidth+1,inwidth-1)) - 1
        b3 = 2*np.random.random((outwidth+1,)) - 1
    
        # Input dimension:  outwidth + 1
        # Output dimension: outwidth
        A4 = 2*np.random.random((outwidth,outwidth+1)) - 1
        b4 = 2*np.random.random((outwidth,)) - 1

    else:

        # Input dimension:  inwidth
        # Output dimension: 28*2
        A2 = 2*np.random.random((28*2,inwidth)) - 1
        b2 = 2*np.random.random((28*2,)) - 1

        # Input dimension:  28*2
        # Output dimension: 28
        A3 = 2*np.random.random((28,28*2)) - 1
        b3 = 2*np.random.random((28,)) - 1
    
        # Input dimension:  28
        # Output dimension: outwidth
        A4 = 2*np.random.random((outwidth,28)) - 1
        b4 = 2*np.random.random((outwidth,)) - 1

    return A2, b2, A3, b3, A4, b4