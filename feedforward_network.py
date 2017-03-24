# Feedforward network
#
# To Do: Remove hardcoding of the depth.
#        Passing 7 parameters isn't clean.
#
import numpy as np
import nonlinearities as nl

# One pass through the network, computing the following at each stage:
# nonlinearity(Ax+b)
def feedforward_network(inval, A2, b2, A3, b3, A4, b4):

    # Hidden stages use the RLU nonlinearity.
    z2 = nl.rlu(np.add(np.dot(A2,inval),b2))
    z3 = nl.rlu(np.add(np.dot(A3,z2),b3))
    
    # Output stage uses the sigmoid as it
    # keeps values in the range [0,1]
    z4 = nl.sigmoid(np.add(np.dot(A4,z3),b4))
    
    return z2, z3, z4