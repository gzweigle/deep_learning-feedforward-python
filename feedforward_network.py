# Feedforward network
#
# To Do: Remove hardcoding of the depth.
#        Passing 7 parameters isn't clean.
#

# External modules.
import numpy as np

# Modules specific to this program.
import nonlinearities as nl


def feedforward_network(inval, A2, b2, A3, b3, A4, b4):

    z2 = np.dot(A2,inval)
    z2 = np.add(z2, b2)
    z2 = nl.nonlinearity(z2)
    
    z3 = np.dot(A3,z2)
    z3 = np.add(z3, b3)
    z3 = nl.nonlinearity(z3)
    
    z4 = np.dot(A4,z3)
    z4 = np.add(z4, b4)
    z4 = nl.sigmoid(z4)
    
    return z2, z3, z4