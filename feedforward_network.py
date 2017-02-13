# Feedforward network
#
# To Do: Remove hardcoding of the depth.
#        Refactor nonlin_type
#        Passing 7 parameters isn't clean.
#

# External modules.
import numpy as np

# Modules specific to this program.
import nonlinearities as nl


def feedforward_network(inval, A2, b2, A3, b3, A4, b4):

    z2 = feedforward_onestage(A2, inval, b2, 0)
    z3 = feedforward_onestage(A3, z2, b3, 0)
    z4 = feedforward_onestage(A4, z3, b4, 1)
    return z2, z3, z4


# zout = nonlinearity(A*z + b)
def feedforward_onestage(A, z, b, nonlin_type):
    zout = np.dot(A,z)
    zout = np.add(zout, b)
    if nonlin_type == 0:
        zout = nl.rlu(zout)
    else:
        zout = nl.sigmoid(zout)
    return zout