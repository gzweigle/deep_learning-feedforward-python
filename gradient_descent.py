# Gradient Descent stage
#
# To Do: Need to add averaging over the minibatch
#        Refactoring:
#           Surely there is a way to make this code more efficient in Python
#           Don't hardcode eta
#           Don't pass so many values.
#

# External modules.
import numpy as np


def gradient_descent(inval, zval, dval, A2, A3, A4, b2, b3, b4):

    eta = .05

    for n in range(len(A2)):
        for m in range(len(np.transpose(A2))):
            A2[n,m] = A2[n,m] - eta * dval[0][n]*inval[m]

    for n in range(len(A2)):
        b2[n] = b2[n] - eta * dval[0][n]
    
    for n in range(len(A3)):
        for m in range(len(np.transpose(A3))):
            A3[n,m] = A3[n,m] - eta * dval[1][n]*zval[0][m]

    for n in range(len(A3)):
        b3[n] = b3[n] - eta * dval[1][n]
            
    for n in range(len(A4)):
        for m in range(len(np.transpose(A4))):
            A4[n,m] = A4[n,m] - eta * dval[2][n]*zval[1][m]

    for n in range(len(A4)):
        b4[n] = b4[n] - eta * dval[2][n]
        
    return A2, A3, A4, b2, b3, b4