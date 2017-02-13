# Gradient Descent stage
#
# To Do: Surely there is a way to make this code more efficient in Python
#        Don't hardcode eta
#        Don't pass so many values.
#

# External modules.
import numpy as np


def gradient_descent(ina, z2a, z3a, d2a, d3a, d4a, A2, A3, A4, b2, b3, b4, minibatch_size):

    eta = .1
    decay = 1

    gradient_matrix(A2, d2a, ina, minibatch_size, eta, decay)
    gradient_matrix(A3, d3a, z2a, minibatch_size, eta, decay)
    gradient_matrix(A4, d4a, z3a, minibatch_size, eta, decay)

    gradient_vector(A2, b2, d2a, minibatch_size, eta, decay)
    gradient_vector(A3, b3, d3a, minibatch_size, eta, decay)
    gradient_vector(A4, b4, d4a, minibatch_size, eta, decay)


def gradient_matrix(A, d, z, ms, eta, decay):
            
    for n in range(len(A)):
        for m in range(len(np.transpose(A))):
            avg = 0
            for j in range(ms):
                avg = avg + d[j][n] * z[j][m]
            avg = avg / ms
            A[n,m] = decay * A[n,m] - eta * avg


def gradient_vector(A, b, d, ms, eta, decay):

    for n in range(len(A)):
        avg = 0
        for j in range(ms):
            avg = avg + d[j][n]
        avg = avg / ms
        b[n] = decay * b[n] - eta * avg