"""Various nonlinearities."""
#
# by Greg C. Zweigle
#
import numpy as np

def rlu(nonlin_in, dropout):
    """ Return the RLU nonlinearity output, zeroing the dropped outputs."""
    outval = nonlin_in * dropout
    outval[nonlin_in < 0] = 0
    return outval

def rlu_derivative(nonlin_in, dropout):
    """ Return the RLU derivative output, zeroing the dropped outputs."""
    outval = dropout
    outval[nonlin_in < 0] = 0
    return outval

def sigmoid(nonlin_in):
    """ Return the sigmoid nonlinearity output."""
    return 1 / (1 + np.exp(-nonlin_in))

def softmax(nonlin_in):
    """ Return the softmax nonlinearity output."""
    nonlin_out = np.empty(nonlin_in.shape)
    # Avoid big numbers by rereferencing to a maximum value of zero.
    # This also makes the denominator >= 1, to avoid underflow.
    nonlin_in_max = nonlin_in.max(axis=0)
    for k in range(nonlin_in.shape[1]):

        nonlin_out[:, k] = (np.exp(nonlin_in[:, k] - nonlin_in_max[k]) /
                            np.sum(np.exp(nonlin_in[:, k] - nonlin_in_max[k])))

    return nonlin_out
