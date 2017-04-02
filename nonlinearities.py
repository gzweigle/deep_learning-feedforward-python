# Various nonlinearities
#
# For RLU and the derivative of RLU, instead of zeroing values less than
# zero, scale them by a small number.  This seems to help convergence
# properties of the gradient descent. It gives a little bit of signal
# to work with.
#
# by Greg C. Zweigle
#
import numpy as np

def rlu(nonlin_in):

    outval = nonlin_in
    outval[outval < 0] = 0.1*outval[outval < 0]
    return outval

def rlu_derivative(nonlin_in):

    outval = np.ones(nonlin_in.shape) * 0.1
    outval[nonlin_in >= 0] = 1
    return outval

def sigmoid(nonlin_in):

    return 1 / (1+np.exp(-nonlin_in))

def sigmoid_derivative(nonlin_in):
    
    return nonlin_in * (1-nonlin_in)