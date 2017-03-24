# Various nonlinearities
#
import numpy as np

def rlu(nonlin_in):

    # Could have changed nonlin_in in place but then had to construct
    # an array at the calling location. So, for performance its a don't care.
    outval = nonlin_in
    outval[outval < 0] = 0
    return outval

def rlu_derivative(nonlin_in):

    outval = np.zeros(nonlin_in.shape)
    outval[nonlin_in >= 0] = 1
    return outval

def sigmoid(nonlin_in):

    return 1 / (1+np.exp(-nonlin_in))

def sigmoid_derivative(nonlin_in):
    
    return nonlin_in * (1-nonlin_in)