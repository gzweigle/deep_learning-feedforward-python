# Various nonlinearities
#
# To Do: Replace for loops using Python better
#

# External modules.
import numpy as np

# Sometimes a small number can help with performance
rlu_neg_slope = 0.000

# Rectified Linear Unit
def nonlinearity(nonlin_in):

    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        if nonlin_in[i] < 0:
            outval[i] = rlu_neg_slope * nonlin_in[i]
        else:
            outval[i] = nonlin_in[i]

    return outval
    
# Derivative of the RLU
def nonlinearity_derivative(nonlin_in):
    
    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        if nonlin_in[i] < 0:
            outval[i] = rlu_neg_slope
        else:
            outval[i] = 1
            
    return outval

# For the output stage, use a sigmoid
def sigmoid(nonlin_in):
    
    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        outval[i] = 1 / (1 + np.exp(-nonlin_in[i]))

    return outval
    
# Derivative of the output stage.
def sigmoid_derivative(nonlin_in):
    
    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        outval[i] = nonlin_in[i] * (1 - nonlin_in[i])

    return outval
