# Various nonlinearities
#
# To Do: Replace for loops using Python better
#

# External modules.
import numpy as np

# Sometimes a small number can help with performance
RLU_NEG_SLOPE = 0.000


def rlu(nonlin_in):

    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        if nonlin_in[i] < 0:
            outval[i] = RLU_NEG_SLOPE * nonlin_in[i]
        else:
            outval[i] = nonlin_in[i]

    return outval
    

def rlu_derivative(nonlin_in):
    
    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        if nonlin_in[i] < 0:
            outval[i] = RLU_NEG_SLOPE
        else:
            outval[i] = 1
            
    return outval


def sigmoid(nonlin_in):
    
    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        outval[i] = 1 / (1 + np.exp(-nonlin_in[i]))

    return outval
    

def sigmoid_derivative(nonlin_in):
    
    outval = np.empty(nonlin_in.shape)
    
    for i in range(len(nonlin_in)):
        outval[i] = nonlin_in[i] * (1 - nonlin_in[i])

    return outval