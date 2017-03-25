# Feedforward network
#
# by Greg C. Zweigle
#
import numpy as np
import nonlinearities as nl

# One pass through the network, computing the following at each stage:
# nonlinearity(Ax+b)
def feedforward_network(inval, layers, A, b):
    
    # Initialize the outputs from each layer.
    z = np.empty((len(layers)-1,max(layers)))
    
    # Initial layer, including the RLU nonlinearity.
    z[0,0:layers[1]] = \
    nl.rlu( \
    np.add( \
    np.dot( \
    A[0,0:layers[1],0:layers[0]],inval), \
    b[0,0:layers[1]]))

    # Hidden layers use the RLU nonlinearity.
    for k in range(1, len(layers)-1):
        z[k,0:layers[k+1]] = \
        nl.rlu( \
        np.add( \
        np.dot( \
        A[k,0:layers[k+1],0:layers[k]], z[k-1,0:layers[k]]), \
        b[k,0:layers[k+1]]))
    
    # Output stage uses the sigmoid as it keeps values in the range [0,1]
    last_layer = len(layers) - 1
    z[last_layer-1,0:layers[last_layer]] = \
    nl.sigmoid( \
    np.add( \
    np.dot( \
    A[last_layer-1,0:layers[last_layer],0:layers[last_layer-1]], \
    z[last_layer-2,0:layers[last_layer-1]]), \
    b[last_layer-1,0:layers[last_layer]]))
        
    return z