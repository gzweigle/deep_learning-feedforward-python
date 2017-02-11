# Build the input data and the expected values.
#
# To Do: Pull this from the MNIST handwriting database.
#        Move test input into a unit test framework.
#

# External modules.
import numpy as np


instances = 4  # Number of instance of the noisy input data for each output.
scale = 1      # Scale the noise.  Make this smaller to make the noise larger.

def get_data():

    # Buld random input data that maps to two different categories of output data.
    # The input data is either [0 0 0 1 1 1] + noise or [1 1 1 0 0 0] + noise.
    # This code is just something quick and simple to test out the algorithms while developing.
    # This code will most certainly be deleted eventually.

    inval = np.zeros((2*instances,6))
    outval = np.zeros((2*instances,2))

    for i in range(instances):
        inval[i,:]   = ([0.5/scale, 0.5/scale, 0.5/scale, 1-0.5/scale, 1-0.5/scale, 1-0.5/scale] + 
                       (2*np.random.random((1,6))-1)/scale)
        outval[i,:]  = [1, 0]

    for i in range(instances):
        inval[i+instances,:] = ([1-0.5/scale, 1-0.5/scale, 1-0.5/scale, 0.5/scale, 0.5/scale, 0.5/scale] +
                               (2*np.random.random((1,6))-1)/scale)
        outval[i+instances,:]  = [0, 1]

    return inval, outval