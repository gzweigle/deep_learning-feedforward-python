# Update the network matrices by training over all input data for this iteration.
#
# To Do: Need to add regularization, probably with dropout.
#        Refactoring:
#           Passing in too many values.
#

# External modules.
import numpy as np

# Modules specific to this program.
import backpropagate as bp
import feedforward_network as fn
import gradient_descent as gd


def training_iteration(input_values, target_values, A2, b2, A3, b3, A4, b4, minibatch_size):

    # Initialize arrays to hold intermediate values for averaging.
    ina = np.empty((minibatch_size,input_values.shape[1]))
    z2a = np.empty((minibatch_size,b2.shape[0]))
    z3a = np.empty((minibatch_size,b3.shape[0]))
    d2a = np.empty((minibatch_size,b2.shape[0]))
    d3a = np.empty((minibatch_size,b3.shape[0]))
    d4a = np.empty((minibatch_size,b4.shape[0]))

    # Create an array of randomly ordered integers in range [0,input_values.shape[0]-1].
    random_order_integers = np.random.permutation(input_values.shape[0])

    # Skip through the randomly ordered integers to create subsets for the minibatch.
    for i in range(0,input_values.shape[0],minibatch_size):

        # The index j represents incrementing through the minibatch.
        for j in range(minibatch_size):

            # The index into the input array.
            index = random_order_integers[i+j]

            # Save all input for this minbatch
            ina[j][:] = input_values[index,:]

            # Run one iteration of the network
            z2, z3, z4 = fn.feedforward_network(ina[j][:], A2, b2, A3, b3, A4, b4)

            # Save all values for this minbatch
            z2a[j][:] = z2
            z3a[j][:] = z3

            # Run a back propagation
            d2, d3, d4 = bp.backpropagate(z2, z3, z4, A3, A4, target_values[index,:])

            # Save all values for this minibatch
            d2a[j][:] = d2
            d3a[j][:] = d3
            d4a[j][:] = d4

        # Update the matrices
        gd.gradient_descent(ina, z2a, z3a, d2a, d3a, d4a, A2, A3, A4, b2, b3, b4, minibatch_size)