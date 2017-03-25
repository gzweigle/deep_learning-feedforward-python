# Update the network weights and biases by training over all input data
# for this iteration.
#
# by Greg C. Zweigle
#
import numpy as np
import backpropagate as bp
import feedforward_network as fn
import gradient_descent as gd

def training_iteration(input_values, target_values, layers, \
                       A, b, minibatch_size, step_size):

    # Initialize arrays to hold intermediate values for averaging.
    in_minibatch = np.empty((minibatch_size, layers[0]))
    # Axis:
    # len(layers) - one entry per layer in the deep network.
    # minibatch_size - one entry per batch, for averaging by gradient descent
    # max(layers) - need sufficient space to hold largest z output
    z_minibatch = np.empty((len(layers), minibatch_size, max(layers)))
    d_minibatch = np.empty((len(layers), minibatch_size, max(layers)))

    # Create an array of randomly ordered integers in range
    # [0,input_values.shape[0]-1].
    # Each integer corresponds to the row of training data.
    random_order_integers = np.random.permutation(input_values.shape[0])

    # Skip through the randomly ordered integers to 
    # create subsets for the minibatch.
    for i in range(0, input_values.shape[0], minibatch_size):

        # The index j represents incrementing through the minibatch inputs.
        for j in range(minibatch_size):

            # The index into the input array.
            index = random_order_integers[i+j]

            # Run one iteration of the network
            z = fn.feedforward_network(input_values[index,:], layers, A, b)

            # Run a back propagation
            d = bp.backpropagate(layers, A, z, target_values[index,:])

            # Save for this set, so can apply as a batch to gradient descent.
            in_minibatch[j,:] = input_values[index,:]
            for k in range(len(layers)-1):
                z_minibatch[k,j,0:layers[k+1]] = z[k,0:layers[k+1]]
                d_minibatch[k,j,0:layers[k+1]] = d[k,0:layers[k+1]]
            
        # Update the matrices
        gd.gradient_descent(in_minibatch, layers, z_minibatch, d_minibatch, \
            A, b, minibatch_size, step_size)