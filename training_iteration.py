# Update the network weights and biases by training over all input data
# for this iteration.
#
# by Greg C. Zweigle
#
import numpy as np
import backpropagate as bp
import feedforward_network as fn
import gradient_descent as gd

def training_iteration(train_input, train_target, layers,
                       A, b, minibatch_size, step_size):

    z_minibatch = [np.empty((minibatch_size, k)) for k in layers]
    d_minibatch = [np.empty((minibatch_size, k)) for k in layers[1:]]

    # Each random integer corresponds to the row of training data.
    random_order_integers = np.random.permutation(train_input.shape[0])

    # Skip through the randomly ordered integers to 
    # create subsets for the minibatch.
    for i in range(0, train_input.shape[0], minibatch_size):

        # The index j represents incrementing through the minibatch inputs.
        for j in range(minibatch_size):

            index = random_order_integers[i+j]

            z = fn.feedforward_network(train_input[index,:], layers, A, b)

            d = bp.backpropagate(train_target[index,:], layers, A, z)

            # Save for this set, so can apply as a batch to gradient descent.
            # z[0] holds the input data.
            for k in range(len(layers)-1):
                z_minibatch[k][j] = z[k]
                d_minibatch[k][j] = d[k]

        # Update the matrices A and b.
        gd.gradient_descent(z_minibatch, d_minibatch, A, b, step_size)