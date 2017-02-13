# Deep Learning network in Python, Feedforward Implementation
#
# Most of my algorithm development is with Matlab.
# Writing this simple network to learn and get better at Python.
#
#
# To Do:    Unit test framework.
#           Add regularization.
#           Remove hardcoding of hyperparameters (especially depth).
#           Connect to a real dataset, most likely MNIST.
#           Test and debug (e.g. occasional overflow in exp(), batch size ratio, ...).
#           Lots of refactoring.
#
#
# Usage: Simply run this file.  It will...
#         - generate fake training and test data (input data and target data)
#         - initialize and train the network
#         - while training, for each iteration it calculates the error (output vs. targets)
#         - when finished it plots the error as a function of the iteration
#         - many things are randomly generated, so each time this is run its a new test case

# External modules.
import numpy as np
import matplotlib.pylab as pl

# Modules specific to this program.
import calculate_performance as cp
import feedforward_network as fn
import get_data as gd
import init_network as ik
import training_iteration as ti

# Initialize the number of iterations to run and the size of the minibatch for each iteration.
iterations = 64
minibatch_size = 10

# Get the input data and the expected output values.
train_input, train_target, test_input, test_target, inwidth, outwidth = gd.get_data()

# Initialize the network matrices.
A2, b2, A3, b3, A4, b4 = ik.init_network(inwidth, outwidth)

# Run the feedforward network.
error_count = np.empty((iterations,1))
for i in range(iterations):

    # One training iteration over the data in the minibatch.
    ti.training_iteration(train_input, train_target, A2, b2, A3, b3, A4, b4, minibatch_size)

    # Calculate the error for this iteration.
    error_count[i] = cp.calculate_performance(test_input, test_target, A2, b2, A3, b3, A4, b4)

# Plot some results
pl.plot(100*error_count)
pl.xlabel('iteration')
pl.ylabel('percent error')
tmp_string = 'final error = ' + str(100*error_count[iterations-1]) + ' percent'
pl.title(tmp_string)
pl.show()




###########################################
#if __name__== "__main__":
#    main()