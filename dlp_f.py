# Deep Learning network in Python, Feedforward Implementation
#
# Most of my algorithm development is with Matlab.
# Writing this simple network to learn and get better at Python.
#
# To Do: Unit test framework.
#        Add regularization.
#        Remove hardcoding of hyperparameters (especially depth).
#        Test and debug (e.g. occasional overflow in exp(),
#            batch size ratio, seed.pkl empty, header...).
#        Lots of refactoring.
#
# Usage: Simply run this file.  It will...
#         - generate fake training and test data (input data and target data)
#         - initialize and train the network
#         - while training, for each iteration it calculates the error
#               (output vs. targets)
#         - when finished it plots the error as a function of the iteration
#         - many things are randomly generated, so each time this is run its
#               a new test case
import numpy as np
import matplotlib.pylab as pl
import pickle
import calculate_performance as cp
import feedforward_network as fn
import get_data as gd
import get_mnist as gm
import init_network as ik
import training_iteration as ti

# Test parameters
set_seed = True  # When false, use the seed generated when was True.
use_mnist = True  # When false, use fake data.

# Get a new random number state and save it as the seed for future simulations.
if set_seed == True:
    rstate = np.random.get_state()
    fd = open('seed.pkl','wb')
    pickle.dump(rstate,fd)
    fd.close()
else:
    fd = open('seed.pkl','rb')
    rstate = pickle.load(fd)
    np.random.set_state(rstate)

# Enable intermdiate updates of the plot to see
# intermediate values when running MNIST data.
if use_mnist == True:
    pl.ion()

# Initialize the number of iterations to run
# and the size of the minibatch for each iteration.
iterations = 64
minibatch_size = 20

if use_mnist == True:
    # Read the MNIST data.
    train_input, train_target, test_input, test_target, inwidth, outwidth = \
    gm.get_mnist()
else:
    # Get the input data and the expected output values for fake data.
    train_input, train_target, test_input, test_target, inwidth, outwidth = \
    gd.get_data()

# Initialize the network matrices.
A2, b2, A3, b3, A4, b4 = ik.init_network(inwidth, outwidth, use_mnist)

# Loop to train against all the training data then test against all
# of the testing data.  Repeat this "iterations' times.
error_count = np.empty((iterations,1))
for i in range(iterations):

    # One training iteration over the data in the minibatch.
    ti.training_iteration(train_input, train_target, \
    A2, b2, A3, b3, A4, b4, minibatch_size, use_mnist)

    # Calculate the error for this iteration.
    error_count[i] = cp.calculate_performance(test_input, test_target, \
    A2, b2, A3, b3, A4, b4)

    # MNIST data takes longer to train so its nice to see intermediate updates.
    if use_mnist == True:
        print(100*error_count[i])

# Plot the performance results
pl.plot(100*error_count)
pl.xlabel('iteration')
pl.ylabel('percent error')
pl.title('final error = ' + str(100*error_count[iterations-1]) + ' percent')
pl.show()
