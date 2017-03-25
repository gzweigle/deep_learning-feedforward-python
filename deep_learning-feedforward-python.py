# Deep Learning network in Python, Feedforward implementation
#
# Most of my algorithm development is with Matlab.
# Writing this simple network to learn and get better at Python.
#
# by Greg C. Zweigle
#
# Usage:
#
#   Written and tested with Python 3.6.
#
#   Configuration parameters include "Parameter:" in the comment field.
#
#   If use_minst == True, then reads MNIST data.
#   Else generates fake training and test data.
#       Expects MNIST data to be in a local directory called mnist_data/
#       Expects the MNIST data to already be decompressed.
#       Expects the MNIST data to be named as follows:
#           mnist_data/train-images.idx3-ubyte are the training images.
#           mnist_data/t10k-images.idx3-ubyte are the testing images.
#           mnist_data/train-labels.idx1-ubyte are the training targets.
#           mnist_data/t10k-labels.idx1-ubyte are the testing targets.
#       To change names or directory, edit get_mnist.py.
#
#   If new_seed == True, then uses a new random number seed and saves it.
#   Else reads the previously saved seed.
#
#   Set iterations to the number of desired training iterations.
#
#   Set minibatch_size to the number of trains per gradient descent.
#       Number of training inputs must be a multiple of minibatch_size.
#       The number of training inputs isn't known until the data is
#       read, so code will notify if this condition isn't met.
#
#   Set step_size to the scaling applied to previous values when updating
#       the weights for gradient descent.
#
#   Set layers to the width of hidden data.
#
#   TO DO:
#       While the code now works for arbitrary depth network,
#       the solution is complicated by having to keep track of the
#       variable shape vectors. Each layer is a different shape.
#       So, I need to figure out a better approach.
#
import numpy as np
import matplotlib.pylab as pl
import pickle
import calculate_performance as cp
import feedforward_network as fn
import get_data as gd
import get_mnist as gm
import init_network as ik
import training_iteration as ti

# Parameter: When new_seed is true, generates a new random number seed.
new_seed = True

# Parameter: When use_mnist is true, use MNIST, otherwise generate fake data.
use_mnist = True

# Parameter: Number of training iterations to run.
iterations = 64

# Parameter: Number of inputs to train per gradient descent operation.
minibatch_size = 20

# Parameter: Step size of gradient descent.
if use_mnist == True:
    step_size = 0.01 / minibatch_size
else:
    step_size = 0.1 / minibatch_size

# Get a new random number state and save it as the seed for future simulations.
if new_seed == True:
    rstate = np.random.get_state()
    with open('seed.pkl','wb') as fd:
        pickle.dump(rstate,fd)
else:
    with open('seed.pkl','rb') as fd:
        rstate = pickle.load(fd)
    np.random.set_state(rstate)
    
if use_mnist == True:
    # Read the MNIST data.
    train_input, train_target, test_input, test_target, inwidth, outwidth = \
    gm.get_mnist()
else:
    # Get the input data and the expected output values for fake data.
    train_input, train_target, test_input, test_target, inwidth, outwidth = \
    gd.get_data()

# Number of training inputs must be a multiple of the minibatch size.
if round(train_input.shape[0]/minibatch_size)*minibatch_size != \
train_input.shape[0]:
    print("Number of training inputs must be a multiple of the minibatch size.")
    print("The number of training inputs is set by the input data. ")
    print("So, to meet this constraint, it may be easiest to change ")
    print("the minibatch size parameter.  Present values:")
    print("Number of training inputs = {0:d}, minibatch size = {1:d}".\
        format(train_input.shape[0],minibatch_size))
    quit()

# Parameter: Width of each layer.
if use_mnist == True:
    layers = (inwidth, 28*3, 20, outwidth)
else:
    layers = (inwidth, 16, 8, 4, outwidth)

# Initialize the network matrices.
A, b = ik.init_network(layers)

# Loop to train against all the training data then test against all
# of the testing data.  Repeat this 'iterations' times.
error_count = np.empty((iterations,1))
for i in range(iterations):

    # One training iteration over the data in the minibatch.
    ti.training_iteration(train_input, train_target, \
        layers, A, b, minibatch_size, step_size)

    # Calculate the error for this iteration.
    error_count[i] = cp.calculate_performance(test_input, test_target, \
        layers, A, b)

    # MNIST data takes longer to train so its nice to see intermediate updates.
    if use_mnist == True:
        print(100*error_count[i])

# Plot the performance results
pl.plot(100*error_count)
pl.xlabel('iteration')
pl.ylabel('percent error')
pl.title('final error = ' + str(100*error_count[iterations-1]) + ' percent')
pl.show()