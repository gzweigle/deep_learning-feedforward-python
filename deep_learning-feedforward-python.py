# Deep Learning network in Python, Feedforward implementation
#
# Most of my algorithm development is with Matlab.
# Writing this simple network to learn and get better at Python.
#
# by Greg C. Zweigle
#
# Written and tested with Python 3.6.
#
# All configuration parameters are at the begining of this .py file
# and include "Parameter:" in the comment field.
# See README.md for the details of each parameter.
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


# All parameters...

# Parameter: When use_mnist is true, use MNIST, otherwise generate fake data.
use_mnist = True

# Parameter: When new_seed is true, generates a new random number seed.
new_seed = True

# Parameter: Number of training iterations to run.
iterations = 60

# Parameter: Number of inputs to train per gradient descent operation.
minibatch_size = 20

# Parameter: Step size of gradient descent.
if use_mnist == True:
    step_size = 0.01 / minibatch_size
else:
    step_size = 0.1 / minibatch_size

# Parameter: Width of each hidden layer.
if use_mnist == True:
    layers = (28*3, 20)
else:
    layers = (4, 3)

# End of parameters.


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
    train_input, train_target, test_input, test_target, inwidth, outwidth = \
    gm.get_mnist()

else:
    train_input, train_target, test_input, test_target, inwidth, outwidth = \
    gd.get_data()

if (round(train_input.shape[0]/minibatch_size)*minibatch_size != 
train_input.shape[0]):
    print("Number of training inputs must be a multiple of the minibatch size.")
    print("The number of training inputs is determined by the input data.")
    print("So, to meet this constraint, it may be easiest to change ")
    print("the minibatch size parameter. Present values:")
    print("Number of training inputs = {0}   minibatch size = {1}".format(
        train_input.shape[0],minibatch_size))
    quit()

# Subsequent functions need to know inwidth and outwidth so concatenate
# them with the layers tuple.  Since the input is an identity, the total
# number of computing stages is equal to len(layers) - 1.
layers = (inwidth,) + layers + (outwidth,)

# Initialize the network matrices.
A, b = ik.init_network(layers)

# For each iteration, train against all the training data then test against all
# of the testing data.
error_count = np.empty((iterations,1))
for iteration_count in range(iterations):

    ti.training_iteration(train_input, train_target, 
        layers, A, b, minibatch_size, step_size)

    # Calculate the error for this iteration using the values of A and b
    # that were computed during training.
    error_count[iteration_count] = cp.calculate_performance(
        test_input, test_target, layers, A, b)

    # MNIST data takes longer to train so its nice to see intermediate updates.
    if use_mnist == True:
        print("classification error = {0:5.2f}  iteration = {1}".format(
            float(100*error_count[iteration_count]), iteration_count))

pl.plot(100*error_count)
pl.xlabel('iteration')
pl.ylabel('percent error')
pl.title('final error = ' + str(100*error_count[iterations-1]) + ' percent')
pl.show()