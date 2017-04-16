"""Deep Learning network in Python, Feedforward implementation"""
#
# by Greg C. Zweigle
#
import pickle
import time
import numpy as np
import matplotlib.pylab as pl

import calculate_performance as cp
import get_quicktest_data as gd
import get_mnist as gm
import init_network as ik
import training_iteration as ti

def learn_then_test(parameters):
    """Initialize based on parameters then train and test over all input data"""

    if parameters.new_seed:
        rstate = np.random.get_state()
        with open('seed.pkl', 'wb') as seed_state_file:
            pickle.dump(rstate, seed_state_file)
    else:
        with open('seed.pkl', 'rb') as seed_state_file:
            rstate = pickle.load(seed_state_file)
        np.random.set_state(rstate)

    if parameters.use_mnist:
        (train_input, train_target, test_input, test_target,
         inwidth, outwidth) = gm.get_mnist()
    else:
        (train_input, train_target, test_input, test_target,
         inwidth, outwidth) = gd.get_quicktest_data()

    # Subsequent functions need to know inwidth and outwidth so concatenate
    # them with the layers tuple.  Since the input is an identity, the total
    # number of computing stages is equal to len(layers) - 1.
    # The outwidth of a layer is the number of neurons in that layer.
    parameters.layers = (inwidth,) + parameters.layers + (outwidth,)

    if parameters.use_mnist:
        # Speed things up a bit by only checking training performance on subset
        # when computing the post-training error rate on the training data.
        random_order_integers = np.random.permutation(train_input.shape[1])
        train_input2 = train_input[:, random_order_integers[0:10000]]
        train_target2 = train_target[:, random_order_integers[0:10000]]
    else:
        train_input2 = train_input
        train_target2 = train_target

    a_matrix, b_vector = ik.init_network(parameters)

    parameters.validate(train_input.shape[1])

    parameters.display_parameters()

    test_err = np.empty((parameters.iterations, 1))
    test_cost = np.empty((parameters.iterations, 1))
    train_err = np.empty((parameters.iterations, 1))
    train_cost = np.empty((parameters.iterations, 1))

    # For each iteration, train against all the training data then
    # test against all of the testing data.
    for iteration_count in range(parameters.iterations):

        start_time = time.clock()

        ti.training_iteration(
            train_input, train_target, a_matrix, b_vector, parameters)

        (train_err[iteration_count],
         train_cost[iteration_count]) = cp.calculate_performance(
             train_input2, train_target2, a_matrix, b_vector, parameters)

        (test_err[iteration_count],
         test_cost[iteration_count]) = cp.calculate_performance(
             test_input, test_target, a_matrix, b_vector, parameters)

        end_time = time.clock()

        print_string = ("Iteration = {0:3d}  " +
                        "Train_error = {1:5.2f}  Test_error = {2:5.2f}  " +
                        "Train_metric = {3:5.3f}  Test_metric = {4:5.3f}  " +
                        "Run_time = {5:6.3f} seconds")
        print(print_string.format(
            iteration_count,
            float(100*train_err[iteration_count]),
            float(100*test_err[iteration_count]),
            float(train_cost[iteration_count]),
            float(test_cost[iteration_count]),
            end_time - start_time))

    return train_err, train_cost, test_err, test_cost

def plot_things(train_error, train_metric, test_error, test_metric, last_time):
    """Plot the results. Set last_time to True to persist plot when done."""

    if last_time:
        pl.ioff()
    else:
        # Allow plots to update but don't block further code execution.
        pl.ion()

    pl.subplot(2, 2, 1)
    pl.plot(100*train_error)
    pl.ylabel('percent error')
    pl.title('Training: final error = ' + str(100*train_error[-1]) + ' percent')

    pl.subplot(2, 2, 3)
    pl.plot(train_metric)
    pl.ylabel('cost metric')
    pl.title('Training: final metric = ' + str(train_metric[-1]))

    pl.subplot(2, 2, 2)
    pl.plot(100*test_error)
    pl.ylabel('percent error')
    pl.title('Testing: final error = ' + str(100*test_error[-1]) + ' percent')

    pl.subplot(2, 2, 4)
    pl.plot(test_metric)
    pl.ylabel('cost metric')
    pl.title('Testing: final metric = ' + str(test_metric[-1]))
    pl.gcf().set_size_inches(8, 8)

    pl.show()
    pl.pause(0.01)
