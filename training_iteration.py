"""Update the network weights and biases by training over all input data
for this iteration.
"""
#
# by Greg C. Zweigle
#
import numpy as np
import backpropagate as bp
import feedforward_network as fn
import gradient_descent as gd

def training_iteration(
        train_input, train_target, a_matrix, b_vector, parameters):
    """
    Run one training iteration and update the weights and biases.

    The a_matrix and b_vector parameters are updated each call.
    """

    # Each random integer corresponds to the row of training data.
    random_order_integers = np.random.permutation(train_input.shape[1])

    b_vector_as_matrix = [np.empty((k, parameters.minibatch_size))
                          for k in parameters.layers[1:]]

    for input_ind in range(0, train_input.shape[1], parameters.minibatch_size):

        train_input_batch = (
            train_input[:, random_order_integers[
                input_ind: input_ind + parameters.minibatch_size]])
        train_target_batch = (
            train_target[:, random_order_integers[
                input_ind: input_ind + parameters.minibatch_size]])

        # Set outputs of randomly selected neurons to zero.
        dropout = [np.ones((k, parameters.minibatch_size))
                   for k in parameters.layers[1:-1]]
        for k in range(len(dropout)):
            dropout[k][np.random.random(
                (dropout[k].shape[0], parameters.minibatch_size)) < \
                parameters.dropout_probability] = 0

        # Replicate b vector into a matrix so can use matrix math (AX+B) instead
        # of matrix times vector math (Ax+b). This should speed things up.
        for k in range(len(b_vector)):
            b_vector_as_matrix[k] = np.tile(
                b_vector[k], (1, parameters.minibatch_size))

        linear_output, nonlinear_output = fn.feedforward_network(
            train_input_batch, a_matrix, b_vector_as_matrix,
            dropout, parameters)

        delta = bp.backpropagate(
            nonlinear_output[-1] - train_target_batch, a_matrix,
            linear_output, dropout, parameters)

        gd.gradient_descent(
            nonlinear_output[:-1], delta, a_matrix, b_vector, parameters)
