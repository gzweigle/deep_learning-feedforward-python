"""
Calculate the performance of the algorithm by running the network
with fixed weight values, comparing the outputs to target outputs, and
calculating the error rate.
 """
#
# by Greg C. Zweigle
#
import numpy as np
import feedforward_network as fn

def calculate_performance(test_input, test_target, a_matrix, b_vector,
                          parameters):
    """Return the error rate and cost metric for network and input/output."""

    total_errors = 0
    total_metric = 0

    output_length = test_target.shape[0]

    number_of_outputs = test_target.shape[1]

    # The output data is always a single 1 output with the other outputs 0.
    exact_output = np.identity(output_length)

    # Use this to convert one-hot outputs to integer outputs.
    listofints = np.arange(output_length)

    # For each of the test inputs, compute an
    # output with the present values of weights and biases.

    b_vector_as_matrix = [np.ones((k, number_of_outputs))
                          for k in parameters.layers[1:]]

    dropout_scale = [np.ones((k, test_input.shape[1])) *
                     (1 - parameters.dropout_probability)
                     for k in parameters.layers[1:-1]]

    for k in range(len(b_vector_as_matrix)):
        b_vector_as_matrix[k] = np.tile(
            b_vector[k], (1, number_of_outputs))

    linear_output, nonlinear_output = fn.feedforward_network(
        test_input, a_matrix, b_vector_as_matrix,
        dropout_scale, parameters)

    squared_error = np.empty((output_length,))

    for out_ind in range(number_of_outputs):

        # nonlinear_output[-1] holds the output from the feedforward network.
        for one_hot_ind in range(output_length):
            squared_error[one_hot_ind] = np.sum(
                np.square(exact_output[:, one_hot_ind] -
                          nonlinear_output[-1][:, out_ind]))

        # The trial output that is minimum distance from the expected output
        # is selected as the decoded output.
        # Then, convert this to its integer representation.
        decoded_number = np.argmin(squared_error)

        # Although using cross entropy or log likelihood for the
        # backpropagation cost, here using Euclidian distance.
        total_metric = total_metric + squared_error[decoded_number]

        # For example, if decoded_number is 2, then check that the expected
        # output is equal to [0, 0, 1, 0, 0, ...., 0].
        # If not then increment the error count.
        if decoded_number != listofints[test_target[:, out_ind] == 1]:
            total_errors = total_errors + 1

    return total_errors / number_of_outputs, total_metric / number_of_outputs
