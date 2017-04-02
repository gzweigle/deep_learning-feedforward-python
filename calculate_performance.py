# Calculate the performance of the algorithm by running the network
# with fixed values, comparing the outputs to expected outputs, and
# calculating the error rate.
#
# by Greg C. Zweigle
#
import numpy as np
import feedforward_network as fn

def calculate_performance(test_input, test_target, layers, A, b):

    error_count = 0
    
    output_length = test_target.shape[1]
    
    number_of_outputs = test_target.shape[0]

    # The output data is always a single 1 output with the other outputs 0.
    exact_output = np.identity(output_length);

    squared_error = np.empty((output_length,))

    # Use this to convert one-hot outputs to integer outputs.
    listofints = np.arange(output_length)

    for i in range(number_of_outputs):

        # For each of the test inputs, compute an
        # output with the present values of weights and biases.
        z = fn.feedforward_network(test_input[i,:], layers, A, b)

        # z[-1] holds the output from the feedforward network.
        for j in range(output_length):
            squared_error[j] = np.sum(np.square(exact_output[j,:] - z[-1]))
        
        # The trial output that is minimum distance from the expected output
        # is selected as the decoded output. Then, convert this to its integer
        # representation. For distance, using sum of squares.
        decoded_number = np.argmin(squared_error)

        # For example, if decoded_number is 2, then check that the expected
        # output is equal to [0, 0, 1, 0, 0, ...., 0].
        # If not then increment the error count.
        if decoded_number != listofints[test_target[i,:] == 1]:
            error_count = error_count + 1

    # Return the number of errors divided by the total
    # number of opportunities to make an error.
    return error_count / number_of_outputs