# Calculate the performance of the algorithm
#

# External modules.
import numpy as np

# Modules specific to this program.
import feedforward_network as fn


def calculate_performance(test_input, test_target, A2, b2, A3, b3, A4, b4):

    # Initialize the error.
    error_count = 0

    # Loop through all of the test inputs.
    for i in range(test_target.shape[0]):
        zout = fn.feedforward_network(test_input[i,:], A2, b2, A3, b3, A4, b4)

        # For each test input, loop through the possible output values.
        # Set the decision region at 1/2 of the output range.
        for j in range(test_target.shape[1]):
            if test_target[i][j] >= 0.5 and zout[2][j] < 0.5:
                error_count = error_count + 1
            elif test_target[i][j] < 0.5 and zout[2][j] >= 0.5:
                error_count = error_count + 1

    # Return the number of errors divided by the total
    # number of opportunities to make an error.
    return error_count / test_target.shape[0] / test_target.shape[1]