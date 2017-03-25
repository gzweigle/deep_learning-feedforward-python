# Calculate the performance of the algorithm
#
# by Greg C. Zweigle
#
import numpy as np
import feedforward_network as fn

# Run the network with fixed values, compare the outputs to
# expected outputs, and calculate the error rate.
def calculate_performance(test_input, test_target, layers, A, b):

    # Initialize the error counting.
    error_count = 0
    
    # Size of the data output from the neural network.
    output_length = test_target.shape[1]
    
    # Number of outputs from the neural network.
    number_of_outputs = test_target.shape[0]

    # The output data is always a single 1 output with the other outputs 0.
    exact_output = np.identity(output_length);

    # For each possible output, this is the squared error to the actual output.
    squared_error = np.empty((output_length,))

    # Convert one-hot to integers.
    listofints = np.arange(output_length)

    # Loop through all of the test inputs.
    for i in range(number_of_outputs):

        # For each of the test input, compute an
        # output with the present values of weights and biases.
        z = fn.feedforward_network(test_input[i,:], layers, A, b)

        # For each output, compute the sum of squares
        # against all possible outputs.
        for j in range(output_length):
            squared_error[j] = \
            np.sum(np.square(exact_output[j,:] - \
            z[len(layers)-2,0:layers[len(layers)-1]]))
        
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
    return error_count/number_of_outputs