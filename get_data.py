# Build the input data and the expected values.
#
# To Do: This is temporary code, for initial testing.
#        Move test into a unit test framework.
#
# External modules.
import numpy as np

# Create matrices of input and output data.
# The input data starts with out_data_width rows.
# Each row is associated with a single output.
# Then, generate multiple instances of that row, each with noise added.
# There are multiple instances because multiple input maps to a single result.
# For example, with handwriting data there would be 100's of noisy
# letter Z's that all map to the correct output of detecting the letter Z.
# For each resulting noisy input value, create a duplicate of the
# correct output value.
def get_data():

    # Number of inputs for each type of output.
    train_instances = 40
    test_instances = 40

    # Hard code these for this test data.
    in_data_width = 8
    out_data_width = 3

    # Scale the noise by this factor.
    # Making this larger makes the noise larger.
    noise_scale = 1/4

    # The output data is always a single 1 output with the other outputs 0.
    exact_output = np.identity(out_data_width);

    # Build an array of exact input data,
    # with randomly selected values of 0 or 1.
    exact_input = np.round(np.random.random((out_data_width,in_data_width)))

    # Now shift the exact data away from 0 and 1 by the noise scaling.
    # By doing this then, after adding noise, they stay in range [0,1]
    for which_output in range(out_data_width):
        exact_input[which_output,np.where(exact_input[which_output,:]==1)] = \
            1-noise_scale
        exact_input[which_output,np.where(exact_input[which_output,:]==0)] = \
            noise_scale

    # Initialize the (data + noise) arrays and the output value arrays.
    # There are out_data_width rows for reach training instance.
    # Training inputs/outputs are for, ... training.
    # Test inputs/outputs are for computing the error rate
    # on data not used for training.
    train_input  = np.zeros((out_data_width*train_instances,in_data_width))
    train_output = np.zeros((out_data_width*train_instances,out_data_width))
    test_input   = np.zeros((out_data_width*train_instances,in_data_width))
    test_output  = np.zeros((out_data_width*train_instances,out_data_width))

    # Build lots of instances of each exact input data, each with noise added.
    # For each instance, also duplicate its associated output.
    for which_output in range(out_data_width):

        build_noisy_data_instances( \
            train_instances, exact_input[which_output,:], \
            train_input, train_output, exact_output[which_output,:], \
            which_output, noise_scale)

        build_noisy_data_instances( \
            test_instances, exact_input[which_output,:], \
            test_input, test_output, exact_output[which_output,:], \
            which_output, noise_scale)

    # Each row of train_input (test_input) holds the input data for a given
    # output.  There are train_instances (test_instances) rows for each output.
    # Each row of train_output (test_output) holds the output for the
    # associated input in the same row or train_input (test_input).
    return (train_input, train_output, test_input, test_output,
            in_data_width, out_data_width)

# Utility function to eliminate duplicated code.
# Construct input data with noise added.
# For each input, replicate the associated output.
def build_noisy_data_instances(num_instances, exact_input, noisy_input, \
    noisy_output, output_value, which_output, noise_scale):

    data_width = exact_input.shape[0]

    for i in range(num_instances):
        noisy_input[i + which_output * num_instances,:] =  \
            (exact_input + (2*np.random.random((1,data_width))-1)*noise_scale)
        noisy_output[i + which_output * num_instances,:] = output_value