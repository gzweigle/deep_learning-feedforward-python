"""
This function creates matrices of input and output data for quick unit testing.

The input data starts with out_data_width columns.
Each columns is associated with a single output.
Then, generate multiple instances of each column, with noise added.

There are multiple instances because multiple inputs map to a single result.
For example, with handwriting data there would be many instances of noisy
letter Z's that all map to the correct output of detecting the letter Z.
For each resulting noisy input value, create a duplicate of the
correct output value.
"""
#
# by Greg C. Zweigle
#
import numpy as np

def get_quicktest_data():
    """ Create and return input and target data for unit testing."""

    # Number of inputs for each type of output.
    train_instances = 40
    test_instances = 40

    # There will be a total of train_instances * out_data_width training
    # vectors and test_instances * out_data_width test instances.
    in_data_width = 8
    out_data_width = 4

    # Making this larger makes the noise larger and so makes it more
    # difficult for the network to determine the underlying structure.
    noise_scale = 1/8

    # Build an array of input data, with randomly selected values of 0 or 1.
    exact_input = np.round(np.random.random((in_data_width, out_data_width)))

    # Now shift the exact data away from 0 and 1 by the noise scaling.
    # By doing this then, after adding noise, they stay in range [0,1]
    for which_output in range(out_data_width):
        exact_input[np.where(
            exact_input[:, which_output] == 1), which_output] = 1 - noise_scale
        exact_input[np.where(
            exact_input[:, which_output] == 0), which_output] = noise_scale

    # Initialize the (data + noise) arrays and the output value arrays.
    # There are out_data_width rows for reach training instance.
    # Training inputs/outputs are for training.  Test inputs/outputs are
    # for computing the error rate on data not used for training.
    train_input = np.zeros((in_data_width, out_data_width*train_instances))
    train_output = np.zeros((out_data_width, out_data_width*train_instances))
    test_input = np.zeros((in_data_width, out_data_width*train_instances))
    test_output = np.zeros((out_data_width, out_data_width*train_instances))

    # Build lots of instances of each exact input data, each with noise added.
    # For each instance, also duplicate its associated output.
    for which_output in range(out_data_width):

        build_noisy_data_instances(
            train_input, train_output, train_instances,
            exact_input[:, which_output], which_output, noise_scale)

        build_noisy_data_instances(
            test_input, test_output, test_instances,
            exact_input[:, which_output], which_output, noise_scale)

    # Each column of train_input (test_input) holds the input data for a given
    # output. There are train_instances (test_instances) columns for each
    # output. Each column of train_output (test_output) holds the output for
    # the associated input in the same column or train_input (test_input).
    return (train_input, train_output, test_input, test_output,
            in_data_width, out_data_width)

def build_noisy_data_instances(
        input_with_noise, replicated_output, num_instances, exact_input,
        which_output, noise_scale):
    """
    Construct input data with noise and return the associated output.

    The input_with_noise and replicated_output parameters are updated each call.
    """

    # The output data is always a single 1 output with the other outputs 0.
    exact_output = np.identity(replicated_output.shape[0])
    output_value = exact_output[:, which_output]

    data_width = exact_input.shape[0]

    for k in range(num_instances):
        input_with_noise[:, k + which_output * num_instances] = (
            exact_input + (2*np.random.random((data_width,))-1)*noise_scale)
        replicated_output[:, k + which_output * num_instances] = output_value
