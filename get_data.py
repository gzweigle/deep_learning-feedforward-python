# Build the input data and the expected values.
#
# To Do: This is temporary code, for initial testing.
#        Move test into a unit test framework.
#        Also need input that pulls from the MNIST handwriting database.
#

# External modules.
import numpy as np


def get_data():

    # Number of inputs for each type of target output.
    train_instances = 40
    test_instances = 40

    # Hard code these for this test data.
    input_data_width = 8
    target_data_width = 3

    # Scale the noise by this factor.
    # Making this larger makes the noise larger.
    noise_scale = 1/4

    # Build an array of exact input data, with randomly selected values of 0 or 1.
    exact_input = np.round(np.random.random((target_data_width,input_data_width)))

    # Build an array of target data.
    # There is a single '1' in each row for the target outputs, all other outputs are zero.
    # So, its a unity diagonal matrix.
    exact_target = np.empty((target_data_width,target_data_width))
    for which_target in range(target_data_width):
        exact_target[which_target][which_target] = 1

    # Now shift the exact data away from 0 and 1 by 1/2 of the noise scaling.
    # By doing this then, after adding noise, it will center the mean around 0 or 1.
    for which_target in range(target_data_width):
        exact_input[which_target][np.where(exact_input[which_target][:]==1)] = 1-0.5*noise_scale
        exact_input[which_target][np.where(exact_input[which_target][:]==0)] = 0.5*noise_scale

    # Initialize the (data + noise) arrays and the target value arrays.
    # Training inputs/targets are for, ... training.
    # Test inputs/targets are for computing the error rate on data not used for training.
    train_input = np.zeros((target_data_width*train_instances,input_data_width))
    train_target = np.zeros((target_data_width*train_instances,target_data_width))
    test_input = np.zeros((target_data_width*train_instances,input_data_width))
    test_target = np.zeros((target_data_width*train_instances,target_data_width))

    # Build lots of instances of each exact input data, each with noise added.
    for which_target in range(target_data_width):

        train_input, train_target = build_noisy_data_instances(train_instances, exact_input[which_target][:], \
        train_input, train_target, exact_target[which_target][:], which_target, noise_scale)

        test_input, test_target = build_noisy_data_instances(test_instances, exact_input[which_target][:], \
        test_input, test_target, exact_target[which_target][:], which_target, noise_scale)

    return train_input, train_target, test_input, test_target, input_data_width, target_data_width


def build_noisy_data_instances(num_instances, exact_input, noisy_input, noisy_target, \
                               target_value, which_target, noise_scale):

    data_width = exact_input.shape[0]

    for i in range(num_instances):
        noisy_input[i + which_target * num_instances][:] =  \
            (exact_input + (2*np.random.random((1,data_width))-1)*noise_scale)

        noisy_target[i + which_target * num_instances][:] = target_value

    return noisy_input, noisy_target