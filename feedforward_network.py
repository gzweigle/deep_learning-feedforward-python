"""
One pass through the network, computing the following at each layer:
nonlinearity(Ax+b)
"""
#
# by Greg C. Zweigle
#
import numpy as np
import nonlinearities as nl

def feedforward_network(inval, a_matrix, b_vector, dropout, parameters):
    """ One pass through the network and return nonlinearity(Ax+b)."""

    num_inputs = inval.shape[1]
    nonlinear_output = [np.empty((k, num_inputs)) for k in parameters.layers]
    linear_output = [np.empty((k, num_inputs)) for k in parameters.layers[1:-1]]

    # Treat the input to the network as the output from a fictitious
    # identity input stage. So, set nonlinear_output[0] equal to the input.
    # The benefit of doing this is avoiding propagation of input values
    # separately to the gradient descent stage.  The downside of doing
    # this is that the nonlinear_output array indices are one larger
    # than the backpropagation array indices.
    nonlinear_output[0] = inval

    for k in range(0, len(a_matrix)-1):
        linear_output[k] = a_matrix[k].dot(nonlinear_output[k]) + b_vector[k]
        nonlinear_output[k+1] = nl.rlu(linear_output[k], dropout[k])

    # Output stage uses a nonlinearity that keeps outputs in the range [0,1]
    if parameters.output_nonlin_type == "softmax":
        nonlinear_output[len(a_matrix)] = nl.softmax(
            a_matrix[len(a_matrix)-1].dot(
                nonlinear_output[len(a_matrix)-1]) + b_vector[len(a_matrix)-1])
    else:
        nonlinear_output[len(a_matrix)] = nl.sigmoid(
            a_matrix[len(a_matrix)-1].dot(
                nonlinear_output[len(a_matrix)-1]) + b_vector[len(a_matrix)-1])

    return linear_output, nonlinear_output
