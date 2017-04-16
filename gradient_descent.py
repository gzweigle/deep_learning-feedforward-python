"""Standard gradient descent search method, with batch averaging."""
#
# by Greg C. Zweigle
#
import numpy as np

def gradient_descent(
        output_minibatch, delta_minibatch, a_matrix, b_vector, parameters):
    """
    Update the network matrices based on the backpropagated delta values.

    The a_matrix and b_vector parameters are updated each call.
    """

    # Include the minibatch size in the scale as its part of the average.
    step_size_and_avg_size = parameters.step_size / delta_minibatch[0].shape[1]

    for k in range(len(a_matrix)):

        # Separate the a_matrix update into a function because it helps
        # me with readability and to understand the Python code better.
        gradient_matrix(
            a_matrix[k], delta_minibatch[k],
            output_minibatch[k], step_size_and_avg_size)

        b_vector[k][:, 0] = (b_vector[k][:, 0] - step_size_and_avg_size *
                             delta_minibatch[k].sum(axis=1))

def gradient_matrix(a_matrix, delta_batch, output_batch, step_size):
    """ Update the a_matrix matrix using gradient descent.

    As an example, let d = d00 d01 d02   and o = o00 o01 o02
                           d10 d11 d12           o10 o11 o12
                           d20 d21 d22           o20 o21 o22
                                                 o30 o31 o32

    Each column of d and o are for a new minibatch while the rows
    are the data for network layer k.

    Then:
    A00 -= step_size * (d00 * o00 + d01 * o01 + d02 * o02)
    A01 -= step_size * (d00 * o10 + d01 * o11 + d02 * o12)
    A02 -= step_size * (d00 * o20 + d01 * o21 + d02 * o22)
    A03 -= step_size * (d00 * o30 + d01 * o31 + d02 * o32)
    A10 -= step_size * (d10 * o00 + d11 * o01 + d12 * o02)
    A11 -= step_size * (d10 * o10 + d11 * o11 + d12 * o12)
    ...
    A23 -= step_size * (d20 * o30 + d21 * o31 + d22 * o32)

    It all simplfies to be an outer product.
    """

    # The sum is over all minibatches.
    for k in range(delta_batch.shape[1]):
        a_matrix[:] = (a_matrix - step_size *
                       np.outer(delta_batch[:, k], output_batch[:, k]))
