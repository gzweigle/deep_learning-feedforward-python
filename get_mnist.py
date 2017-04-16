"""
Get a set of MNIST handwriting data. The data must already be downloaded,
decompressed, and stored in a local directory called mnist_data/
"""
#
# by Greg C. Zweigle
#
import numpy as np

def get_mnist():
    """ Read and return the MNIST handwriting data."""

    train_input = read_individual_mnist_file(
        'mnist_data/train-images.idx3-ubyte', 'input')
    test_input = read_individual_mnist_file(
        'mnist_data/t10k-images.idx3-ubyte', 'input')
    input_data_width = train_input.shape[0]

    # Retrieve the output data and convert it from digits (0, ..., 9) into
    # a one-hot coded array (0 = 1000000000,..., 9 = 000000001).

    train_output_digits = read_individual_mnist_file(
        'mnist_data/train-labels.idx1-ubyte', 'output')
    test_output_digits = read_individual_mnist_file(
        'mnist_data/t10k-labels.idx1-ubyte', 'output')

    output_data_width = 10  # Hardcoded, because know digits are in range 0..9.
    train_output = np.zeros((
        output_data_width, train_output_digits.shape[1]))
    for row in range(train_output_digits.shape[1]):
        train_output[train_output_digits[:, row], row] = 1

    test_output = np.zeros((
        output_data_width, test_output_digits.shape[1]))
    for row in range(test_output_digits.shape[1]):
        test_output[test_output_digits[:, row], row] = 1

    return (train_input, train_output, test_input, test_output,
            input_data_width, output_data_width)

def read_individual_mnist_file(filename, filecontents):
    """Read a file, convert it from bytes to an ndarray, and return the ndarray.
    """

    with open(filename, 'rb') as raw:
        data_type = np.dtype(np.uint32).newbyteorder('B')  # Byte order.
        header = np.frombuffer(raw.read(4), dtype=data_type)[0]
        num_instances = np.frombuffer(raw.read(4), dtype=data_type)[0]
        if filecontents == 'input':
            num_rows = np.frombuffer(raw.read(4), dtype=data_type)[0]
            num_columns = np.frombuffer(raw.read(4), dtype=data_type)[0]
        else:
            # The correct values are a stream of numbers, not images.
            num_rows = 1
            num_columns = 1
        data_as_bytes = raw.read(num_instances * num_rows * num_columns)
        data_as_ndarray = np.frombuffer(data_as_bytes, dtype=np.uint8)

    # Normalize so image data is in range [0,1]
    if filecontents == 'input':
        data_as_ndarray = data_as_ndarray / np.max(data_as_ndarray)

    # Reshape then transpose in order to get the data propertly
    # vectorized into columns of image data.
    data_as_ndarray = data_as_ndarray.reshape(
        num_instances, num_rows * num_columns).transpose()

    return data_as_ndarray
