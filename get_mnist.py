# Get a set of MNIST data. The data must already be downloaded,
# decompressed, and stored in a local directory called mnist_data
#
# by Greg C. Zweigle
#
import numpy as np

def get_mnist():
 
    # Retrieve the input data.
    train_input = read_individual_mnist_file(
        'mnist_data/train-images.idx3-ubyte','input')
    test_input = read_individual_mnist_file(
        'mnist_data/t10k-images.idx3-ubyte','input')
    input_data_width = train_input.shape[1]

    # Retrieve the output data and convert it from digits (0, ..., 9) into
    # a one-hot coded array (0 = 1000000000,..., 9 = 000000001).

    train_output_digits = read_individual_mnist_file(
        'mnist_data/train-labels.idx1-ubyte','output')
    test_output_digits  = read_individual_mnist_file(
        'mnist_data/t10k-labels.idx1-ubyte','output')

    output_data_width = 10  # Hardcoded, because know digits are in range 0..9.
    train_output = np.zeros(
        (train_output_digits.shape[0],output_data_width))
    for row in range(train_output_digits.shape[0]):
        train_output[row,train_output_digits[row]] = 1

    test_output = np.zeros(
        (test_output_digits.shape[0],output_data_width))
    for row in range(test_output_digits.shape[0]):
        test_output[row,test_output_digits[row]] = 1

    return (train_input, train_output, test_input, test_output,
        input_data_width, output_data_width)

# Read a file, convert it from bytes to an ndarray, and return the ndarray.
# Got some help reading other examples from github.
def read_individual_mnist_file(filename, filecontents):

    with open(filename,'rb') as raw:
        data_type = np.dtype(np.uint32).newbyteorder('B')  # Byte order.
        header = np.frombuffer(raw.read(4),dtype=data_type)[0]
        num_instances = np.frombuffer(raw.read(4),dtype=data_type)[0]
        if (filecontents == 'input'):
            num_rows = np.frombuffer(raw.read(4),dtype=data_type)[0]
            num_columns = np.frombuffer(raw.read(4),dtype=data_type)[0]
        else:
            # The correct values are a stream of numbers, not images.
            num_rows = 1
            num_columns = 1
        data_as_bytes = raw.read(num_instances * num_rows * num_columns)
        data_as_ndarray  = np.frombuffer(data_as_bytes,dtype=np.uint8)

    # Normalize so image data is in range [0,1]
    if (filecontents == 'input'):
        data_as_ndarray = data_as_ndarray / 256

    # There are num_instances input images, each treated as an array.
    data_as_ndarray  = data_as_ndarray.reshape(
        num_instances , num_rows * num_columns)

    return data_as_ndarray