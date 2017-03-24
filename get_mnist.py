# Read MNIST data out of local files and return as ndarrays.
#
import numpy as np

# Get a set of MNIST data. The data must already be downloaded,
# decompressed, and stored in a local directory called mnist_data/
def get_mnist():

    # Control how much of this data set want to apply.
    # Less data speeds things up for testing, but error rate suffers.
    # train_instances - train_start_index must be <= 60,000
    # test_instances - test_start_index must be <= 10,000
    train_start_index = 0
    train_instances = 20000
    test_start_index = 0
    test_instances = 2000
   
    # Retrieve the input data.
    train_input_full = read_individual_mnist_file( \
        'mnist_data/train-images.idx3-ubyte','input')
    test_input_full = read_individual_mnist_file( \
        'mnist_data/t10k-images.idx3-ubyte','input')
    input_data_width = train_input_full.shape[1]

    # Retrieve the output data and convert it from digits (0, ..., 9) into
    # a one-hot coded array (0 = 1000000000,..., 9 = 000000001).
    train_output_digits = read_individual_mnist_file( \
        'mnist_data/train-labels.idx1-ubyte','output')
    test_output_digits  = read_individual_mnist_file( \
        'mnist_data/t10k-labels.idx1-ubyte','output')

    output_data_width = 10
    train_output_full = \
        np.zeros((train_output_digits.shape[0],output_data_width))
    for row in range(train_output_digits.shape[0]):
        train_output_full[row,train_output_digits[row]] = 1

    test_output_full = \
        np.zeros((test_output_digits.shape[0],output_data_width))
    for row in range(test_output_digits.shape[0]):
        test_output_full[row,test_output_digits[row]] = 1

    # Now pare down the data into smaller subsets.
    train_input = \
        train_input_full[train_start_index:train_start_index+train_instances,:]
    test_input  = \
        test_input_full[test_start_index:test_start_index+test_instances,:]
    train_output = \
        train_output_full[train_start_index:train_start_index+train_instances,:]
    test_output  = \
        test_output_full[test_start_index:test_start_index+test_instances,:]

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
    data_as_ndarray  = \
        data_as_ndarray.reshape(num_instances,num_rows*num_columns)

    return data_as_ndarray