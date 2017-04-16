# deep_learning_feedforward
Deep Learning neural network, Feedforward implementation.

## Purpose

I chose this learning network as a simple (and fun) target application while learning Python. I'm learning and the network is learning.

## Usage

The top level is deep_learning_feedforward.py. All setup parameters are located in the Parameters class and initialized in this top level code.

Here is a brief description of the parameters:

### use_mnist

If use_mnist == False, the code generates fake training and test data. This is useful for quick testing. Else, it reads MNIST data. Default is True.

MNIST data is not included in the repository. Requirements of the MNIST data:
- Expects MNIST data is in a local directory called mnist_data/
- Expects the MNIST data is decompressed.
- Expects the MNIST data is named as follows (these are the standard names):
  * mnist_data/train-images.idx3-ubyte are the training images.
  * mnist_data/t10k-images.idx3-ubyte are the testing images.
  * mnist_data/train-labels.idx1-ubyte are the training targets.
  * mnist_data/t10k-labels.idx1-ubyte are the testing targets.
- To change the file names or location, edit get_mnist.py.

### new_seed

If new_seed == True, then a new random number seed is generated and pickled. Else, the previously saved seed is used. Random numbers are needed to generate fake training and test data (when use_mnist is false) and also to initialize the network ndarrays. A pickled seed is not included with the repository and so the first run needs to have this set as True. Default is True.

### iterations

The iterations parameter is the number of training iterations. Default is 60.

### minibatch_size

Set minibatch_size to the number of training inputs per gradient descent. The number of training inputs must be a multiple of minibatch_size. However, since number of training inputs isn't known until the data is read, the code will notify and exit if this condition isn't met. Default is 20.

### step_size

Set step_size to the scaling applied when updating the weights for gradient descent. Default is 0.1.

### dropout_probability

During training, at each minibatch iteration a random subset of nodes is selected for removal. This parameter sets the probability of removal. During testing, all nodes are kept but weighted by one minus this parameter. Default is 0.5

### output_nonlin_type

If output_nonlin_type == sigmoid then the output layer is a sigmoid function and the cost metric is cross-entropy. If output_nonlin_type == softmax then the output layer is a softmax function and the cost metric is log likelihood. The output nonlinearity and cost metric are coupled to ensure that the first backpropagation delta reduces to subtraction between the network output and target values. Default is softmax.

### layers (tuple)

Set layers to the width of each hidden layer. This is the number of neurons in that layer. There is no need to include the width of input or output layers. Those are set automatically based on the MNIST data sizes. Default is (84, 20).

## Algorithm

- Straightforward, textbook implementation of a feedforward neural network.
- RLU nonlinearity for hidden layers.
- Selectable sigmoid or softmax nonlinearity for output layer.
- Gradient descent with minibatches based on random subsets of the input data.
- Displays error performance on training and test data, along with training time, after each training iteration.
- Displays a graph of error performance at completion of all training iterations.

## Status

The code is stable and seems to work ok, but is not 100% tested.

Written and tested with Python 3.6.

## History

The initial commit(s) did not utilize Python well and also hardcoded the network depth. After reading "Learning Python, 5th Ed." by Mark Lutz, the Python code got better (I hope!). Subsequently, I tried to make the network depth configurable by adding an extra dimension to the ndarrays. However, the code became very complicated and so I switched to lists of ndarrays. This code is simpler. The recent commit changed to block matrix operations in an attempt to speed things up.
