# deep_learning-feedforward-python
Deep Learning network in Python, Feedforward implementation.

## Background

I chose this learning network as a simple (and fun) target application while learning Python. The repository name includes the language name because the intent of it is to get better at Python.

## Usage

The code was written and tested with Python 3.6.

All parameters are located in the first few lines of deep_learning-feedforward-python.py. Here is a description of the simulation parameters:

### use_mnits

If use_mnist == False, the code generates fake training and test data. This is useful for quick unit testing. Else, it reads MNIST data. Default is True.

MNIST data is not included in the repository. Requirements of the MNIST data:
- Expects MNIST data is in a local directory called mnist_data/
- Expects the MNIST data is decompressed.
- Expects the MNIST data is named as follows (these are the standard names):
  * mnist_data/train-images.idx3-ubyte are the training images.
  * mnist_data/t10k-images.idx3-ubyte are the testing images.
  * mnist_data/train-labels.idx1-ubyte are the training targets.
  * mnist_data/t10k-labels.idx1-ubyte are the testing targets.
- To change names or directory, edit get_mnist.py.

### new_seed

If new_seed == True, then a new random number seed is generated and pickled. Else, the previously saved seed is used. Random numbers are needed to generate fake training and test data (when use_mnist is false) and also to initialize the network ndarrays. A pickled seed is not included with the repository and so the first run needs to have this set as True. Default is True.

### iterations

The iterations parameter is the number of training iterations. Default is 60.

### minibatch_size

Set minibatch_size to the number of training inputs per gradient descent. The number of training inputs must be a multiple of minibatch_size. However, since number of training inputs isn't known until the data is read, the code will notify and exit if this condition isn't met. Default is 20.

### step_size

Set step_size to the scaling applied when updating the weights for gradient descent. Default for MNIST is 0.01 / minibatch_size.

### layers (tuple)

Set layers to the width of hidden layers. There is no need to include the width of input or output layers. Those are handled automatically. Default is (84, 20).

## Algorithm details

- Straightforward, textbook, implementation of a feed-forward network.
- RLU nonlinearity for hidden layers.
- Sigmoid nonlinearity for output layer.
- Cross entropy cost function.

## Status update

The code is stable. I would like to add some form of regularization as well as additional details to the performance plots. Also, as I continue to improve my Python skills elsewhere, I may come back and refactor some sections.

## Code history

The initial commit(s) did not utilize Python well and also hardcoded the network depth. After reading "Learning Python, 5th Ed." by Mark Lutz, the Python code got better. Subsequently, I tried to make the network depth configurable by adding an extra dimension to the ndarrays. However, the code became very complicated and so I switched to lists of ndarrays. This code is simpler.
