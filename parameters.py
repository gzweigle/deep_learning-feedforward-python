"""Sets the user configurable parameters for the program."""
#
# by Greg C. Zweigle
#

class Parameters:
    "User configurable parameters class."""

    def __init__(self):
        """ Initialize user configurable parameters."""

        # When True, use MNIST, otherwise generate fake data.
        self.use_mnist = True

        # When True, generate and pickle a new random number seed.
        # Otherwise, read the previously pickled random number seed.
        self.new_seed = True

        # Number of training iterations to run.
        self.iterations = 60

        # Number of inputs to train per gradient descent operation.
        self.minibatch_size = 20

        # Step size of gradient descent.
        self.step_size = 0.1

        # Drop out percent.
        self.dropout_probability = 0.5

        # Set to either softmax or sigmoid.
        self.output_nonlin_type = "softmax"

        # Width of (number of neurons in) each hidden layer.
        if self.use_mnist:
            self.layers = (28*3, 20)
        else:
            self.layers = (4, 3)

    def display_parameters(self):
        """Display the selected parameters for this iteration."""

        print_string = (
            "MNIST({0:d}) Seed({1:d}) Iterations({2:d}) " +
            "Batch_size({3:d}) Step_size({4:4.2f}) Dropout({5:3.1f}) " +
            "Output_nonlinearity({6}) Input_width({7}) " +
            "Neurons_per_layer({8})")
        print(print_string.format(
            self.use_mnist, self.new_seed, self.iterations,
            self.minibatch_size, float(self.step_size),
            float(self.dropout_probability), self.output_nonlin_type,
            self.layers[0], self.layers[1:-1]))

    def validate(self, number_of_inputs):
        """Check to ensure valid parameters."""

        if (round(number_of_inputs / self.minibatch_size) *
                self.minibatch_size != number_of_inputs):

            print_string = (
                "The number of training inputs must be a multiple of the " +
                "minibatch size. The number of training inputs is determined " +
                "by the input data. So, to meet this constraint, it may be " +
                "easiest to change the minibatch size parameter. Present " +
                "values: Number of training inputs = {0}  minibatch size = " +
                "{1}.")

            print(print_string.format(number_of_inputs, self.minibatch_size))
            quit()
