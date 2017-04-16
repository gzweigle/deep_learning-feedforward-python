"""Setup the parameters and then run complete learning / testing cases."""
#
# by Greg C. Zweigle
#
import learn_then_test as lt
import parameters

print("Deep Learning Neural Network, Feedforward Implementation")

parameters = parameters.Parameters()

use_mnist = [True]*4
new_seed = [True, False, False, False]
iterations = [50]*4
minibatch_size = [20]*4
step_size = [0.1]*4
dropout_probability = [0.5]*4
output_nonlin_type = ['softmax', 'sigmoid', 'softmax', 'softmax']
layers = [(28*3, 20), (28*3, 20), (28*5, 20), (28*8, 40)]

for k in range(len(use_mnist)):

    parameters.use_mnist = use_mnist[k]
    parameters.new_seed = new_seed[k]
    parameters.iterations = iterations[k]
    parameters.minibatch_size = minibatch_size[k]
    parameters.step_size = step_size[k]
    parameters.dropout_probability = dropout_probability[k]
    parameters.output_nonlin_type = output_nonlin_type[k]
    parameters.layers = layers[k]

    (train_err, train_cost, test_err, test_cost) = lt.learn_then_test(parameters)

    if k == len(use_mnist) - 1:
        # On last iteration keep the plot from closing.
        lt.plot_things(train_err, train_cost, test_err, test_cost, True)
    else:
        lt.plot_things(train_err, train_cost, test_err, test_cost, False)
