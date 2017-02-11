# Over one minibatch of data, update the network matrices.
#
# To Do: Doesn't yet work for minibatch > 1.
#        Need to add regularization, probably with dropout.
#        Refactoring:
#           Final_out isn't a good way to return values.
#           Passing in too many values.
#

# External modules.
import numpy as np

# Modules specific to this program.
import backpropagate as bp
import feedforward_network as fn
import gradient_descent as gd


def training_iteration(inval, expected_val, A2, A3, A4, b2, b3, b4, datalen, minibatch):

    for i in range(datalen):

        for j in range(minibatch):
            
            # Run one iteration of the network
            zout = fn.feedforward_network(inval[i,:], A2, b2, A3, b3, A4, b4)

            # Run a back propagation
            dout = bp.backpropagate(zout, A3, A4, expected_val[i,:])

        # Update the matrices
        final_out = gd.gradient_descent(inval[i,:], zout, dout, A2, A3, A4, b2, b3, b4)
        
        # Save results for the next iteration, trying another method of returning values.
        A2 = final_out[0]
        A3 = final_out[1]
        A4 = final_out[2]
        b2 = final_out[3]
        b3 = final_out[4]
        b4 = final_out[5]
        
    # Return the output for the last stage and the last input tested against.
    return zout[2]