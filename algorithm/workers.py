###################################################
# This file represents workers used in my neural
# network code. This class should ONLY be used
# internally.
#
# Author: Andrew Fisher
###################################################

import exception
import activation

####################################################################
# Calculate the output for each neuron in the next layer
# This method should NOT be called manually. It is for internal use
# only.
####################################################################
def _forward_worker(neuron, neurons, i, count):
    output = 0
    for j in range(0, count):
        output += neurons[j].get_output() * neurons[j].weightList[i]
    
    # Add the bias if there is one
    if (neuron.get_bias() != None):
        output += neuron.get_bias()
    
    # Return the output
    return float(output)

####################################################################
# Calculate the error for each neuron and update their weights
# This method should NOT be called manually. It is for internal use
# only.
####################################################################
def _backward_worker(neuron, neurons, learn, count, learningRate):
    error = 0
    weights = neuron.get_weights()

    # Cycle through each connection's neuron
    for j in range(0, count):
        nextOutput = neurons[j].get_output()

        # Ensure that the output has been set
        if (nextOutput == None):
            raise exception.OutputNotSet("The output for this neuron is not set (have you propagated forward yet?)")

        nextError = neurons[j].get_error()
        activationVal = activation.sigmoid(nextOutput)

        # Add on to the error
        error += nextError * activationVal * neuron.get_weight(j)
        
        # Check if it should learn
        if(learn):
            # Update the weight
            weights[j] += (learningRate * nextError * activationVal * neuron.get_output())
        
    # Return the output
    ret = []
    temp = []
    temp.append(error)
    ret.append(temp)
    ret.append(weights)
    return ret