###################################################
# This file represents a neuron in a layer in a 
# neural network
#
# You can build the network automatically using the
# neural.py implementation instead of using this
# class directly
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import random
import exception

class main():
    ####################################################################
    # This method initalizes the layer with the following arguments:
    # weightL == the list of weights to the next layer if applicable.
    #            it is assumed that there is a 1-to-1 relationship with
    #            this list and the next layer. That is, the first element
    #            in the list is the weight to the first neuron in the 
    #            next layer and so forth. The argument is optional and 
    #            if it is not passed, you are saying that it is an output
    #            neuron.
    # biasV   == the bias for this neuron. The argument is optional.
    ####################################################################
    def __init__(self, weightL = None, biasV = None):
        self.weightList = weightL

        if(biasV != None):
            self.bias = float(biasV)
        else:
            self.bias = None

        self.output = None
        self.error = None
    
    ####################################################################
    # This method sets a weight for the neuron
    ####################################################################
    def set_weight(self, i, weight):
        self.weightList[i] = float(weight)

    ####################################################################
    # This method sets all of the weights for the neuron
    ####################################################################
    def set_weights(self, weights):
        self.weightList = weights

    ####################################################################
    # This method sets the output for the neuron
    ####################################################################
    def set_output(self, outputV):
        self.output = float(outputV)

    ####################################################################
    # This method sets the bias for the neuron
    ####################################################################
    def set_bias(self, biasV):
        self.bias = float(biasV)

    ####################################################################
    # This method sets the error for the neuron
    ####################################################################
    def set_error(self, errorV):
        self.error = float(errorV)

    ####################################################################
    # This method returns a weight
    ####################################################################
    def get_weight(self, i):
        return float(self.weightList[i])

    ####################################################################
    # This method returns all weights
    ####################################################################
    def get_weights(self):
        return self.weightList

    ####################################################################
    # This method returns the output for this neuron
    ####################################################################
    def get_output(self):
        return (None if (self.output == None) else float(self.output))

    ####################################################################
    # This method returns the bias for this neuron
    ####################################################################
    def get_bias(self):
        return (None if (self.bias == None) else float(self.bias))

    ####################################################################
    # This method returns the error for this neuron
    ####################################################################
    def get_error(self):
        return (None if (self.error == None) else float(self.error))