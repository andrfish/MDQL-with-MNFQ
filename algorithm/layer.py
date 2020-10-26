###################################################
# This file represents a layer in a neural network
#
# You can build the network automatically using the
# neural.py implementation instead of using this
# class directly
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import neuron
import workers
import activation
import exception
import random

from copy import deepcopy

class main():
    ####################################################################
    # This method initalizes the layer with the following arguments:
    # size    == the number of neurons in the layer
    # learningR == the learning rate for this layer
    # And the following optional argument:
    # neuronL == the list of neurons in this layer. Size must be equal
    #            to the length of this list. The argument is optional, 
    #            if none is specified, a list of them will be randomly
    #            created based on the size.
    # nextL   == the next layer connected to this layer. The argument is
    #            optional and if it is not passed, you are saying that this 
    #            is the output layer.
    # seed    == a seed for the random generator
    ####################################################################
    def __init__(self, size, learningR, neuronL = None, nextL = None, seed = None):
        # Error checking for the size
        if (size <= 0): 
            raise exception.InvalidLayerCount("The layer size must be a positive number (" + size + ")")

        # Set the seed if needed
        if(seed != None):
            self.rnd = random
            self.rnd.seed(seed)
        else:
            self.rnd = random

        # If a neuron list was passed, check that the size is equal to it
        if(neuronL != None):
            if(size != len(neuronL)):
                raise exception.InvalidLayerCount("The layer size (" + str(size) + ") must be equal to the neuron list length (" + str(len(neuronL)) + ")")
            else:
                self.neuronList = neuronL

        # Else, need to create the list of neurons
        else:
            self.neuronList = []
            for _i in range(0, size):
                # First, randomly create weights, if needed, in the range -0.5 to 0.5
                weights = None
                if(nextL):
                    weights = []
                    for _j in range(0, nextL.count):
                        weights.append(self.rnd.uniform(-0.5, 0.5))

                # Next, create the neuron with a bias of 1.0
                self.neuronList.append(neuron.main(deepcopy(weights), 1.0))
        
        self.count = int(size)

        if(nextL != None):
            self.nextLayer = nextL
        else:
            self.nextLayer = None

        # The code assumes that this will be the same for all layers
        self.learningRate = float(learningR)

    ####################################################################
    # This method propagates forward in the network from this layer
    ####################################################################
    def forward(self):
        # Check that there is a layer to propagate forward to
        if(self.get_next_layer() == None):
            raise exception.NoConnectionException("There is no layer to propagate forward to")

        neurons = self.get_neuron_list()
        nextNeurons = self.get_next_layer().get_neuron_list()   
        for i in range(0, self.get_next_layer().get_count()):
            # Set the output
            neuron = nextNeurons[i]
            neuron.set_output(workers._forward_worker(neuron, neurons, i, self.get_count()))

    ####################################################################
    # This method propagates backward in the network from this layer
    ####################################################################
    def backward(self, learn = True):
        # Check that there is a layer to propagate backward from
        if(self.get_next_layer() == None):
            raise exception.NoConnectionException("There is no layer to propagate backward from")

        neurons = self.get_neuron_list()
        nextNeurons = self.get_next_layer().get_neuron_list()
        for i in range(0, self.get_count()):
            # Set the error and weights if applicable
            neuron = neurons[i]
            ret = workers._backward_worker(neuron, nextNeurons, learn, self.get_next_layer().get_count(), self.learningRate)
            neuron.set_error(ret[0][0])

            if(learn):
                neuron.set_weights(ret[1])
        
        # Update the biases for the next layer if applicable
        if(learn):
            for i in range(0, len(nextNeurons)):
                curNeuron = nextNeurons[i]
                curBias = curNeuron.get_bias()
                if(curBias != None):
                    newBias = curBias + (self.learningRate * curNeuron.get_error() * activation.sigmoid(curNeuron.get_output()))
                    curNeuron.set_bias(newBias)

    ####################################################################
    # This method sets the next layer for this layer and randomly
    # generates the weights to it from this layer's neurons
    # You can pass false to not regen the weights but this should only
    # be done if you have already set the weights for the neurons and 
    # are sure it is the correct number of connections needed
    ####################################################################
    def set_next_layer(self, nextL, regenWeights = True):
        self.nextLayer = nextL

        if (regenWeights):
            for i in range(0, self.count):
                    # Randomly create weights in the range -0.5 to 0.5
                    weights = []
                    for _j in range(0, self.nextLayer.count):
                        weights.append(self.rnd.uniform(-0.5, 0.5))

                    # Assign weights to the neuron
                    self.neuronList[i].set_weights(weights)

    ####################################################################
    # This method returns the neuron list for this layer
    ####################################################################
    def get_neuron_list(self):
        return self.neuronList

    ####################################################################
    # This method returns the count for this layer
    ####################################################################
    def get_count(self):
        return int(self.count)
    
    ####################################################################
    # This method returns the next layer for this layer
    ####################################################################
    def get_next_layer(self):
        return self.nextLayer