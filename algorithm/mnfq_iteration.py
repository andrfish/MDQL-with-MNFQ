###################################################
# This file demonstrates my modifed deep
# neural fitted q-iteration algorithm
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import sys

import neural
import exception
import error
import math

class main():
    ####################################################################
    # This method initalizes the MNFQ algorithm with the following arguments:
    # states         == a count of the number of states
    # actions        == a count of the number of actions
    # learningR      == the learning rate for the network. An optional
    #                   argument IFF you pass the networkF argument; if 
    #                   it is passed with networkF, it will override
    #                   whatever value it reads from the file
    # ONE of the following arguments:
    # networkL       == a 1D array of the layers for the network. It is
    #                   assumed that the layers have already been populated
    #                   with neurons and that the ordering of the elements
    #                   is such that the first element is the input layer
    #                   which is connected to the second to N elements for
    #                   the hidden layer, and the last layer is the output
    #                   layer. This argument has priority over networkMatrix.
    # networkM       == a 1D array where the element is the layer
    #                   and the value is the number of neurons in that layer.
    #                   It is assumed that the matrix is built such that the 
    #                   first element is the input layer which is connected 
    #                   to the second to N elements for the hidden layer, 
    #                   and the last layer is the output layer.
    # networkF       == a 2D set of file names that points to an export as defined in the
    #                   export_network method for this network to import
    # networkT       == an 2D set of arrays of text that contains the export data of
    #                   a network, as defined in the export_network method
    # And the following optional argument:
    # outputs        == the outputs to compare the networks' to. 
    #                   An optional array for each network but it must be set 
    #                   before propgating forward
    ####################################################################
    def __init__(self, states, actions, learningR, networkL = None, networkM = None, networkF = None, networkT = None, outputs = None):
        
        # Check that the states and actions were passed
        if(states == None):
            raise exception.MissingArgument("The states have not been set")
        if(actions == None):
            raise exception.MissingArgument("The actions have not been set")

        # Check that the states and actions are valid
        if(states == 0):
            raise exception.InvalidLength("The amount of states must be greater than zero")
        if(actions == 0):
            raise exception.InvalidLength("The amount of states must be greater than zero")

    	#The number of states and actions should be equal
    	#The action performed is the new state
        if(states != actions):
    	    raise exception.InvalidLength("The number of states must equal the number of actions")

        # Initialize a list of networks
        self.networks = [[None for x in range(actions)] for y in range(states)]

        #Check if the layers were passed
        if(networkL != None):
            # Check that the learning rate was passed
            if(learningR == None):
                raise exception.NoLearningRate("The learning rate must be passed if creating the network from a layer array")

            # Create (states by actions) networks
            for i in range(states):
                for j in range(actions):
                    if(outputs != None):
                        self.networks[i][j] = neural.main(learningR, outputA=outputs[j], networkL=networkL)
                    else:
                        self.networks[i][j] = neural.main(learningR, networkL=networkL)

        # Else, check if we need to generate the network from the matrix
        elif(networkM != None):
            # Check that the learning rate was passed
            if(learningR == None):
                raise exception.NoLearningRate("The learning rate must be passed if creating the network from a matrix")

            # Create (states by actions) networks
            for i in range(states):
                for j in range(actions):
                    if(outputs != None):
                        self.networks[i][j] = neural.main(learningR, outputA=outputs[j], networkM=networkM)
                    else:
                        self.networks[i][j] = neural.main(learningR, networkM=networkM)
        
        # Else, import the file
        elif(networkF != None):
            # Create (states by actions) networks
            for i in range(states):
                for j in range(actions):
                    if(outputs != None):
                        self.networks[i][j] = neural.main(learningR, outputA=outputs[j], networkF=networkF[i][j])
                    else:
                        self.networks[i][j] = neural.main(learningR, networkF=networkF[i][j])
        
        # Else, import from array
        else:
            # Create (states by actions) networks
            for i in range(states):
                for j in range(actions):
                    if(outputs != None):
                        self.networks[i][j] = neural.main(learningR, outputA=outputs[j], networkT=networkT[i][j])
                    else:
                        self.networks[i][j] = neural.main(learningR, networkT=networkT[i][j])

        # Store the passed values
        self.learningRate = learningR
        self.states = states
        self.actions = actions
        self.outputs = outputs

    def calc_immediate_offset(self, i, j, table, curPop, weeksLeft, curQ):
        # Cycle each q-value through its network
        # This will store the lowest absolute error and the corresponding q-value
        lowestError = float('-inf')
        alphaQ = curQ
        
        inp = [curPop, weeksLeft, 0]
        output = [self.outputs[j]]
        
        curNetwork = self.networks[i][j]
        
        # Put the alpha q through the network and back propagate
        inp[2] = alphaQ
        curNetwork.set_output(output)
        curNetwork.set_input(inp)
        curNetwork.propagate_forward(True)
        
        networkOutput = curNetwork.get_network_output()[0]
        lowestError = self.outputs[j] - networkOutput
        
        # If the error is less than 20%, back propagate
        avg = (self.outputs[j] + networkOutput) / 2
        perError = 100 * (abs(lowestError) / avg)
        
        if(abs(perError) < 20):
            curNetwork.propagate_backward(True)
        
        # Sum up the q-values (with alpha instead of the value at that state)
        summ = 0
        for k in range(len(table)):
            if(k != i):
                summ += table[k][j]
            else:
                summ += alphaQ
        
        ret = alphaQ / summ

        try:
            ret *= (-lowestError)/self.outputs[j]
        except: # Output is likely zero
            ret *= (-lowestError)
        
        return ret


    def calc_offset(self, i, j, table, qValues, curPop, weeksLeft):
        # Cycle each q-value through its network
        # This will store the lowest absolute error and the corresponding q-value
        lowestError = float('inf')
        alphaQ = float('inf')
        
        inp = [curPop, weeksLeft, 0]
        output = [self.outputs[j]]
        
        curNetwork = self.networks[i][j]
        for q in qValues:
            # Set the q-value input
            inp[2] = q
            
            # Set the network's input and output
            curNetwork.set_output(output)
            curNetwork.set_input(inp)
            
            # Propagate forward
            curNetwork.propagate_forward(True)
            
            # Determine the relative error
            curOutput = curNetwork.get_network_output()[0]
            error = self.outputs[j] - curOutput
            
            # See if the absolute error is the lowest
            if(abs(error) < abs(lowestError)):
                lowestError = error
                alphaQ = q
        
        # Put the alpha q through the network and back propagate
        inp[2] = alphaQ
        curNetwork.set_output(output)
        curNetwork.set_input(inp)
        curNetwork.propagate_forward(True)
        curNetwork.propagate_backward(True)
        
        # Sum up the q-values (with alpha instead of the value at that state)
        summ = 0
        for k in range(len(table)):
            if(k != i):
                summ += table[k][j]
            else:
                summ += alphaQ
        
        ret = alphaQ / summ
        
        try:
            ret *= (-lowestError)/self.outputs[j]
        except: # Output is likely zero
            ret *= (-lowestError)

        return ret

    def _set_output(self, output):
        self.outputs = output