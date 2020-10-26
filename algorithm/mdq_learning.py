###################################################
# This file demonstrates my modifed deep
# q-learning algorithm
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import sys
import time

import neural
import exception
import error
import mnfq_iteration
import random
import math
from copy import deepcopy

class main():
    ####################################################################
    # This method initalizes the MDQ network with the following arguments:
    # states         == a list of the possible states in the dataset
    # actions        == a list of the possible actions in the dataset
    # learningR      == the learning rate for the algorithm
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
    # networkF       == a file name that points to an export as define in the
    #                   export_network method for this network to import
    # networkT       == an array of text that contains the export data of
    #                   a network, as defined in the export_network method
    # Plus ONE of the following arguments for the MNFQ networks:
    # mnfqL          == a 1D array of the layers for the network. It is
    #                   assumed that the layers have already been populated
    #                   with neurons and that the ordering of the elements
    #                   is such that the first element is the input layer
    #                   which is connected to the second to N elements for
    #                   the hidden layer, and the last layer is the output
    #                   layer. This argument has priority over networkMatrix.
    # mnfqM          == a 1D array where the element is the layer
    #                   and the value is the number of neurons in that layer.
    #                   It is assumed that the matrix is built such that the 
    #                   first element is the input layer which is connected 
    #                   to the second to N elements for the hidden layer, 
    #                   and the last layer is the output layer.
    # mnfqF          == a 2D set of file names that points to an export as defined in the
    #                   export_network method for this network to import
    # mnfqT          == a 2D set of arrays of text that contains the export data of
    #                   a network, as defined in the export_network method
    # And the following optional arguments:
    # onehot         == change the inputs to onehot encoding. Index based by default
    # table          == a 2D array of the transition probability table to
    #                   initialize this for all 52 weeks of probabilities. 
    #                   The height/width must be equal to len(states) / len(actions) if passed
    # outputs        == an array of final populations for the states to let the
    #                   MNFQ networks compare to
    # seed           == a seed for the random object to produce results that can
    #                   be replicated
    # impossible     == if the table is not defined, this states which (if any) of the
    #                   transitions are NOT allowed
    ####################################################################
    def __init__(self, states, actions, learningR, 
                 networkL = None, networkM = None, networkF = None, networkT = None,
                 mnfqL = None, mnfqM = None, mnfqF = None, mnfqT = None,
                 onehot = False, table = None, outputs = None, seed = None, impossible = None):
        
        # Check that the states and actions were passed
        if(states == None):
            raise exception.MissingArgument("The states have not been set")
        if(actions == None):
            raise exception.MissingArgument("The actions have not been set")

        # Check that the states and actions are valid
        if(len(states) == 0):
            raise exception.InvalidLength("The amount of states must be greater than zero")
        if(len(actions) == 0):
            raise exception.InvalidLength("The amount of states must be greater than zero")

    	#The number of states and actions should be equal
    	#The action performed is the new state
        if(len(states) != len(actions)):
    	    raise exception.InvalidLength("The number of states must equal the number of actions")
        
        #Check if the layers were passed
        if(networkL != None):
            # Check that the learning rate was passed
            if(learningR == None):
                raise exception.NoLearningRate("The learning rate must be passed if creating the network from a layer array")
            self.network = neural.main(learningR, networkL=networkL, seed=seed)

        # Else, check if we need to generate the network from the matrix
        elif(networkM != None):
            # Check that the learning rate was passed
            if(learningR == None):
                raise exception.NoLearningRate("The learning rate must be passed if creating the network from a matrix")
            self.network = neural.main(learningR, networkM=networkM, seed=seed)
        
        # Else, import the file
        elif(networkF != None):
            self.network = neural.main(learningR, networkF=networkF, seed=seed)
        
        # Else, import from array
        else:
            self.network = neural.main(learningR, networkT=networkT, seed=seed)

        # Store the passed values
        self.learningRate = learningR
        self.states = states
        self.actions = actions
        self.onehot = onehot

        # Set the seed if needed
        if(seed != None):
            self.rnd = random
            self.rnd.seed(seed)
        else:
            self.rnd = random

        # Check if the table was passed
        self.tables = list()
        if(table != None):
            # Make sure it is the correct size
            if(len(table) != len(states) or len(table[0]) != len(actions)):
                raise exception.NotEqualLength("The passed table must be equal to the number of states by the number of actions")

            for i in range(0, 52):
                self.tables.append(deepcopy(table))
            self.random_table = False
        else:
            # Randomly initialize the table
            table = [[0 for x in range(len(actions))] for y in range(len(states))]
            temp = [0 for x in range(len(states))]
            self.impossible = impossible

            for i in range(0, len(states)):
                for j in range(0, len(actions)):
                    # See if this is allowed
                    pair = [i, j]

                    if(pair in impossible): val = 0
                    else:                   val = self.rnd.random()

                    table[i][j] = val
                    temp[i] += table[i][j]
            
            # Normalize it
            for i in range(0, len(states)):
                t = temp[i]
                for j in range(0, len(actions)):
                    table[i][j] /= t

            # Add to the algorithm
            for i in range(0, 52):
                self.tables.append(deepcopy(table))
            self.random_table = True

        # Initialize the mnfq network
        self.mnfq = mnfq_iteration.main(len(states), len(actions), learningR, mnfqL, mnfqM, mnfqF, mnfqT, outputs)

    ####################################################################
    # This method performs an epoch in the modified deep q-learning algorithm.
    # The population defines the population where each value corresponds to a member's state. 
    # This method returns the loss from the previous iteration.
    ####################################################################
    def perform_epoch(self, population, currentSubE, epochsLeft, learn):
        # Check that the population is set
        if (len(population) == 0):
            raise exception.InvalidLength("The population size must be positive")
        
        # Stores the current state of the table
        oldTable = [[0 for x in range(len(self.actions))] for y in range(len(self.states))]
        
        # Store the q-values as well as rewards in a 3D array
        # The first dimension is the state, the second is the action- which stores the q-value or reward summation
        populationQs = [[0 for x in range(len(self.actions))] for y in range(len(self.states))]
        populationRs = [[0 for x in range(len(self.actions))] for y in range(len(self.states))]
        
        # Populate the old table and store it in the MNFQ algorithm
        ind = (currentSubE % 52)
        table = self.tables[ind]
        oldTable = deepcopy(table)

        # Get the current populations
        statePopulations = [0 for x in range(len(self.states))]
        for i in range(len(population)):
            for j in range(len(self.states)):
                if(population[i] == j):
                    statePopulations[j] += 1
                    pass
        
        # Run (population) number of sub-epochs
        qVals = list()

        # If the probability table was read from file or we're testing, use it perfectly
        if(not self.random_table or not learn):
            partitions = [[0 for x in range(len(self.actions))] for y in range(len(self.states))]
            for i in range(len(self.states)):
                for j in range(len(self.actions)):
                    # Determine number of individuals to move
                    partitions[i][j] = int(statePopulations[i] * table[i][j])

                # Ensure this adds up to the original pop
                diff = sum(partitions[i]) - statePopulations[i]
                if (diff != 0):
                    # Add to highest prob if not
                    high_prob = -1.0
                    high_ind = -1
                    for k in range(len(self.actions)):
                        if (table[i][k] > high_prob):
                            high_prob = table[i][k]
                            high_ind = k
                    partitions[i][high_ind] += abs(diff)

            # Cycle each individual through the algorithm
            for i in range(len(self.states)):
                for j in range(len(self.actions)):
                    for _p in range(partitions[i][j]):
                        #Set the input of the network
                        self.set_input(i, j)
                        
                        #Propagate forward in the network
                        self.network.propagate_forward(False)
                        
                        if(learn):
                            # Get the q-value and reward
                            qValue = self.network.get_network_output()[0]
                            # Normalize it
                            qValue = (qValue) / (qValue + table[i][j])

                            reward = -1
                            reward = self.mnfq.calc_immediate_offset(i, j, oldTable, statePopulations[j], epochsLeft, qValue)
                        
                            if(math.isnan(reward)):
                                reward = 0
                            elif (math.isinf(reward)):
                                reward = 0
                            
                            # Store these in their array lists
                            qVals += [qValue]
                            populationQs[i][j] += qValue
                            populationRs[i][j] += reward

                        # Change the populations
                        statePopulations[i] -= 1
                        statePopulations[j] += 1

        # Else, follow the random approach
        else:
            for i in range(len(population)):
                # Select a random action to take
                state = population[i]
                action = self.get_rand_action(state, table)
                
                #Set the input of the network
                self.set_input(state, action)
                
                #Propagate forward in the network
                self.network.propagate_forward(False)
                
                if(learn):
                    # Get the q-value and reward
                    qValue = self.network.get_network_output()[0]
                    # Normalize it
                    qValue = (qValue) / (qValue + table[state][action])

                    reward = -1
                    reward = self.mnfq.calc_immediate_offset(state, action, oldTable, statePopulations[action], epochsLeft, qValue)
                
                    if(math.isnan(reward)):
                        reward = 0
                    elif (math.isinf(reward)):
                        reward = 0
                    
                    # Store these in their array lists
                    qVals += [qValue]
                    populationQs[state][action] += qValue
                    populationRs[state][action] += reward
                
                # Update the population member's state
                population[i] = action

                # Change the populations
                statePopulations[state] -= 1
                statePopulations[action] += 1

        if(learn):
            sumS = [0 for x in range(len(self.states))]
            # Calculate the new q-values
            # Also, get sum for normalizing
            for i in range(len(self.states)):
                for j in range(len(self.actions)):
                    multiplier = 1
                    offset = self.mnfq.calc_offset(i, j, oldTable, qVals, statePopulations[j], epochsLeft)
                    
                    if(math.isnan(offset)):
                        offset = 0
                    elif (math.isinf(offset)):
                        if(offset < 0):
                            offset = -1
                        else:
                            offset = 1
                    
                    # Only change if this transition is possible
                    if(table[i][j] != 0):
                        table[i][j] += offset * self.learningRate
                        
                        if(offset < 0): multiplier = -1
                        
                        table[i][j] += multiplier * ((1 - populationRs[i][j]) / len(population)) * self.learningRate
    
                    # Base case checking
                    # Seems to happen if initial population was zero
                    # Or offset is making the value negative
                    if(math.isnan(table[i][j])):
                        table[i][j] = 0

                    elif (math.isinf(table[i][j])):
                        table[i][j] = 0

                    elif (table[i][j] < 0):
                        table[i][j] = -table[i][j]
                    
                    sumS[i] += table[i][j]

            # Normalize the q-values between zero and one
            for i in range(len(self.states)):
                summ = sumS[i]
                for j in range(len(self.actions)):
                    table[i][j] = table[i][j] / summ
            
            # Calculate the error
            error = self.out_error(populationQs, len(population), table)
            
            # Set the error on the output neuron
            err = [error]
            self.network._set_output_error(err)
            
            # Perform back-propagation
            self.network.propagate_backward(True)
        
        self.tables[ind] = table
        if(learn):
            # Get the loss
            loss = self.avg_loss(table, oldTable)
            return loss, statePopulations
        else:
            return -1, statePopulations

    ####################################################################
    # This method sets the input of the network. The int values
    # correspond to an index in the states and actions arrays
    ####################################################################
    def set_input(self, state, action):
        # Set the input
        input = None
        
        # Check input specification
        if(self.onehot):
            input = [0 for x in range(len(self.states) + len(self.actions))]

            for i in range(len(self.states)):
                input[i] = 1 if i == state else 0
            
            for i in range(len(self.actions)):
                input[len(self.states) + i] = 1 if i == action else 0
        else:
            input = [0 for x in range(2)]
            
            input[0] = state
            input[1] = action
        
        self.network.set_input(input)

    ####################################################################
    # This method uses roulette wheel to randomly select an action
    ####################################################################
    def get_rand_action(self, state, table):
        # Pick a random number
        sel = -1
        rand = self.rnd.random()
        
        # Determine which action to pick
        temp = 0
        actions = table[state]
        for i in range(len(actions)):
            temp += actions[len(actions) - i - 1]
            if(rand < temp):
                sel = i
        
        # If not valid, pick highest probability that is valid 
        pair = [state, sel]
        if (pair in self.impossible):
            high_ind = -1
            high_prob = -1

            for action in range(len(actions)):
                pair = [state, action]
                if(pair not in self.impossible and actions[action] > high_prob):
                    high_prob = actions[action]
                    high_ind = action

            sel = high_ind

        return sel

    ####################################################################
    # This method determines the error for the output neruon
    ####################################################################
    def out_error(self, popQs, popSize, table):
        ret = 0
        
        # Run through each q-value and minus the average
        # population's q-value from it. Add that to a sum
        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                ret += (table[i][j] - (popQs[i][j] / popSize))
        
        # Divide the sum by the number of q-values for the average error
        ret /= (len(self.states) * len(self.actions))
        
        return ret

    ####################################################################
    # This method determines the average loss for the population
    # based on the old q-values
    ####################################################################
    def avg_loss(self, table, oldTable):
    	ret = 0
    	
    	# Calculate the difference for each state
    	for i in range(len(self.states)):
    		# Cycle through each action in the state
    		for j in range(len(self.actions)):
    			# Add the difference onto the return value
    			ret += math.pow((table[i][j] - oldTable[i][j]), 2)
    	
    	# Divide it to get the average
    	ret /= (len(self.actions) * len(self.states))
    	
    	return ret

    ####################################################################
    # This method sets the output of the MNFQ network. The int values
    # correspond to an index in the states and actions arrays
    ####################################################################
    def set_output(self, output):
    	self.mnfq._set_output(output)