###################################################
# This file demonstrates the homelessness
# simulation referred to as MDQL with MNFQ
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import mdq_learning
import mnfq_iteration

import matplotlib.pyplot as plt
import time
from os import listdir
from os.path import isfile, join
from copy import deepcopy

def main():
        #############################################################################
        # Declare whether or not to use pre-defined probabilities
        random_probs = False

        # Define learning rates and epochs to try out
        learningRates = [0.01]
        epochList = [10]

        # Define a random seed to replicate results (set to None for a random seed)
        seed = 22

        # Declare whether or not onehot encoding should be used
        onehot = True
        #############################################################################

        # Define the states
        stateNames = ["S1", "S2", "S3"]
        
        # Declare the directory to import the populations from for training
        importDir = "../data/output/"
        
        # Declare an initial transition probability table to speed up training (optional, set to None if not being used)
        if(not random_probs):
            probsFile = "../data/sample_data.csv"
        else:
            probsFile = None

        # If probabilities aren't defined, state if any transitions aren't allowed (optional, set to None if not being used)
        impossible = None
        
        # Run the algorithm
        states = len(stateNames)
        for lr in learningRates:
            for epochs in epochList:
                print("Running the algorithm on the sanity dataset with:\nLearning rate (Lr)= \"" + str(lr) +  
                      "\"\nTraining epoch length (E)= \"" + str(epochs) + 
                      "\"\n" + ('RANDOM' if probsFile == None else 'GIVEN') + " probabilities (P)\n" + 
                      ('RANDOM seed (S)' if seed == None else 'Seed (S)= ' + str(seed)) + '\n' +
                      ("USING " if onehot else "NOT USING ") + "onehot encoding!\n")

                # If set, read the initial transition probabilities
                startingProbs = None
                if(probsFile != None):
                    startingProbs = [[0 for x in range(states)] for y in range(states)]
                    flag = True
                    count = 0
                    with open(probsFile) as f:
                            for line in f:
                                if(flag):
                                    flag = False
                                else:
                                    temp = line.split(',')
                                    
                                    for i in range(states):
                                        startingProbs[count][i] = float(temp[i + 1]) * 0.01
                                    
                                    count += 1
                
                # Read the input data to a database
                database = list()

                importFiles = [f for f in listdir(importDir) if isfile(join(importDir, f))]
                importFiles.sort()

                for files in importFiles:
                    # Next, add to the database
                    population = [0 for x in range(states)]
                    total = 0
                    i = 0
                    with open(importDir + files) as f:
                            for line in f:
                                population[i] = int(line)
                                total += population[i]
                                i += 1

                    pop = [0 for x in range(2)]

                    tempPop = [0 for x in range(total)]
                    tmp = 0
                    for i in range(states):
                        for j in range(population[i]):
                            tempPop[tmp] = i
                            tmp += 1

                    pop[0] = population
                    pop[1] = tempPop
                    database.append(pop)

                # Create the network to use
                networkMatrix = list()

                # Define the input layer
                networkMatrix.append(states * 2 if onehot else 2)

                # Define four hidden layers with 4, 8, 16, and 24 neurons each respectively
                # This seems to produce the best outputs from testing but, since the weights
                # are randomly initialized, performance will vary
                networkMatrix.append(4)
                networkMatrix.append(8)
                networkMatrix.append(16)
                networkMatrix.append(24)

                # Define the output layer
                networkMatrix.append(1)

                # Define the MNFQ networks' layouts
                mnfqM = list()

                # Define the input layer
                mnfqM.append(3)

                # Define two hidden layers with 6 and 12 neurons each respectively
                # This seems to produce the best outputs from testing but, since the weights
                # are randomly initialized, performance will vary
                mnfqM.append(6)
                mnfqM.append(12)

                # Define the output layer
                mnfqM.append(1)

                algorithm = mdq_learning.main(states=       stateNames, 
                                              actions=      stateNames, 
                                              learningR=    lr, 
                                              networkM=     networkMatrix, 
                                              mnfqM=        mnfqM, 
                                              onehot=       onehot, 
                                              table=        startingProbs,
                                              seed=         seed,
                                              impossible=   impossible)

                # Train the algorithm
                for i in range(epochs):
                    start = time.time()

                    # Cycle through all weeks
                    loss = 0.0
                    count = 0
                    for j in range(len(database) - 1):
                        # Set the output
                        algorithm.set_output(deepcopy(database[j + 1][0]))

                        # Run the sub-epochs
                        lossT, _statePopulations = algorithm.perform_epoch(deepcopy(database[j][1]), j, 1, True)
                        loss += lossT
                        count += 1

                    duration = time.time() - start
                    loss /= count
                    print("Epoch " + str(i + 1) + " loss of " + str(loss) + " after " + str(duration) + "s")
                print("\nComplete!\n")
                
                # Test the algorithm
                print("Testing the algorithm with the initial population:")

                # Get the initial population
                for i in range(len(database[0][0])):
                    print(stateNames[i] + ": " + str(database[0][0][i]))

                # Run the sub-epochs
                graph = list()
                population = database[0][1]
                graph.append(states_from_pop(population, states))
                for i in range(len(database) - 1):
                    lossT, population = algorithm.perform_epoch(deepcopy(population), i, 1, False)
                    graph.append(population)
                    population = pop_from_state(population, states)

                # Display final results
                print("\nFinal populations: ")
                population = states_from_pop(population, states)
                for i in range(states):
                    print(stateNames[i] + ": " + str(population[i]))
                print("Complete!\n")
                
                # Graph the results
                plt.figure()
                plt.title("Lr= " + str(lr) + ", E= " + str(epochs) + ", P: " + ('GIVEN' if probsFile != None else 'RANDOM') + ", S: " + ('RANDOM' if seed == None else str(seed)))
                plt.plot(graph)
                plt.draw()
                plt.pause(0.1) 
        plt.show()

###################################################
# This method takes a population of individuals
# and converts them to an array of state populations
###################################################
def states_from_pop(population, num_states):
    ret_pop = list()

    for i in range(num_states):
        count = 0
        for j in range(len(population)):
            count += (1 if population[j] == i else 0)
        ret_pop.append(count)

    return ret_pop

###################################################
# This method takes a set of state populations
# and converts them to an array of individuals
###################################################
def pop_from_state(population, num_states):
    total = sum(population)
    tempPop = [0 for x in range(total)]
    tmp = 0
    for i in range(num_states):
        for j in range(population[i]):
            tempPop[tmp] = i
            tmp += 1

    return tempPop

if __name__ == '__main__':
        main()