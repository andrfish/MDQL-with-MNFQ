###################################################
# This file represents a neural network
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import layer
import neuron
import exception
import error

class main():
    ####################################################################
    # This method initalizes the network with the following arguments:
    # learningRate   == the learning rate for the network. An optional
    #                   argument IFF you pass the networkF argument; if 
    #                   it is passed with networkF, it will override
    #                   whatever value it reads from the file
    # inputA         == the input of the network. An optional argument
    #                   but it must be set before propagating forward
    # outputA        == the output to compare the network's to. 
    #                   An optional argument but it must be set before
    #                   propgating forward
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
    # And the following optional argument:
    # seed           == a seed for the random generator
    ####################################################################
    def __init__(self, learningR = None, inputA = None, outputA = None, networkL = None, networkM = None, networkF = None, networkT = None, seed = None):

        # Check if the layers were passed
        if(networkL != None):
            # Check that the learning rate was passed
            if(learningR == None):
                raise exception.NoLearningRate("The learning rate must be passed if creating the network from a layer array")
            self.learningRate = learningR
            self.networkLayers = networkL

        # Else, check if we need to generate the network from the matrix
        elif(networkM != None):
            # Check that the learning rate was passed
            if(learningR == None):
                raise exception.NoLearningRate("The learning rate must be passed if creating the network from a matrix")
            self.learningRate = learningR
            self.networkLayers = []

            # Make sure that the number of layers is valid
            # This implementation assumes that you will have, at least, an
            # input layer and an output layer
            if (len(networkM) < 2):
                raise exception.InvalidLayerCount("There must be at least two layers in the network")

            for i in range(0, len(networkM)):
                # Make sure that the number of neurons is valid
                if (networkM[i] <= 0):
                    raise exception.InvalidNeuronCount("The number of neurons (" + str(networkM[i]) + ") in layer \"" + str(i + 1) + "\"  must be greater than zero")

                # Create a new layer
                curLayer = layer.main(networkM[i], self.learningRate, seed=seed)
                
                # Set it as the next layer for the previous if applicable
                if(i != 0):
                    self.networkLayers[i - 1].set_next_layer(curLayer)
                
                # Add it to the array of layers
                self.networkLayers.append(curLayer)
        
        # Else, import the file
        elif(networkF != None):
            self.import_network(networkF)
            if(learningR != None):
                self.set_learning_rate(learningR)
        
        # Else, import from array
        else:
            self.import_network(None, networkT)
            if(learningR != None):
                self.set_learning_rate(learningR)

        self.size = len(self.networkLayers)

        if(outputA != None):
            self.set_output(outputA)
        else:
            self.output = None
        
        if(inputA != None):
            self.set_input(inputA)

    ####################################################################
    # This method propagates forward through the network and returns
    # the mean squared error for the output if errorRet is True
    ####################################################################
    def propagate_forward(self, errorRet = True):
        # Check that the output has been set if returning the error
        if (self.output == None and errorRet):
            raise exception.OutputNotSet("The output for the network to compare to has not been set")

        # Propagate through the network
        for i in range(0, self.size - 1):
            self.networkLayers[i].forward()

        if (errorRet):
            # Set the error for the output neurons
            outputLayer = self.networkLayers[self.size - 1]
            outputNeurons = outputLayer.get_neuron_list()
            networkOutput = self.get_network_output()
            for i in range(0, len(outputNeurons)):
                outputNeurons[i].set_error(self.output[i] - networkOutput[i])

            # Return the mean squared error
            return error.mse(self.output, networkOutput)

    ####################################################################
    # This method propagates backward through the network. Assumes to 
    # train the networkby default
    # TODO: Add in support for Rprop which is vital for NFQ (currently
    #       using normal back propagation)
    ####################################################################
    def propagate_backward(self, learn = True):
        # Propagate through the network
        for i in range(self.size - 2, -1, -1):
            self.networkLayers[i].backward(learn)

    ####################################################################
    # This method exports the network to an array and/or file depending
    # on the writeToFile argument. If it is true, the filename argument
    # must be passed.
    # First line:        number of layers
    # Second line:       the learning rate
    # Third line:        number of neurons in the input layer (refer to as nI)
    # Fourth to nI line: each neuron will have 5 lines. The first line will
    #                    be the number of weights. The second line will be
    #                    the weights, each separated by a space. The third
    #                    line will be the bias, the fourth the output, and
    #                    the fith the error.
    # The rest of the file will follow a similar format from the third
    # line to the fourth to nO line definitions above until the output
    # layer has been written. Note that the output is NOT exported
    ####################################################################
    def export_network(self, writeToFile, filename = None):
        ret = []

        if(writeToFile == True):
            # Check that the filename was passed
            if(filename == None):
                raise exception.NoFilenameSpecified("The filename was not specified to export to")

        # Add the number of layers
        ret.append(self.size)

        # Add the learning rate
        ret.append(self.learningRate)

        # Add each layer
        for i in range(0, self.size):
            curLayer = self.networkLayers[i]

            # Add number of neurons
            ret.append(curLayer.count)

            # Add each neuron in the layer
            for j in range(0, curLayer.count):
                curNeuron = curLayer.neuronList[j]

                # Add the number of weights
                ret.append(0 if curNeuron.weightList == None else len(curNeuron.weightList))

                # Add each weight
                if(curNeuron.weightList != None):
                    temp = ""
                    for k in range(0, len(curNeuron.weightList)):
                        temp += str(curNeuron.weightList[k]) + " "
                    temp = temp.strip()

                    # Add weights
                    ret.append(temp)
                
                # Add bias
                if(curNeuron.bias != None):
                    ret.append(curNeuron.bias)
                else:
                    ret.append("None")
                
                # Add output
                if(curNeuron.output != None):
                    ret.append(curNeuron.output)
                else:
                    ret.append("None")
                
                # Add error
                if(curNeuron.error != None):
                    ret.append(curNeuron.error)
                else:
                    ret.append("None")
        
        if(writeToFile == True):
            # Write to file
            output = open(filename, "w")
            for line in ret:
                # write line to output file
                output.write(str(line))
                output.write("\n")
            output.close()

        return ret

    ####################################################################
    # This method imports a network from a text file based on the export
    # method above. It will overwrite everything in this neural network.
    # Pass ONE of the following arguments:
    # inputF    == the filename to read the input from
    # inputT    == the array of text that is of equivalent format as defined
    #              in the export_network method
    ####################################################################
    def import_network(self, inputF = None, inputT = None):
        self.networkLayers = []
        self.output = None

        # Read the size and learning rate first
        # Then, put the rest into an array to deal with easily
        inArray = []
        if(inputF != None):
            count = 0
            with open(inputF) as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if (count == 2):
                        # Read the rest of the file into an array
                        inArray.append(line)
                    elif (count == 0):
                        # Read the number of layers
                        self.size = int(line)
                        count += 1
                    elif (count == 1):
                        # Read the learning rate
                        self.learningRate = float(line)
                        count += 1
        else:
            inArray = inputT

            # Read size and learning rate
            self.size = inArray[0]
            self.learningRate = inArray[1]

            # Remove those values from the array
            del inArray[0]
            del inArray[0]

        # Create the network
        count = 0

        # Read each layer
        for _i in range(0, self.size):
            # Read the number of neurons in the layer
            layerSize = int(inArray[count])
            count += 1

            # Make sure it's greater than 0
            if layerSize < 1:
                raise exception.InvalidNeuronCount("The number of neurons must be greater than zero")

            # Read each neuron in the layer
            neurons = []
            for _j in range(0, layerSize):
                numWeights = int(inArray[count])
                count += 1

                # First, read the weights
                weights = None
                if(numWeights > 0):
                    temp = inArray[count].split()
                    weights = []
                    for k in range(0, len(temp)):
                        weights.append(float(temp[k]))
                    count += 1
                
                # Next, read the bias
                bias = None
                if(inArray[count] != "None"):
                    bias = float(inArray[count])
                count += 1

                # Next, read the output
                output = None
                if(inArray[count] != "None"):
                    output = float(inArray[count])
                count += 1

                # Lastly, read the error
                error = None
                if(inArray[count] != "None"):
                    error = float(inArray[count])
                count += 1

                # Create the neuron
                layerNeuron = neuron.main(None if weights == None else weights,
                                          -999999.0 if bias == None else bias)
                
                # Set the output
                if(output != None):
                    layerNeuron.set_output(output)
                
                # Set the error
                if(error != None):
                    layerNeuron.set_error(error)

                # Add to the neuron list
                neurons.append(layerNeuron)

            # Create the layer
            newLayer = layer.main(layerSize, self.learningRate, neurons)

            # Link to previous layer if applicable
            if (len(self.networkLayers) >= 1):
                self.networkLayers[len(self.networkLayers) - 1].set_next_layer(newLayer, False)
            
            # Add to layers
            self.networkLayers.append(newLayer)

    ####################################################################
    # This method sets the input for the network
    ####################################################################
    def set_input(self, input):
        # Check that the number of inputs matches the number
        # of input neurons
        if(len(input) != self.networkLayers[0].count):
            raise exception.InvalidNeuronCount("The number of neurons (" + str(self.networkLayers[0].count) + ") does not match the number of inputs passed (" + str(len(input)) + ")")

        # Set the input
        inputNeurons = self.networkLayers[0].get_neuron_list()
        for i in range(0, self.networkLayers[0].count):
            inputNeurons[i].set_output(input[i])

    ####################################################################
    # This method sets the output to compare the network to
    ####################################################################
    def set_output(self, outputA):
        # Check that the output equals the number of output neurons
        if(len(outputA) != self.networkLayers[self.size - 1].count):
            raise exception.InvalidNeuronCount("The number of neurons (" + str(self.networkLayers[self.size - 1].count) + ") does not match the number of outputs passed (" + str(len(outputA)) + ")")
        
        # Set the output
        self.output = outputA

    ####################################################################
    # This method sets the learning rate for all layers
    ####################################################################
    def set_learning_rate(self, learningR):
        self.learningRate = learningR

        for i in range(0, len(self.networkLayers)):
            self.networkLayers[i].learningRate = self.learningRate

    ####################################################################
    # This method gets the output for the network
    ####################################################################
    def get_network_output(self):
        ret = []

        outputLayer = self.networkLayers[self.size - 1]
        outputNeurons = outputLayer.get_neuron_list()
        for i in range(0, len(outputNeurons)):
            ret.append(outputNeurons[i].get_output())
            
        return ret

    ####################################################################
    # This method sets the error for the output layer
    ####################################################################
    def _set_output_error(self, errors):
        # Set the error for the output neurons
        outputLayer = self.networkLayers[self.size - 1]
        outputNeurons = outputLayer.get_neuron_list()
        
        # Check that the length of errors is correct
        if(len(errors) != len(outputNeurons)):
        	raise exception.InvalidLength("The amount of errors must be equal to the amount of output neurons")
        
        for i in range(len(outputNeurons)):
            outputNeurons[i].error = errors[i]