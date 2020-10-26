###################################################
# This file defines all of the exceptions used in 
# the NFQ project
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python

####################################################################
# This is thrown when the size doesn't match the number of neurons
# or is too small for a network
####################################################################
class InvalidLayerCount(Exception):
    pass

####################################################################
# This is thrown when there isn't a next layer and you try to go 
# forward or backward
####################################################################
class NoConnectionException(Exception):
    pass

####################################################################
# This is thrown if the number of neurons specified for a layer
# is less than or equal to zero
####################################################################
class InvalidNeuronCount(Exception):
    pass

####################################################################
# This is thrown if the output of the network has not been set
# before propogating forward
####################################################################
class OutputNotSet(Exception):
    pass

####################################################################
# This is thrown if the size of two arrays are not the same that are
# supposed to be
####################################################################
class NotEqualLength(Exception):
    pass

####################################################################
# This is thrown if the learning rate is not passed to the layer
# class when creating it
####################################################################
class NoLearningRate(Exception):
    pass

####################################################################
# This is thrown if the neural network passed to the dql instance
# is not initialized
####################################################################
class NetworkNotInitialized(Exception):
    pass

####################################################################
# This is thrown if the neural network passed to the dql instance
# is not initialized
####################################################################
class NoFilenameSpecified(Exception):
    pass

####################################################################
# This is thrown if the states or actions passed to the dql class
# are less than 1
####################################################################
class InvalidLength(Exception):
    pass

####################################################################
# This is thrown if an argument that was expected to be passed is
# not passed
####################################################################
class MissingArgument(Exception):
    pass