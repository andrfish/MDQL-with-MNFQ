####################################################################
# This class is used for activation functions in my neural network
# code
#
# Author: Andrew Fisher
####################################################################

import math

####################################################################
# This implements a sigmoid activation function
# TODO: Add support for more types of activation functions
####################################################################
def sigmoid(x):
    try:
        y = 1 / (1 + math.exp(-x))
    except: # X is too large
        if(x > 0):
            y = 1
        else:
            y = 0

    return y * (1 - y)