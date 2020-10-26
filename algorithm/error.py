####################################################################
# This class is used for error functions in my neural network
# code
#
# Author: Andrew Fisher
####################################################################

import exception

####################################################################
# This implements a mean-squared-error error function
# TODO: Add support for more types of error functions
####################################################################
def mse(correct, output):
    # Check that the lengths are the same
    if (len(correct) != len(output)):
        raise exception.NotEqualLength("The size of the correct array (" + str(len(correct)) + ") is not equal to the size of the output (" + str(len(output)) + ")")

    # Calculate the error
    ret = 0
    for i in range(0, len(correct)):
        ret += (correct[i] - output[i]) ** 2
    ret /= len(correct)

    return ret