#### Define the quadratic and cross-entropy cost functions
import numpy as np
from activations import Sigmoid

class QuadraticCost(object):

    @staticmethod
    def function(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * Sigmoid.derivative(z)


class CrossEntropyCost(object):

    @staticmethod
    def function(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a - y)

class CategoricalCrossEntropyCost(object):
    @staticmethod
    def function(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``. The Categorical Cross-Entropy is a modified version of the Cross-Entropy 
        cost function thinked for Categorical features.  While it is possible to use the 
        Categorical Cross-Entropy function for non-categorical data, it may not be the best choice. 
        It is important to choose a cost function that is appropriate for the specific problem you are trying to solve.

        """
        # apply Softmax function to output a for each example in the mini-batch
        exp_a = np.exp(a - np.max(a))  # subtract max to avoid numerical instability
        softmax_a = exp_a / np.sum(exp_a)

        # calculate Categorical Cross-Entropy loss for mini-batch
        cost = -1/len(a) * np.sum(y * np.log(softmax_a))

        return cost

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a - y)
 