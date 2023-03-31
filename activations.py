import numpy as np

class Sigmoid():
    @staticmethod
    def function(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))
    
    @staticmethod
    def derivative(z):
        """Derivative of the sigmoid function."""
        return Sigmoid.function(z)*(1-Sigmoid.function(z))
    
class ReLu():
    @staticmethod
    def function(z):
        """ReLU activation function"""
        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        """Derivative of ReLU activation function"""
        return np.where(z <= 0, 0, 1)