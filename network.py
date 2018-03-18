"""
Network.py
File containing a simple script which receives a list of values and another list of of weights, 
it creates a set of list of adjusted weights , it runs based on a fixed number of iterations and
learning rate. manipulating these values will produce different set of results
"""
import numpy as np


def adjust_learning(input_vector, weight_vector):
    pass


def dot_product(input_vector, weight_vector):
    """Returns the sum of the scalar product
    
    Arguments:
        input_vector {array} -- Array of input values
        weight_vector {array} -- Array of weights
    """
    return np.dot(input_vector, weight_vector) 


def activation_function(input_vector, weight_vector):
    """ Binary activation function
        Returns 1 if sum > 0, returns 0 otherwise
    
    Arguments:
        input_vector {array} -- Array of input values
        weight_vector {array} -- Array of weigths
    """
    return 1 if dot_product(input_vector, weight_vector) > 0 else 0


