import numpy as np
from math import exp

from .base.activations import Activation
from .utils.data_utils import set_eval_data

class Linear(Activation):
    @staticmethod
    def eval(x, w): 
        x, w = set_eval_data((x, w))
        output = x.dot(w)
        return output
    
    @staticmethod
    def grad(x, w):
        x, w = set_eval_data((x, w))
        gradient = -x
        return gradient

class Sigmoid(Activation):
    @staticmethod
    def eval(x, w):
        x, w = set_eval_data((x, w))
        output = (1 + exp(-x@w))**(-1)
        return output
   
    @staticmethod
    def grad(x, w):
        x, w = set_eval_data((x, w))
        gradient = exp(-x@w)
        return gradient