import numpy as np
from ..utils.data_utils import set_eval_data

class Activation(object):
    def __init__(self):
        pass
        
    @staticmethod
    def eval(x, w):
        x, w = set_eval_data((x, w))
        output = lambda x,w: None
        return output

    @staticmethod
    def grad(x, w):
        x, w = set_eval_data((x, w))
        gradient = None 
        return gradient

