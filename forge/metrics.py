import abc
import numpy as np

from .utils.data_utils import to_vector


# data setter for metric evaluation
def set_data(y_true, y_pred):
    return (to_vector(y_true), to_vector(y_pred))


# Metric Base Class 
class Metric(metaclass = abc.ABCMeta):
    """
    A metric is used to judge the performance of your model.
    It has nothing to do with model training. 

    Overwrite eval() to implement different metrics.
    """
    def __init__(self):
        pass
    
    @abc.abstractstaticmethod
    def eval(y_true, y_pred):
        y_true, y_pred = set_data(y_true, y_pred)

        error = None # Modify the code to evaluate error
        return error
    
    def __call__(self, y_true, y_pred):
        return self.eval(y_true, y_pred)
    
    def __repr__(self):
        return '"%s"'%self.__class__.__name__



##### Metrics Implemetations #####

class Accuracy(Metric):
    @staticmethod
    def eval(y_true, y_pred):
        y_true, y_pred = set_data(y_true, y_pred)
        
        n = len(y_true)
        error = sum(y_pred == y_true)/n
        return error

class MeanSquaredError(Metric):
    @staticmethod
    def eval(y_true, y_pred):
        y_true, y_pred = set_data(y_true, y_pred)
        
        n = len(y_true)
        error = sum((y_true - y_pred)**2)/n
        return error

class R2(Metric):
    @staticmethod
    def eval(y_true, y_pred):
        y_true, y_pred = set_data(y_true, y_pred)

        n = len(y_true)
        y_bar = sum(y_pred)/n
        total_sum_of_squares = sum((y_true-y_bar)**2)/n
        residual_sum_of_squares = sum((y_true-y_pred)**2)/n
        r2 = 1 - residual_sum_of_squares/total_sum_of_squares
        return r2