#------------------------------------------------------------------------------
# TODO LIST:
# 1. GD 
#   1.2 delete trash info
#   1.3 converge metrics?
#   1.4 regularize
#   1.5 momentum 
#	1.6 remove reference calling side effect (ok)
# 2. PLA
# 3. pocket PLAr
# 4. SGD    
# 5. sigmoid
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# exception handling guide
# http://teddy-chen-tw.blogspot.tw/2010/03/blog-post_13.html
#------------------------------------------------------------------------------



import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from . import losses
from .exception import * 
from .data_util import *


def gradient(x_train, y_train, weight, loss, activation):
	x_train = torch.as_tensor(x_train, dtype=torch.double)
	y_train = torch.as_tensor(y_train, dtype=torch.double)
	weight = torch.as_tensor(weight, dtype=torch.double)

	return (loss.first_deriv(activation.evaluate(x_train, weight), y_train) @
		   activation.first_deriv(x_train)).t


class Optimizer():
	
	"""base class of optimizers"""
	
	def __init__(self):
		# initialize parameter
		self.parameter = {}

	def execute(self, data, features, label):
		# implication
		# 
		# input: data 	 >> DataFrame
		#		features >> list[str]
		#		label 	 >> str 
		# 
		# output: weight >> dict{str: float}
         pass 

class BatchGradientDescent(Optimizer):   
    def __init__(self, lr):
        self.parameter = {}
        self.parameter['learning_rate'] = lr

    def execute(self, x_train, y_train, loss):
        learning_rate = self.parameter['learning_rate']
        data_num = x_train.shape[0]
        feature_num = x_train.shape[1] + 1
        x_train = np.hstack((x_train, np.ones((data_num, 1))))
        
        init_guess = [0 for i in range(feature_num)]
        weight = np.array(init_guess)[np.newaxis].T


        is_converged = False
        while not is_converged:
            temp_weight = weight
            gradient = losses.gradient[loss]
            # gradient = np.dot(x_train.T, x_train.dot(weight)-y_train)
            weight = weight - learning_rate / data_num * gradient(x_train, weight, y_train)
            if np.array_equal(weight, temp_weight):
                is_converged = True
            
            if (float('inf') in weight) or (float('-inf') in weight):
                raise DivergenceError(lr = learning_rate)

        print(weight)
        min_loss = losses.loss[loss](y_train, np.dot(x_train, weight))
        print(min_loss)
        return weight


class StocasticGradientDescent(Optimizer):
	def __init__(self, lr):
        self.parameter = {}
        self.parameter['learning_rate'] = lr

    def execute(self, x_train, y_train, loss):
		learning_rate = self.parameter['learning_rate']
		x_train = torch.tensor(x_train, dtype=torch.double)
		y_train = torch.tensor(y_train, dtype=torch.double)
		length = len(y_train)

		x_train = torch.cat([torch.tensor([1]*length), x_train], dim=1)
		weight = torch.rand((length, 1), dtype=torch.double)

		is_converged = False
        while not is_converged:            
            for x, y in zip(x_train, y_train): 
            	temp_weight = torch.weight(weight)
            	weight = weight - learning_rate / data_num * gradient(x_train, y_train, weight, loss, activation)
            	
            	if (float('inf') in weight) or (float('-inf') in weight):
                	raise DivergenceError(lr = learning_rate)
            	if torch.equal(weight, temp_weight):
                	is_converged = True
                	break
        return weight


class MiniBatchGradientDescent(Optimizer):
	def __init__(self, lr):
        self.parameter = {}
        self.parameter['learning_rate'] = lr