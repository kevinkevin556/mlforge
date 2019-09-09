import numpy as np
from ..utils.data_utils import set_eval_data

class Loss(object):
	"""
	The loss function is used to optimize your model. The function will get minimized by the optimizer.
	
	set_eval_data() is to coerce input data into proper form for calculation,
	User does not need to rewrite the function unless you request for inputs other than 1d-array.

	Overwrite eval() for evaluating loss.
	Overwrite eval_grad() for evaluating gradient for certain loss. 
	"""
	def __init__(self):
		pass 

	@staticmethod
	def eval(y_fit, y_true):
		if (w is not None) and (x is not None):
			y_fit = w.dot(x)
		y_fit, y_true = set_eval_data((y_fit, y_true))
		loss = None
		return loss 

	@staticmethod
	def grad(y_fit, y_true):
		y_fit, y_true = set_eval_data((y_fit, y_true))
		gradient = None
		return gradient