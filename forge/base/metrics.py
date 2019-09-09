import numpy as np
from ..utils.data_utils import set_eval_data

class Metric(object):
	"""
	A metric is used to judge the performance of your model.
	It has nothing to do with model training. 

	set_eval_data() is to coerce input data into proper form for calcualtion,
	User does not need to rewrite the function unless you request for inputs other than 1d-array.

	Overwrite eval() to implement different metrics.
	"""
	def __init__(self):
		pass

	@staticmethod
	def eval(y_fit, y_true):
		y_fit, y_true = set_eval_data(y_fit, y_true) # Unnecessary to rewrite
		error = None # Modify the code to evaluate error
		return error


