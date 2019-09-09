import numpy as np
from .base.metrics import Metric
from .utils.data_utils import set_eval_data

class ZeroOneError(Metric):
	@staticmethod
	def eval(y_fit, y_true):
		y_fit, y_true = set_eval_data((y_fit, y_true))
		error = sum(y_fit != y_true)
		return error

class MeanSquareError(Metric):
	@staticmethod
	def eval(y_fit, y_true):
		y_fit, y_true = set_eval_data((y_fit, y_true))
		n = len(y_true)
		error = sum((y_true - y_fit)**2)/n
		return error

class RSquare(Metric):
	@staticmethod
	def eval(y_fit, y_true):
		y_fit, y_true = set_eval_data((y_fit, y_true))
		n = len(y_true)
		y_bar = sum(y_fit)/n
		total_sum_of_squares = sum((y_true-y_bar)**2)/n
		residual_sum_of_squares = sum((y_true-y_fit)**2)/n
		rsquare = 1 - residual_sum_of_squares/total_sum_of_squares
		return rsquare  		

