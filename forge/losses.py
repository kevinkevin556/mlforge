import numpy as np
from .base.losses import Loss
from .utils.data_utils import set_eval_data

class ZeroOneError(Loss):
	@staticmethod
	def eval(y_fit, y_true):
		y_fit, y_true = set_eval_data((y_fit, y_true))
		loss = sum(y_fit != y_true)
		return loss
	
	@staticmethod
	def grad(y_fit, y_true):
		raise ValueError("ZeroOneError is not derivativable.")

class MeanSquareError(Loss):
	@staticmethod
	def eval(y_fit, y_true):
		y_fit, y_true = set_eval_data((y_fit, y_true))
		n = len(y_true)
		loss = sum((y_true - y_fit)**2)/n
		return loss

	@staticmethod
	def grad(y_fit, y_true):
		y_fit, y_true = set_eval_data((y_fit, y_true))
		gradient = 2 * (y_true-y_fit)
		return gradient