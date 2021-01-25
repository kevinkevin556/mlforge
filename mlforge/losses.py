import numpy as np
from numba.experimental import jitclass

from .utils.operation_utils import logistic, sign


class Loss:
	def __init__(self):
		pass 

	def eval(self, w, x_fit, y_true):
		""" Compute loss value. 
		
		Make sure you use the correct input type since
		the function is numba-compiled.

		Parameters
		----------
		w: 1d-array

		x_fit: 2d-array

		y_true: 1d-array
	
		Returns
		-------
			loss: float
		"""
		
		loss = None
		return loss 


	def grad(self, w, x_fit, y_true):
		""" Compute gradient.

		Make sure you use the correct input type since
		the function is numba-compiled.

		Parameters
		----------
		w: 1d-array

		x_fit: 2d-array

		y_true: 1d-array
	
		Returns
		-------
		grad: 1d-array
		"""
		
		gradient = None
		return gradient



##### Losses Implementation #####

@jitclass([])
class ZeroOneError(Loss):
	def eval(self, w, x_fit, y_true):
		n = len(y_true)
		loss = np.sum(sign(x_fit@w) != y_true) / n
		return loss
	

	def grad(self, w, x_fit, y_true):
		raise ValueError("0/1 Error is not derivativable.")



@jitclass([])
class MeanSquaredError(Loss):
	def eval(self, w, x_fit, y_true):
		n = len(y_true)
		loss = (1/n) * np.sum((y_true - (x_fit @ w))**2)
		return loss


	def grad(self, w, x_fit, y_true):
		gradient = 2 * (y_true - (x_fit @ w)) @ (-x_fit)
		return gradient



@jitclass([])
class CrossEntropyError(Loss):
	def eval(self, w, x_fit, y_true):
		n = len(y_true)
		loss = (1/n) * np.sum(np.log(1 + np.exp(-y_true * (x_fit @ w))))
		return loss


	def grad(self, w, x_fit, y_true):
		n = len(y_true)
		gradient = (1/n) * logistic(-y_true * (x_fit @ w)) @ (-np.diag(y_true) @ x_fit)
		return gradient