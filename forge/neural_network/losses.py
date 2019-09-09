import torch

class Loss(object):
	
	def __init__(self):
		pass 

	@classmethod
	def evaluate(self, y_fit, y_true):
		""" Evaluate loss with given function.

		Arguments:
			y_fit (matrix): fitting values				(dim: n_data * 1)
			y_true (column vec): training true values 	(dim: n_data * 1)  

		Notations:
			L: loss function
			n_data: number of data
			n_feature: number of invloving features
		
		Returns:
			(scalar): L(y_fit, y_true).t * L(y_fit, y_true) 
				(dim: 1 * 1)
		"""
		pass 

	@classmethod
	def first_deriv(cls, y_fit, y_true):
		""" Get first derivative of loss function.

		Arguments:
			y_fit (matrix): fitting values				(dim: n_data * 1)
			y_true (column vec): training true values 	(dim: n_data * 1)  

		Notations:
			L: loss function
			A: activation function
			n_data: number of data
			n_feature: number of invloving features (including constant)
		
		Returns:
			(row vector): [dL/dA].t 
						= [(dL/dA)_1, (dL/dA)_2, ..., (dL/dA)_n] 
				(dim: 1 * n_features)
		"""
		pass


class MeanSquareError(Loss):

	@classmethod
	def evaluate(self, y_true, y_pred):
		y_true = torch.as_tensor(y_true, dtype=torch.double)
		y_pred = torch.as_tensor(y_pred, dtype=torch.double)
		return torch.mean((y_true-y_pred)**2)


	@classmethod
	def first_deriv(cls, y_fit, y_true):
		y_fit = torch.as_tensor(y_fit, dtype=torch.double)
		y_true = torch.as_tensor(y_true, dtype=torch.double)
		return 2 * (y_fit-y_true).t


	


class CategoricalCrossEntropy(Loss):
	"""
	Not implemented yet.
	"""





		
	


