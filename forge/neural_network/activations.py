class Activaiton(object):
	"""docstring for Activaiton"""
	def __init__(self, arg):
		super(Activaiton, self).__init__()
	
	@classmethod
	def evaluate(cls, x_train, weight):
		""" Evaluate matrix with given activation function.

		Arguments:
			x_train (matrix): training data				(dim: n_data * n_feature)
			weight (column vec): weight for features 	(dim: n_feature * 1)  

		Notations:
			A: activation function
			x: x_train
			w: weight vector
			n_data: number of data
			n_feature: number of invloving features
		
		Returns:
			(scalar): [[A(x_0*w_0)],
					   [A(x_1*w_1)],
					    ...,
					   [A(x_n*w_n)]]	(dim: n_data * 1)
		""" 
		pass

	@classmethod
	def first_deriv(cls, x_train, weight):
		""" Get first derivative of activation function.

		Arguments:
			x_train (matrix): training features 
				(dim: n_data * n_features)

		Notations:
			A: activation function
			w: weight vector
			n_data: number of data
			n_feature: number of invloving features (including constant)
		
		Returns:
			(matrix):[dA/d(w_0), dA/d(w_1), ... , dA/d(w_n)]
				(dim: n_data * n_feature)
		
		"""
		pass
		

class Linear(Activaiton):
	@classmethod
	def evaluate(cls, x_train, weight):
		x_train = torch.as_tensor(x_train, dtype=torch.double)
		weight = torch.as_tensor(weight, dtype=torch.double)
		return torch.mul(x_train, weight)

	@classmethod
	def first_deriv(cls, x_train):
		return torch.as_tensor(x_train)

class Sigmoid(Activaiton):
	def __init__(self):
		pass

class Softmax(Activaiton):
	def __init__(self):
		pass