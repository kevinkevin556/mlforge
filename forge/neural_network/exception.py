class BaseError(Exception):
	def __init__(self):
		pass
	def __str__(self):
		pass

class ModelError(BaseError):
	def __init__(self):
		pass
	def __str__(self):
		pass

class OptimizerError(BaseError):
	def __init__(self):
		pass
	def __str__(self):
		pass

class DivergenceError(OptimizerError):
	def __init__(self, lr):
		self.message =  "Gradient descent diverges with lr == {}.\n".format(lr)
		self.message +=	"Suggest trying a smaller learning rate. "

	def __str__(self):
		return self.message

class TrainingDataError(OptimizerError):
	def __init__(self, msg):
		self.message = msg

	def __str__(self):
		return self.message