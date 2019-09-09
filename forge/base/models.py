class Model(object):
	"""
	Base class for all machine learning models.

	Use __init__() to intialize model.
	Use compile() to set loss, optimizer and metric for machine learning model.
	Call fit() to train the model using training data,
		 evaluate() to view the performance on testing data,
		 predict() to produce forecasting value with trained model.
	"""

	def __init__(self):
		self.weight = None
		self.activation = lambda x, w: None

	def compile(self, loss, optimizer, metrics):
		self.loss = loss
		self.optimizer = optimizer 
		self.metrics = metrics

	def fit(self, x_train, y_train):
		try:
			self.weight = self.optimizer.execute(x_train=x_train,
												 y_train=y_train,
												 loss=self.loss)
		except Exception as e:
			raise e
		else:
			pass

	def evaluate(self, x_test, y_test):
		res = []
		for m in self.metrics:
			res.append(m.eval(self.functional_form(x_test, self.weight), y_test))
		return res

	def predict(self, x_test):
		pass
		