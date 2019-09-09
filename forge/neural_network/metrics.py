class Metric(object):
	def __init__(self):
		pass

	def __call__(self, x_train, weight, y_train):
		return self.evaluate(cls, x_train, weight, y_train)

	@classmethod
	def evaluate(cls, x_train, weight, y_train):
		pass

	@classmethod
	def evaluate_gradient(cls, x_train, weight, y_train):
		pass

class ZeroOneError(Metric):
	def evaluate()

class MeanSquareError(Metric):
	def __init__(self):
		pass

	@classmethod
	def evaluate_gradient(cls, x_train, weight, y_train):
		return sum(np.squeeze(y_true-y_pred)**2) / len(np.squeeze(y_true-y_pred))


