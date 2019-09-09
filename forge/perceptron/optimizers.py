import random

from ..utils import * 
from ..metrics import ZeroOneError
from ..base.optimizers import Optimizer


class LinearSeparable(Optimizer):
	def __init__(self, lr=1, iterations="until_converged", init="zeros"):
		self.params = {}
		self.params['lr'] = lr
		self.params['iterations'] = iterations
		self.params['init'] = init

	def execute(self, x_train, y_train,
				fn_form = lambda x, w: sign(x.dot(w)),
				loss = ZeroOneError()):
		lr = self.params['lr']
		it = self.params['iterations']
		w = initialize_weight(x_train, y_train, method=self.params['init'])
		x = cons_augment(x_train)
		y = y_train

		iter_off = False
		if it == "until_converged":
			iter_count = np.Inf
		elif type(it) is int:
			if it > 0:
				iter_count = it
			else:
				raise ValueError
		else: 
			raise ValueError
		
		while not iter_off :
			for x_i, y_i in zip(x, y):
				iter_count -= 1
				if fn_form(x_i, w) != y_i:
					w = w + lr * y_i * x_i
				if iter_count == 0:
					iter_off = True
					break
			if  loss.eval(fn_form(x, w), y) == 0:
				iter_off = True
			if iter_off:
				break

		return w

class Pocket():
	def __init__(self, lr=1, iterations=1000, init="zeros"):
		self.params = {}
		self.params['lr'] = lr
		self.params['iterations'] = iterations
		self.params['init'] = init
	
	def execute(self, x_train, y_train,
				fn_form = lambda x, w: sign(x.dot(w)),
				loss = ZeroOneError()):
		lr = self.params['lr']
		it = self.params['iterations']
		w = initialize_weight(x_train, y_train, method=self.params['init'])
		x = cons_augment(x_train)
		y = y_train

		iter_off = False
		iter_count = it

		w_pocket = w
		loss_pocket = loss.eval(fn_form(x, w_pocket), y)
		while not iter_off :
			i = random.choice(range(len(x)))
			x_i, y_i = (x[i,:], y[i])
			iter_count -= 1
			
			if fn_form(x_i, w) != y_i:
				w = w + lr * y_i * x_i
				
			loss_current = loss.eval(fn_form(x, w), y)
			if  loss_current < loss_pocket:
				w_pocket = w
				loss_pocket = loss_current 

			if (iter_count == 0) or (w_pocket == 0):
				iter_off = True
				break

		return w_pocket