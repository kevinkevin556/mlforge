#------------------------------------------------------------------------------
# TODO LIST:
# 1. predict()
# 	1.1 constant handling refactor (refactor predict)
# 	1.2 ValueError: 'untrained' or NaN weight in model
#	1.3 no_enough_parameter exception
#	1.4 copy dataFrame issue
# 2. evaluate()
# 3. __str__()
#
#------------------------------------------------------------------------------

import numpy as np
import pandas as pandas
from collections import OrderedDict
from pandas import DataFrame, Series

from . import formula_parser as fp
from .optimizers import *
from .exception import *
from .losses import * 

class Model(object):
	
	"""This is the base class of all the other models"""
	
	def __init__(self, formula):
		pass

	def add(self):
		pass

	def compile(self, loss, optimizer, metrics):
		pass

	def fit(self, x_train, y_train):
		pass

	def evaluate(self, x_test, y_test):
		pass

	def predict(self, x_test):
		pass
		


class Linear(Model):
	""" docstring """


	### interface implecation ###

	def __init__(self, model = None):
		self.model = fp.parse_formula(model)
		self.optimizer = Optimizer()
		# self.metrics = Metric()

	def compile(self, model = None, loss = 'mean_squared_error', optimizer = GradientDescent(lr = 0.01), metrics = 'mean_squared_error'):
		if not (model == None):
			self.model =  fp.parse_formula(model) 
			
		self.loss = loss
		self.optimizer = optimizer 
		self.metrics = metrics

	def fit(self, x_train = None, y_train = None, data_train = None, using_model = False):
		try:
			self._set_model_for_seperated_data(x_train, y_train, using_model)
			features = list(self.model['features'].keys())
			label = self.model['label']
			(x_train, y_train) = self._set_training_data(x_train, y_train, data_train, features, label)
			
			weight = self.optimizer.execute(x_train = x_train, y_train = y_train, loss = self.loss)
		except Exception as e:
			raise e
		else:
			features = self.model['features'].keys()
			self.model['features'] = OrderedDict({i[0]:i[1] for i in zip(features, weight.T)})

	def predict(self, data, pred_column = ['pred_value']):
		try:
			feature = list(data.columns)
			feature.insert(0, '_cons')
			data['cons'] = 1
		except Exception as e:
			raise e

		# parameter matching exception
		try:
			weight = [self.model['feature'][i] for i in feature]
			weight = np.array(weight)[np.newaxis].T
		except Exception as e:
			raise e	
		else:
			pred_value = DataFrame(data = np.dot(data[feature].values, weight),\
								columns = pred_column )
			return pred_value

	def evaluate(self, test_data):
		pass


	### implict subroutines ###
	def _set_model_for_seperated_data(self, x_train, y_train, using_model):
		if (not using_model) or (self.model == None):
				formula = y_train.name
				for i in x_train.columns.values:
					if i == x_train.columns.values[0]:
						formula = formula + "="
					else:
						formula = formula + "+"
					formula = formula + i
				self.model = fp.parse_formula(formula)

	def _set_training_data(self, x_train = None, y_train = None, data_train = None, features = None, label = None):
		using_data_train = self._check_data_input(x_train, y_train, data_train)
		features.remove('_cons')
		y_train = DataFrame(y_train)

		if using_data_train:
			x_train = data[features].values
			y_train = data[label].values[np.newaxis].T
		else:
			x_train = x_train[features].values
			y_train = y_train[label].values[np.newaxis].T

		return (x_train, y_train)

	def _check_data_input(self, x_train = None, y_train = None, data_train = None):
		if data_train is None:
			if x_train is None or y_train is None:
				raise TrainingDataError(msg = "No enough data input.")
			else:
				return False
		else:
			if not (x_train is None and  y_train is None):
				raise TrainingDataError(msg = "Double data input.")
			else:
				return True