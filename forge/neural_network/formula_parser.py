#------------------------------------------------------------------------------
# This is a module for the LinearModel class or other classes to deal with 
# string form of the model. It returns a dictionary.
#
# TODO LIST:
# 1. exception handling
# 2. interaction terms
# 3. power of terms
# 4. functional form
#------------------------------------------------------------------------------

from collections import OrderedDict

def parse_formula(formula):
	if not formula == None:	
		model = OrderedDict({})
		formula = delete_whitespace(formula)
		
		if formula_valid(formula):
			(label, features) = parse_field(formula)
			model['label'] = label
			model['features'] = OrderedDict({i:'untrained' for i in features})
			model['features']['_cons'] = 'untrained'
		
		return model
	else: 
		return None


def delete_whitespace(formula):
	return formula.replace(' ', '')

def structure_valid(formula):
	one_equal_flag = False
	no_prior_plus_to_equal_flag = False

	if formula.count('=') == 1:
		one_equal_flag = True
	
	if not ('+' in formula):
		no_prior_plus_to_equal_flag = True
	elif formula.find('=') < formula.find('+'):
		no_prior_plus_to_equal_flag = True
	else:
		no_prior_plus_to_equal_flag = False

	return (one_equal_flag and no_prior_plus_to_equal_flag)


def formula_valid(formula):
	formula_valid_flag = False

	if structure_valid(formula):
		if label_field_valid(formula) & feature_field_valid(formula):
			formula_valid_flag = True

	return formula_valid_flag
		

def label_field_valid(formula):
	label_valid_flag = False

	if not (formula.find('=') == 0):
		label_valid_flag = True

	return label_valid_flag

def feature_field_valid(formula):
	feature_valid_flag = False

	# currently no restriction
	feature_valid_flag = True
	
	return feature_valid_flag

def parse_field(formula):
	(label_field, feature_field) = tuple(formula.split('='))
	feature_field = feature_field.split('+')
	
	return (label_field, feature_field)
