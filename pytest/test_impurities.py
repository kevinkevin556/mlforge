import numpy as np
import pytest
import sys
sys.path.append("./")

from forge.impurities import *


# fixtures

@pytest.fixture
def class_data():
    np.random.seed(1)
    y_test = ((np.random.random(20)-0.5) > 0).astype(int)
    y_test[y_test==0] = -1
    return y_test


@pytest.fixture
def reg_data():
    np.random.seed(1)
    y_test = np.random.random(20)
    return y_test


# helper functions

def iterative_solver(y_test, criterion):
    min_impurity = np.inf
    position = 0
    for i in range(1, len(y_test)-1):
        y_front = y_test[0:i]
        y_back = y_test[i:]
        mse_front = criterion.eval(y_front)
        mse_back = criterion.eval(y_back)
        temp_impurity = i * mse_front + (len(y_test)-i) * mse_back 
        if temp_impurity < min_impurity:
            min_impurity = temp_impurity
            position = i
    return position, min_impurity


def eval_impurity(y_test, criterion, position):
    n_left = position
    n_right = len(y_test) - n_left
    impurity = n_left * criterion.eval(y_test[0:n_left]) + \
               n_right * criterion.eval(y_test[n_left:])
    return impurity


# test

def test_mean_squared_error(reg_data):
    y_test = reg_data
    criterion = MeanSquaredError()
    
    split_pos, split_impur = criterion.find_split(y_test)
    impur_test = eval_impurity(y_test, criterion, split_pos)
    _, min_impur = iterative_solver(y_test, criterion)


    assert np.allclose(split_impur, impur_test)
    assert np.allclose(split_impur, min_impur)


def test_gini_index(class_data):
    y_test = class_data
    criterion = GiniIndex()
    
    split_pos, split_impur = criterion.find_split(y_test)
    impur_test = eval_impurity(y_test, criterion, split_pos)
    _, min_impur = iterative_solver(y_test, criterion)


    assert np.allclose(split_impur, impur_test)
    assert np.allclose(split_impur, min_impur)