import numpy as np
import random
import pytest
import sys
sys.path.append("../")

from forge.perceptron.optimizers import LinearSeparable, Pocket, GradientDescent
from forge.utils.data_utils import add_cons
from forge.utils.operation_utils import sign
from forge.losses import ZeroOneError

# Fixtures

@pytest.fixture
def linear_separable_data():
    """generate linear Seperable data"""
    np.random.seed(123)
    x_test = np.random.rand(30, 2)
    w_test = np.random.rand(2)
    b_test = np.random.rand(1)
    y_test = sign(x_test @ w_test - b_test)
    return x_test, y_test

@pytest.fixture
def non_linear_separable_data():
    """generate non-separable data"""
    np.random.seed(123)
    x_test = np.random.rand(40, 2)
    w_test = np.random.rand(2)
    b_test = np.random.rand(1)
    margin = np.abs((x_test@w_test-b_test)/np.sqrt(np.sum(w_test**2)+b_test**2)) > 0.03
    x_test = x_test[margin, :]
    y_test = sign(x_test @ w_test - b_test)
    wrong_idx = np.random.randint(0, len(x_test), 1) # One mistake
    y_test[wrong_idx] = -1 * y_test[wrong_idx]
    return x_test, y_test, wrong_idx

# Tests

def test_linear_seperable_correctness(linear_separable_data):
    x_test, y_test = linear_separable_data

    w_ans = LinearSeparable().execute(x_test, y_test)
    y_pred = sign(add_cons(x_test) @ w_ans)
    assert np.array_equal(y_pred, y_test) 


def test_pocket_correctness_linsep_data(linear_separable_data):
    x_test, y_test = linear_separable_data

    w_ans = Pocket().execute(x_test, y_test, updates=np.Inf)
    y_pred = sign(add_cons(x_test) @ w_ans)
    assert np.array_equal(y_pred, y_test) 


def test_gradient_descent_correctness_linsep_data(linear_separable_data):
    x_test, y_test = linear_separable_data

    w_ans = GradientDescent(lr=0.01).execute(x_test, y_test, epochs=np.Inf)
    y_pred = sign(add_cons(x_test) @ w_ans)
    assert np.array_equal(y_pred, y_test) 


def test_pocket_correctness_nonsep_data(non_linear_separable_data):
    x_test, y_test, wrong_idx = non_linear_separable_data

    w_ans = Pocket().execute(x_test, y_test, updates=100)
    y_pred = sign(add_cons(x_test) @ w_ans)
    
    assert len(np.where(y_test!=y_pred)[0].tolist()) == 1
    assert wrong_idx[0] == np.where(y_test!=y_pred)[0][0]


def test_gradient_descent_correctness_nonsep_data(non_linear_separable_data):
    x_test, y_test, wrong_idx = non_linear_separable_data

    w_ans = GradientDescent(lr=0.01).execute(x_test, y_test, epochs=10)
    y_pred = sign(add_cons(x_test) @ w_ans)

    assert len(np.where(y_test!=y_pred)[0].tolist()) == 1
    assert wrong_idx[0] == np.where(y_test!=y_pred)[0][0]
