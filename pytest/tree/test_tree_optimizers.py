import numpy as np
import pytest
import sys
sys.path.append("../..")
from sklearn.datasets import make_moons, make_classification, make_regression

from forge.impurities import GiniIndex, MeanSquaredError, Entropy
from forge.tree.optimizers import DecisionStumpSolver, CART, ID3


@pytest.fixture
def test_decision_stump_data():
    x_test, y_test = make_moons(20, random_state=123)
    y_test[y_test==0] = -1
    return x_test, y_test


@pytest.fixture
def test_decision_tree_class_data():
    x_test, y_test = make_classification(random_state=1)
    y_test[y_test==0] = -1
    return x_test, y_test

@pytest.fixture
def test_nominal_data():
    np.random.seed(1)
    x_test = np.array([np.random.choice(range(i), size=100) for i in range(1,5)]).T
    y_test = np.random.choice([-1, 1], size=100)
    return x_test, y_test


@pytest.fixture
def test_decision_tree_regression_data():
    x_test, y_test = make_regression(random_state=1)
    return x_test, y_test


# helper functions

def decison_stump_brutal_solve(x_test, y_test):
    min_error = len(y_test)

    for i in range(x_test.shape[1]):
        x_i = x_test[:, i]
        idx = np.argsort(x_i)
        y_i = y_test[idx]

        for s in [1, -1]:
            for j in range(len(y_i)+1):
                error = y_i[0:j].tolist().count(s) + y_i[j:].tolist().count(-s)
                if error < min_error:
                    min_error = error
    return min_error

# Tests

def test_decision_stump_solver(test_decision_stump_data):
    x_test, y_test = test_decision_stump_data
    error_test = decison_stump_brutal_solve(x_test, y_test)
    
    solver = DecisionStumpSolver()
    sign, dim, threshold = solver.execute(x_test, y_test)
    
    result = 1 * ((x_test[:, dim] - threshold) > 0)
    result[result == 0] = -1
    result = sign * result
    error = np.sum(result != y_test)
    assert error == error_test


def test_CART_class(test_decision_tree_class_data):
    x_test, y_test = test_decision_tree_class_data
    solver = CART(criterion=GiniIndex())
    root = solver.execute(x_test, y_test)


def test_CART_regression(test_decision_tree_regression_data):
    x_test, y_test = test_decision_tree_regression_data
    solver = CART(criterion=MeanSquaredError())
    root = solver.execute(x_test, y_test)


def test_ID3(test_nominal_data):
    x_test, y_test = test_nominal_data
    solver = ID3()
    root = solver.execute(x_test, y_test)

    solver = ID3(criterion=GiniIndex())
    root = solver.execute(x_test, y_test)