import numpy as np
import pytest
import sys
sys.path.append("../")

from mlforge.regression.optimizers import AnalyticSolution, GradientDescent, StocasticGradientDescent
from mlforge.utils.data_utils import add_cons
from mlforge.losses import MeanSquaredError, ZeroOneError, CrossEntropyError
from mlforge.regularizers import L2

# fixture

@pytest.fixture
def test_data():
    np.random.seed(1)
    x_test = np.random.rand(100, 4)
    w_correct = np.random.rand(5)
    y_test = add_cons(x_test) @ w_correct
    return x_test, y_test, w_correct


# testing functions

def test_analytic_solution_correctness(test_data):
    x_test, y_test, w_correct = test_data

    w_ans = AnalyticSolution().execute(x_test, y_test)
    print(AnalyticSolution())
    assert np.allclose(w_ans, w_correct) 


def test_sgd_convergence(test_data):
    x_test, y_test, w_correct = test_data

    training_error = []
    optimizer = StocasticGradientDescent(lr=0.01)

    for i in range(5):
        w_ans = optimizer.execute(x_test, y_test, loss=MeanSquaredError(), epochs=i)
        training_error.append(MeanSquaredError().eval(w_ans, add_cons(x_test), y_test))
    
    for i in range(4):
        assert training_error[i] > training_error[i+1]


def test_sgd_correctness(test_data):
    x_test, y_test, w_correct = test_data

    w_ans = StocasticGradientDescent(lr=0.01).execute(x_test, y_test, loss=MeanSquaredError(), epochs=200)
    print(StocasticGradientDescent(lr=0.01))
    assert np.allclose(w_ans, w_correct)


def test_gradient_descent_correctness(test_data):
    x_test, y_test, w_correct = test_data

    w_ans = GradientDescent(lr=0.0005).execute(x_test, y_test, loss=MeanSquaredError())
    print(GradientDescent(lr=0.0005))
    assert np.allclose(w_ans, w_correct)