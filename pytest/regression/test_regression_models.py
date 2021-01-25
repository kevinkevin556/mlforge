import numpy as np
import pytest
import sys
sys.path.append("../..")
import sklearn.linear_model as skl


from mlforge.regression.models import LinearRegression, LogisticRegression, RidgeRegression
from mlforge.regression.optimizers import GradientDescent, StocasticGradientDescent
from mlforge.regularizers import L2

# fixture

@pytest.fixture
def test_data():
    np.random.seed(123)
    x_train = np.random.randint(-10, 10, (100, 4))
    y_train = np.random.choice([-1, 1], 100)
    return x_train, y_train


# testing functions

def test_log_reg_model(test_data):
    x_train, y_train = test_data

    model = LogisticRegression(optimizer=GradientDescent(lr=0.05))
    model.fit(x_train, y_train)
    clf = skl.LogisticRegression(solver="lbfgs", penalty="none")
    clf.fit(x_train, y_train)

    model_score = model.evaluate(x_train, y_train)[0]
    model_result = model.weight_
    clf_score = clf.score(x_train, y_train)
    clf_result  = np.hstack((clf.intercept_[:, None], clf.coef_))
    assert model_score == clf_score
    assert np.allclose(model_result, clf_result, atol=1e-3)


def test_log_reg_model_with_regularizer(test_data):
    x_train, y_train= test_data

    model = LogisticRegression(optimizer=GradientDescent(lr=0.05), regularizer=L2())
    model.fit(x_train, y_train)
    clf = skl.LogisticRegression(solver="lbfgs", penalty="l2")
    clf.fit(x_train, y_train)
 
    model_score = model.evaluate(x_train, y_train)[0]
    model_result = model.weight_
    clf_score = clf.score(x_train, y_train)
    clf_result  = np.hstack((clf.intercept_[:, None], clf.coef_))
    assert model_score == clf_score
    assert np.allclose(model_result, clf_result, atol=1e-3)