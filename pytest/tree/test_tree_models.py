import numpy as np
import pytest
import sys
sys.path.append("../")
from sklearn.datasets import make_classification

from mlforge.tree.models import *
from mlforge.utils.operation_utils import sign


# fixtures
@pytest.fixture
def test_data():
    x_test, y_test = make_classification(random_state=1)
    y_test = sign(y_test, zero=-1)
    return x_test, y_test


# testing functions
def test_decision_stump():
    y_test = np.hstack((np.ones(50), -np.ones(50)))
    x_test = np.hstack((np.arange(100).reshape(-1, 1), np.random.rand(100, 4)))

    model = DecisionStump()
    model.fit(x_test, y_test)

    # test fit()
    assert model.sign_ == -1
    assert model.feature_ == 0
    assert (model.threshold_ > 49) and (model.threshold_ < 50)
    # test predict()
    assert np.array_equal(model.predict(x_test), y_test)
    # test evaluate()
    assert model.evaluate(x_test, y_test)[0] == 1.0   # Accuracy == 1.0



def test_decision_tree(test_data):
    x_test, y_test = test_data

    model = DecisionTree(max_height=np.inf)
    model.fit(x_test, y_test)

    # test predict()
    assert np.array_equal(model.predict(x_test), y_test)
    # test evaluate()
    assert model.evaluate(x_test, y_test)[0] == 1.0   # Accuracy == 1.0



def test_random_forest(test_data):
    x_test, y_test = test_data

    model = RandomForest(n_estimators=15)
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)

    # test predict()
    assert np.array_equal(y_pred, y_test)
    # test evaluate()
    assert model.evaluate(x_test, y_test)[0] == 1.0   # Accuracy == 1.0



def test_gradient_boosted_decision_tree(test_data):
    x_test, y_test = test_data

    model = GradientBoostedDecisionTree(n_estimators=10)
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)
    mse = model.evaluate(x_test, y_test)

    # test evaluate()
    assert np.allclose(mse, 0)   # MSE == 0