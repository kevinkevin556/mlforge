import numpy as np
import pytest
import sys
sys.path.append("../")

from sklearn.datasets import make_moons
from forge.ensemble.models import *
from forge.perceptron.optimizers import Pocket
from forge.utils.operation_utils import sign


def test_bagging_pocket():
    x_test, y_test = make_moons(20, noise=0.05, random_state=4, shuffle=False)
    x_test, y_test = np.delete(x_test, [0, 10], axis=0), np.delete(y_test, [0, 10])
    y_test = sign(y_test)
    
    model = BaggingPerceptron(
        n_estimators=15,
        optimizer=Pocket(updates=2000)
    )
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)

    # test predict()
    assert np.array_equal(y_pred, y_test)
    # test evaluate()
    assert model.evaluate(x_test, y_test)[0] == 1.0   # Accuracy == 1.0


def test_adaboost_stump():
    x_test, y_test = make_moons(20, random_state=123)
    y_test = sign(y_test)

    model = AdaBoostStump(n_estimators=10)
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)
    accuracy = model.evaluate(x_test, y_test)[0]

    # test predict()
    assert np.array_equal(y_pred, y_test)
    # test evaluate()
    assert accuracy == 1.0