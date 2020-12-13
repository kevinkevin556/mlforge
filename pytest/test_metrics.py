import numpy as np
import pytest
import sys
sys.path.append("./")

from forge.metrics import Accuracy


def test_accracy():
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_true = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    assert Accuracy.eval(y_true, y_pred) == 0.5
