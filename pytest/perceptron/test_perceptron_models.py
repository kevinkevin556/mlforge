import numpy as np
import pytest
import sys
sys.path.append("../")

from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from forge.perceptron.models import Perceptron, Adaline
from forge.perceptron.optimizers import Pocket, GradientDescent
from forge.utils.data_utils import add_cons
from forge.utils.operation_utils import sign
from forge.losses import ZeroOneError


def test_perceptron_model():
    X, y = make_classification()
    y[y==0] = -1

    model = Perceptron(optimizer=Pocket())
    model.fit(X, y)
    model.predict(X)
    model.evaluate(X, y)


def test_perceptron_sklearn_cv():
    X, y = make_classification()
    y[y==0] = -1

    model = Perceptron(optimizer=Pocket())
    scores = cross_val_score(model, X, y, cv=5)
    print(scores)


def test_perceptron_sklearn_pipeline():
    X, y = make_classification()
    y[y==0] = -1

    model = Perceptron(optimizer=Pocket())
    estimators = Pipeline([('reduce_dim', PCA(n_components=2)), ('model', model)])
    estimators.fit(X, y)


def test_adaline_model():
    X, y = make_classification()
    y[y==0] = -1

    model = Adaline(optimizer=GradientDescent(lr=0.001))
    model.fit(X, y)
    model.predict(X)
    model.evaluate(X, y)


def test_adaline_sklearn_cv():
    X, y = make_classification()
    y[y==0] = -1

    model = Adaline(optimizer=GradientDescent(lr=0.001))
    scores = cross_val_score(model, X, y, cv=5)
    print(scores)


def test_adaline_sklearn_pipeline():
    X, y = make_classification()
    y[y==0] = -1

    model = Adaline(optimizer=GradientDescent(lr=0.001))
    estimators = Pipeline([('reduce_dim', PCA(n_components=2)), ('model', model)])
    estimators.fit(X, y)