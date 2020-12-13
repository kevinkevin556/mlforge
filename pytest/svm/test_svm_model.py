import numpy as np
import pytest
import sys
sys.path.append("../..")
from sklearn.svm import SVC
from sklearn.datasets import make_blobs


from forge.svm.models import HardMarginSVM, SoftMarginSVM, ProbabilisticSVM
from forge.kernels import Linear, Gaussian

# fixture

@pytest.fixture
def test_data():
    np.random.seed(123)
    x_train = np.random.randint(-10, 10, (100, 4))
    y_train = np.random.choice([-1, 1], 100)
    return x_train, y_train

@pytest.fixture
def test_linear_sep_data():
    x_train, y_train = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=0.5, random_state=0)
    y_train[y_train==0] = -1
    return x_train, y_train


# testing functions

def test_hard_margin_svm_model(test_linear_sep_data):
    x_train, y_train = test_linear_sep_data

    model = HardMarginSVM()
    model.fit(x_train, y_train)
    benchmark = SVC(kernel='linear')
    benchmark.fit(x_train, y_train)

    model_score = model.evaluate(x_train, y_train)[0]
    model_result = model.predict(x_train)
    benchmark_score = benchmark.score(x_train, y_train)
    benchmark_result = benchmark.predict(x_train)
    assert model_score == benchmark_score
    assert np.allclose(model_result, benchmark_result, atol=1e-3)


def test_soft_margin_svm_model(test_data):
    x_train, y_train = test_data

    model = SoftMarginSVM(kernel=Gaussian())
    model.fit(x_train, y_train)
    benchmark = SVC(kernel='rbf')
    benchmark.fit(x_train, y_train)

    model_score = model.evaluate(x_train, y_train)[0]
    model_result = model.predict(x_train)
    benchmark_score = benchmark.score(x_train, y_train)
    benchmark_result = benchmark.predict(x_train)
    assert model_score == benchmark_score
    assert np.allclose(model_result, benchmark_result, atol=1e-3)


def test_probalistic_svm_model(test_data):
    x_train, y_train = test_data

    model = ProbabilisticSVM(kernel=Gaussian())
    model.fit(x_train, y_train)
    model_result = model.predict(x_train)