import numpy as np
import sys

sys.path.append("../../")
import forge.regression.optimizers as optimizers
import forge.losses as loss
import forge.activations as activaction

def test_GD_InputNdarray_ReturnCorrectWeight():
    # arrange
    test_x = np.array([[1,2], [3,4], [5,6]])
    test_y = np.array([6, 12, 18])
    ls = loss.MeanSquareError()
    activ = activaction.Linear()
    opt = optimizers.GradientDescent(learning_rate=0.01)
    ans = np.array([1, 1, 2])

    # act
    weight = opt.execute(test_x, test_y, ls, activ)

    # assert
    assert np.allclose(weight, ans, rtol=0, atol=1e-08) # GD may not converge to closed form solution

def test_SGD_InputNdarray_ReturnCorrectWeight():
    # arrange
    test_x = np.array([[1,2], [3,4], [5,6]])
    test_y = np.array([6, 12, 18])
    ls = loss.MeanSquareError()
    activ = activaction.Linear()
    opt = optimizers.StochasticGradientDescent(learning_rate=0.01)
    ans = np.array([1, 1, 2])

    # act
    weight = opt.execute(test_x, test_y, ls, activ)

    # assert
    assert np.allclose(weight, ans, rtol=0, atol=1e-08)


def test_Adagrad_InputNdarray_ReturnCorrectWeight():
    # arrange
    np.random.seed(1509) # This seed makes adagrad trained better
    test_x = np.array([[1,2], [3,4], [5,6]])
    test_y = np.array([6, 12, 18])
    ls = loss.MeanSquareError()
    activ = activaction.Linear()
    opt = optimizers.Adagrad(learning_rate=0.04)
    ans = np.array([1, 1, 2])

    # act
    weight = opt.execute(test_x, test_y, ls, activ)

    # assert
    assert np.allclose(weight, ans, rtol=0, atol=0.1) # Adagrad stops so early.




