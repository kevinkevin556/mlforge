import numpy as np
import random
import sys
import pytest

sys.path.append("./")
import forge.losses as losses

class Test_ZeroOneError:
    def test_Eval_1darrayAsParam_ReturnCorrectError(self):
        # arrange
        ans = 10
        sign = [1, -1]
        n = 10000
        np.random.seed(0)
        test_y_true = np.random.choice(sign, n)
        wrong_index = np.random.randint(low=0, high=n, size=ans)
        test_y_fit = np.array([-test_y_true[i] if i in wrong_index else test_y_true[i] for i in range(n)])

        # act
        output = losses.ZeroOneError.eval(test_y_fit, test_y_true)

        # assert
        assert output == 10

    def test_Eval_ListAsParam_ReturnCorrectError(self):
        # arrange
        ans = 10
        sign = [1, -1]
        n = 10000
        random.seed(0)
        test_y_true = random.choices(sign, k=n)
        wrong_index = random.sample(range(n), ans)
        test_y_fit = [-test_y_true[i] if i in wrong_index else test_y_true[i] for i in range(n)]

        # act
        output = losses.ZeroOneError.eval(test_y_fit, test_y_true)

        # assert
        assert output == 10

    def test_Grad_1darrayAsParam_ReturnException(self):
        # arrange
        sign = [1, -1]
        n = 10000
        k = 10
        np.random.seed(0)
        test_y_true = np.random.choice(sign, n)
        wrong_index = np.random.randint(low=0, high=n, size=k)
        test_y_fit = np.array([-test_y_true[i] if i in wrong_index else test_y_true[i] for i in range(n)])

        # act & assert
        with pytest.raises(Exception):
            assert losses.ZeroOneError.grad(test_y_fit, test_y_true)


    def test_Grad_ListAsParam_ReturnException(self):
        # arrange
        sign = [1, -1]
        n = 10000
        k = 10
        random.seed(0)
        test_y_true = random.choices(sign, k=n)
        wrong_index = random.sample(range(n), k)
        test_y_fit = [-test_y_true[i] if i in wrong_index else test_y_true[i] for i in range(n)]   

        # act & assert
        with pytest.raises(Exception):
            assert losses.ZeroOneError.grad(test_y_fit, test_y_true)


class Test_MeanSquareError:
    def test_Eval_1darrayAsParam_ReturnCorrectError(self):
        # arrange
        n = 10000
        k = 100
        delta = 10
        random.seed(0)
        np.random.seed(0)
        different_index = random.sample(range(n), k)
        test_y_fit = np.random.rand(n)
        test_y_true = np.array([test_y_fit[i]+delta if i in different_index else test_y_fit[i] for i in range(n)] )
        ans = (delta**2)*k/n 

        # act 
        output = losses.MeanSquareError.eval(test_y_fit, test_y_true)

        # assert
        assert output == ans 
    
    def test_Eval_ListAsParam_ReturnCorrectError(self):
        # arrange
        n = 10000
        k = 100
        delta = 10
        random.seed(0)
        different_index = random.sample(range(n), k)
        test_y_fit = [random.random() for i in range(n)]
        test_y_true = [test_y_fit[i]+delta if i in different_index else test_y_fit[i] for i in range(n)]
        ans = (delta**2)*k/n 

        # act 
        output = losses.MeanSquareError.eval(test_y_fit, test_y_true)

        # assert
        assert output == ans 

    def test_Grad_1darrayAsParam_ReturnCorrectGradient(self):
        # arrange
        n = 10000
        np.random.seed(0)
        test_y_fit = np.random.rand(n)
        test_y_true = np.random.rand(n)
        ans = 2 * (test_y_true - test_y_fit)

        # act 
        output = losses.MeanSquareError.grad(test_y_fit, test_y_true)

        # assert
        assert np.array_equal(output, ans) == True

    def test_Grad_ListAsParam_ReturnCorrectGradient(self):
        # arrange
        n = 10000
        random.seed(0)
        test_y_fit = [random.random() for i in range(n)]
        test_y_true = [random.random() for i in range(n)]
        ans = np.array([2*(i-j) for i, j in zip(test_y_true, test_y_fit)])

        # act
        output = losses.MeanSquareError.grad(test_y_fit, test_y_true)

        # assert 
        assert np.array_equal(output, ans) == True