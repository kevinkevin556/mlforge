import numpy as np
import random
import sys
import pytest

sys.path.append("./")
import forge.activations as activation

class Test_Linear:
    def test_Eval_InputNdarray_ReturnXdotW(self):
        # arrange
        n_sample = 10000
        n_feature = 23
        test_x = np.random.rand(n_sample, n_feature)
        test_w = np.random.rand(n_feature)
        ans = test_x @ test_w

        # act
        output = activation.Linear.eval(test_x, test_w)

        # assert
        assert np.array_equal(output, ans) == True

    def test_Eval_InputList_ReturnXdotW(self):
        # arrange
        n_sample = 10000
        n_feature = 23
        test_x = np.random.rand(n_sample, n_feature)
        test_w = np.random.rand(n_feature)
        ans = test_x @ test_w
        test_x = test_x.tolist()
        test_w = test_w.tolist()

        # act
        output = activation.Linear.eval(test_x, test_w)

        # assert
        assert np.array_equal(output, ans) == True 

