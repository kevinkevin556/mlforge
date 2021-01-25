import numpy as np
import sys
sys.path.append("../..")

from mlforge.utils.data_utils import *

def test_add_cons():
    # arrange 
    np.random.seed(0)
    n_row = np.random.randint(0, 1000)
    n_col = np.random.randint(0, 100)
    x_test = np.random.rand(n_row, n_col)

    # act 
    x_modified = add_cons(x_test)

    # assert 
    assert np.unique(x_modified[:,0]) == 1
    assert x_modified.shape == (n_row, n_col+1)
    assert np.array_equal(x_modified[:, 1:], x_test)


def test_re_weight_data_method_copying():
    n_row = 10
    n_col = 5
    x_test = np.arange(n_row*n_col).reshape(n_row, n_col)
    y_test = np.random.randint(-1, 1, size=n_row)
    w_test = np.random.randint(low=1, high=10, size=n_row)
    
    x_reweighted, y_reweighted = re_weight_data((x_test, y_test), weight=w_test, method="copying")
    assert x_reweighted.shape[0] == sum(w_test)
    assert y_reweighted.shape[0] == sum(w_test)

    accu_index = 0
    for i in range(x_test.shape[0]):
        for _ in range(w_test[i]):
            assert np.array_equal(x_test[i,:], x_reweighted[accu_index,:])
            assert np.array_equal(y_test[i], y_reweighted[accu_index])
            accu_index = accu_index + 1
        

def test_re_weight_data_method_sampling():
    n_row = 10
    n_col = 5
    x_test = np.arange(n_row*n_col).reshape(n_row, n_col)
    y_test = np.random.randint(-1, 1, size=n_row)
    w_test = np.random.randint(low=1, high=10, size=n_row)
    
    x_reweighted, y_reweighted = re_weight_data((x_test, y_test), weight=w_test, method="sampling")
    assert x_reweighted.shape[0] == n_row
    assert y_reweighted.shape[0] == n_row