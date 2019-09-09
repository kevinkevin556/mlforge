import numpy as np
import random
import sys
sys.path.append("../../")
import pytest

import forge.utils.data_utils as dt_utils



def test_AddBiasTerm_Input2darray_Return2darray():
    # arrange 
    np.random.seed(0)
    test_x = np.random.rand(3, 4)
    ans = np.hstack((np.ones((3,1)), test_x))

    # act 
    modified_x = dt_utils.add_bias_term(test_x)

    # assert 
    assert np.array_equal(ans, modified_x)

def test_AddBiasTerm_Input1darray_Return2darray():
    # arrange 
    np.random.seed(0)
    test_x = np.random.rand(3)
    ans = np.hstack((np.ones((3,1)), test_x[np.newaxis].T))

    # act 
    modified_x = dt_utils.add_bias_term(test_x)

    # assert 
    assert np.array_equal(ans, modified_x)

def test_AddBiasTerm_Input3darray_RaiseException():
    # arrange 
    np.random.seed(0)
    test_x = np.random.rand(3, 4, 4)
    
    # act & assert
    with pytest.raises(Exception) as e_info:
        output = dt_utils.add_bias_term(test_x)



def test_Unidimensionalize_InputList_Return1darray():
    # arrange 
    random.seed(0)
    test_x = [random.random()for i in range(10)]
    ans = np.array(test_x)

    # act
    output = dt_utils.unidimensionalize(test_x)

    # assert 
    assert np.array_equal(ans, output)

def test_Unidimensionalize_Input1darray_Return1darray():
    # arrange
    np.random.seed(0)
    test_x = np.random.rand(10)
    ans = test_x

    # act 
    output = dt_utils.unidimensionalize(test_x)

    # assert
    assert np.array_equal(output, ans)

def test_Unidimensionalize_Input2darrayWithOneColumn_Return1darray():
    # arrange
    np.random.seed(0)
    ans = np.random.rand(10)
    test_x = ans[np.newaxis].T

    # act 
    output = dt_utils.unidimensionalize(test_x)

    # assert
    assert np.array_equal(output, ans)

def test_Unidimensionalize_Input2darrayWithOneRow_Return1darray():
    # arrange
    np.random.seed(0)
    ans = np.random.rand(10)
    test_x = ans[np.newaxis]

    # act 
    output = dt_utils.unidimensionalize(test_x)

    # assert
    assert np.array_equal(output, ans)    

def test_Unidimensionalize_Input2darrayWithManyColumns_RaiseException():
    # arrange
    np.random.seed(0)
    test_x = np.random.rand(10, 2)

    # act & assert
    with pytest.raises(Exception) as e_info:
        output = dt_utils.unidimensionalize(test_x)

def test_Unidimensionalize_InputNdarray_RaiseException():
    # arrange
    np.random.seed(0)
    test_x = np.random.rand(10, 2, 2)

    # act & assert
    with pytest.raises(Exception) as e_info:
        output = dt_utils.unidimensionalize(test_x)



def test_SetEvalData_InputLists_ReturnArray():
    # arrange 
    test_x = [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]
    test_y = [11, 12, 13, 14]
    test_w = [-1, -2, -3]

    # act 
    output_x, output_y, output_w = dt_utils.set_eval_data((test_x, test_y, test_w))

    # assert
    assert np.array_equal(output_x, np.array(test_x))
    assert np.array_equal(output_y, np.array(test_y))
    assert np.array_equal(output_w, np.array(test_w))

def test_SetEvalData_InputArrayNeedsModified_ReturnArray():
    # arrange
    ans = np.array([1, 2, 3, 4, 5])
    test_column_vector = ans[np.newaxis].T
    test_row_vector =  np.array([[1, 2, 3, 4, 5]])

    # act 
    output_column, output_row = dt_utils.set_eval_data((test_column_vector, test_row_vector))

    # assert
    assert np.array_equal(ans, output_column)
    assert np.array_equal(ans, output_row)

def  test_SetEvalData_InputMixedTypes_ReturnArray():
    # arrange
    test_2darray = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
    test_column_vector = np.array([[1], [2], [3], [4], [5]])
    test_row_vector =  np.array([[1, 2, 3, 4, 5]])
    test_list = [1, 2, 3, 4, 5]
    ans_1darray = np.array([1, 2, 3, 4, 5])

    # act
    output_2darray, output_column, output_row, output_list = dt_utils.set_eval_data(
        (test_2darray, test_column_vector, test_row_vector, test_list)
    )

    # assert
    assert np.array_equal(test_2darray, output_2darray)
    assert np.array_equal(ans_1darray, output_column)
    assert np.array_equal(ans_1darray, output_row)
    assert np.array_equal(ans_1darray, output_list)