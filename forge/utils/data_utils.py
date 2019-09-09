import numpy as np

def add_bias_term(x):
    """
    Add constant terms into features

    input: 2d-array    
    output: 2d-array
    """
    
    if len(x.shape) > 2:
        raise ValueError("Invalid data: not for array with more than 2 dimensions.")
    if len(x.shape) == 1:
        x = x[np.newaxis].T

    n = len(x)
    cons_feature = np.ones(n).reshape((n, 1))
    return np.hstack((cons_feature, x))


def unidimensionalize(x):
    """
    Coerce list or ndarray into 1d-array.
    This should be done before evaluating loss or metrics.

    input: list, 1d-array or 2d-array   
    output: 1d-array
    """
    if type(x) != np.array:
        x = np.array(x)
    if len(x.shape) == 1:
        return x
    if len(x.shape) == 2:
        if (x.shape[0] == 1) or (x.shape[1] == 1):
            x = np.squeeze(x)
            return x
        else: 
            raise ValueError("Invalid input: 2d-array but neither a column vector nor a row vector.")
    else:
        raise ValueError("Invalid input: not for array with more than 2 dimensions.")


def set_eval_data(data_tuple):
    """
    Standarize data into the specific form for the sake of computation.   
    We need x to be 2d-array, and both w and y to be 1d-array(vector).

    input: list or ndarray   
    output: 1d_array or 2d-array, depends on what data is sended into the function. 
    """

    output = [np.array(dt) for dt in data_tuple]
    output = [unidimensionalize(dt) 
              if len(dt.shape)==2 and
              ((dt.shape[0] == 1) or (dt.shape[1] == 1)) 
              else dt for dt in output]
    
    for dt in output:
        if dt is len(dt.shape) > 2:
            raise ValueError("Invalid input")        
    
    return tuple(output)