import numpy as np
import numba
import copy

from ..base import models, meta_algorithms
from .data_utils import add_cons, to_vector

@numba.jit(nopython=True)
def init_weight(x, y, method="zeros"):
    """ 
    The function is called to initialize weight vector.
    Initializer is a function in the current version.
    It will soon be developed into a class.

    input: 2d-array, 1d-array, str
    output: 1d-array  
    """

    if method == "zeros":
        w = np.zeros(x.shape[1])
    elif method == "random":
        w = np.random.rand(x.shape[1])
    elif method == "linear_reg":
        w = np.dot(np.linalg.pinv(x), y)
    else:
        raise ValueError("Invalid initializing approach.")
    return w


def new_instance(model, mode="default"):
    """
    Create a new instance of the given model.

    mode: 
    "default", create a new instance with constructor
    "copy", make a deepcopy of the given instance   

    input: class, or an instance of the class; string
    output: an instance 
    """
    if type(model) is type and issubclass(model, (models.Model, meta_algorithms.MetaAlgorithm)): 
        return model()
    elif isinstance(model,  (models.Model, meta_algorithms.MetaAlgorithm)):
        if mode == "default":
            return model.__class__()
        elif mode == "copy":
            return copy.deepcopy(model)
        else:
            raise ValueError("Invalid mode.")
    else:
        raise ValueError("This model does not exist.")


def set_x_train(x, add_bias=True, kernelgram_X=False):
    output = np.asarray(x)

    # (1) In some case we tackle constant terms and features
    #  seperatly (eg. SVM),
    #  we could set add_bias = False
    # (2) Kernelized data need not be transformed
    if add_bias and (not kernelgram_X):
        output = add_cons(output)
    
    output = np.array(output, dtype=np.float64)
    return output


def set_y_train(y):
    output = to_vector(y)
    output = np.array(output, dtype=np.float64)
    return output
    
