import numpy as np

def initialize_weight(x=None, y=None, method="zeros"):
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
        raise ValueError("Invalid initializing approach!!")
    return w



