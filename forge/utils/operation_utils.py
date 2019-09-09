import numpy as np
from ..base.losses import Loss


def sign(x):
    """
    Returns the sign of a number, i.e.
    x > 0  return 1 
    x <= 0  return -1
    If x is a nd-array, the functions is applied element-wisely.

    input: number or ndarray
    output: number or ndarray
    """
    return 1*(x >= 0) + ((x>=0)-1)


# def dd(y, x):
#     if y is Loss and x is Activation:
#         d = Derivative(loss=y, activation=x, weight=None)
#     else if y is Activation and x is Weight:
#         d = Derivation(loss=None, activation=y, weight=x)
#     else:
#         raise ValueError("Invalid input class.")
#     return d
