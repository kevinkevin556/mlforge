import numpy as np
import numba


@numba.jit(nopython=True)
def sign(s, zero=-1):
    """
    Returns the sign of a number, i.e.
    s > 0  return 1 
    s < 0  return -1
    If s is a nd-array, the functions is applied element-wisely.

    input: number or ndarray
    output: number or ndarray
    """

    if zero == -1:
        return 1*(s>0) + ((s>0) - 1)
    if zero == 1:
        return 1*(s>=0) + ((s>=0) - 1)
    if zero == 0:
        return 1*(s>=0) + ((s>0) - 1)


@numba.jit(nopython=True)
def logistic(s):
    return 1 / (1 + np.exp(-s)) 


@numba.jit(nopython=True)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if equal_nan:
        a[np.isnan(a)] = 0
        b[np.isnan(b)] = 0

    return np.all(np.less_equal(np.abs(a-b), atol + rtol*np.abs(b)))


@numba.jit(nopython=True)
def unique_count(ar):
    unique = np.unique(ar)
    unique_counts = np.empty(unique.shape, dtype=np.int32)
    for i in range(len(unique)):
        unique_counts[i] = np.sum(ar == unique[i])
    
    return unique, unique_counts

@numba.jit(nopython=True)
def mean(a, axis=0):
    output = np.empty(a.shape[1-axis])
    for i in range(a.shape[1-axis]):
        if axis == 0:
            output[i] = np.mean(a[:, i])
        if axis == 1 :
            output[i] = np.mean(a[i, :])
    return output

