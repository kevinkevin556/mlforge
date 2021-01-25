import numpy as np
import numba

from .jitpickle_utils import jitpickle


def abstractmethod(func):
    return func


def implementation(numba_jit=True, multiple_func_impl=False, **kwargs):
    def func_wrapper(func):
        if numba_jit:
            return staticmethod(numba.jit(nopython=True, **kwargs)(func))
        elif multiple_func_impl:
            return func
        else:
            return staticmethod(func)
            
    return func_wrapper


def fit_method(func):
    def func_wrapper(self, X, y):
        def check_2d(X):
            X = np.asarray(X)
            if len(X.shape) == 1:
                return X.reshape(-1,1)
            else:
                return X

        def data_class_encode(self, y):
            if self.__estimator_type__ == "binary_classification":
                values = np.unique(y)
                n_class = len(values)
                if n_class > 2:
                    raise ValueError("Invalid y_fit for binary classificator.")
                else:
                    value_encoding = {-1:values[0], 1:values[1], "dtype": y.dtype}
                    setattr(self, "classes_", value_encoding)
                    new_y = np.empty(y.shape)
                    new_y[y==values[0]] = -1
                    new_y[y==values[1]] = 1
                    return new_y
            else:
                return y

        X = check_2d(X)
        new_y = data_class_encode(self, y)
        return func(self, X, new_y)
    
    return func_wrapper


def predict_method(func):
    def func_wrapper(self, X, **kwargs):
        def data_class_decode(y_pred):
            output = np.empty(shape=y_pred.shape, dtype=value_encoding["dtype"])
            output[y_pred == -1] = value_encoding[-1]
            output[y_pred == 1] = value_encoding[1]
            return output
        
        if self.__estimator_type__ == "binary_classification":
            value_encoding = self.classes_
            output = data_class_decode(func(self, X, **kwargs))
        else:
            output = func(self, X, **kwargs)
        
        return output

    return func_wrapper






