import numpy as np
import numba

from .alias_utils import add_alias


def abstractmethod(func):
    return func


def alias(aliases):
    def class_wrapper(obj_class):
        add_alias(aliases, obj_class)
        return obj_class

    return class_wrapper


def implementation(tag="", compile="numba", **kwargs):
    def func_wrapper(func):
        if compile == None:
            return func
        if compile == "numba":
            return staticmethod(numba.jit(nopython=True, **kwargs)(func))
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
            if self.model_type in ["regressor", "multiclass-classifier"]:
                return y
            if self.model_type == "binary-classifier":
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
        
        if self.model_type == "binary-classifier":
            value_encoding = self.classes_
            output = data_class_decode(func(self, X, **kwargs))
        else:
            output = func(self, X, **kwargs)    
        
        return output

    return func_wrapper