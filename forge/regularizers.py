import warnings
import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from .utils.operation_utils import sign
from .utils.decorator_utils import abstractmethod
from .utils.jitpickle_utils import jitpickle
from .utils.print_utils import format_repr



class Regularizer:
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def grad(self, w):
        return 0
    
# API

    def get_config(self):
        var_names = self.__init__.__code__.co_varnames
        params = {v:getattr(self, v) for v in var_names if v!="self"}
        return params
    
    def __repr__(self):
        return format_repr(self.__class__.__name__, self.get_config())


# is regularizer_cls' instance
def is_instance(regularizer_obj, regularizer_cls):
    return regularizer_obj._numba_type_.classname == regularizer_cls.class_type.class_name



@jitclass([])
class Null(Regularizer):
    def __init__(self):
        pass
    
    def grad(self, w):
        return 0


@jitclass([('l', float64)])
class L1(Regularizer):
    def __init__(self, l=0.01):
        self.l = l

    def grad(self, w):
        return (self.l) * sign(w, zero=0)


@jitclass([('l', float64)])
class L2(Regularizer):
    def __init__(self, l=0.01):
        self.l = l
    
    def grad(self, w):
        return (self.l) * w



@jitclass([('l',            float64),
           ('Gamma',        float64[:,:]),
           ("Gamma_square", float64[:,:])])
class Tikhonov(Regularizer):
    def __init__(self,l=0.01, Gamma_square=np.zeros((1,1)), Gamma=np.zeros((1,1))):
        self.l = l
        self.Gamma = Gamma
        if Gamma_square is None:
            self.Gamma_square = self.Gamma.T @ self.Gamma
        else:
            self.Gamma_square = Gamma_square
    
    def grad(self, w):
        A = self.Gamma_square
        return (self.l) * (A + A.T) @ w