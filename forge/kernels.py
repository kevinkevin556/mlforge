import numpy as np
import inspect
import operator
from functools import reduce 
from itertools import combinations_with_replacement
from collections import Counter 
from math import sqrt, factorial
from numpy.linalg import norm

from .utils.data_utils import add_cons
from .utils.polynomial_util import C, H, get_poly_coef
from .utils.print_utils import format_repr


class Kernel():
    def __init__(self):
        pass

    def transform(self, x):
        pass

    def inner_product(self, x1, x2):
        pass 

    def __call__(self, x1, x2):
        return self.inner_product(x1, x2)

    def __repr__(self):
        init_sig = inspect.signature(self.__init__)
        param_names = [param.name for param in init_sig.parameters.values()
                                  if  param.name != "self" and 
                                      param.kind != param.VAR_KEYWORD ]
        output = {}
        for p_name in param_names:
            p_object = getattr(self, p_name)
            output[p_name] = p_object
        return format_repr(self.__class__.__name__, output)


class Linear(Kernel):
    def transform(self, x):
        return x
    
    def inner_product(self, x1, x2):
        return  x1 @ x2.T


class SimplePoly2(Kernel):
    def transform(self, x, return_label=False):
        d = x.shape[1]
        result = np.empty((len(x), 1+d+d**2))   
        label = list(range(0, d+1))
        result[:, 0:(d+1)] = add_cons(x)

        for i in range(d):
            for j in range(d):
                xi_xj = x[:, i] * x[:, j]
                label.append((i+1, j+1))
                result[:, len(label)-1] = xi_xj

        if return_label:
            return result, label
        else:
            return result

    def inner_product(self, x1, x2):
        xx = x1 @ (x2.T)
        return 1 + xx + xx**2


class Polynomial(Kernel):
    def __init__(self, degree=2, beta=1, gamma=1):
        self.degree = degree
        self.beta = beta
        self.gamma = gamma

    def transform(self, x, return_label=False):
        x = add_cons(x)
        
        beta = self.beta
        gamma = self.gamma
        degree = self.degree
        num_features = x.shape[1]

        label = list(combinations_with_replacement(range(num_features), degree))
        result = np.empty((len(x), len(label)))

        for i in range(len(label)):
            count = dict(Counter(label[i]))

            cons_deg = count.get(0, 0)
            noncons_deg = degree - cons_deg
            poly_coef = get_poly_coef(deg=degree, terms=count)
            coef = sqrt((beta**cons_deg) * (gamma**noncons_deg) * poly_coef)

            x_product = coef * reduce(np.multiply,
                                        [np.power(x[:, k], v) for k, v in count.items()])
            result[:, i] = x_product
        
        if return_label:
            return result, label
        else:
            return result

    def inner_product(self, x1, x2):
        b = self.beta
        r = self.gamma
        k = self.degree
        xx = x1 @ (x2.T)
        return (r * xx + b)**k


class Gaussian(Kernel):
    def __init__(self, gamma=1, inf_dim=3):
        self.gamma = gamma
        self.inf_dim = inf_dim  # max dimension if taylor expansion is conducted
        
    def transform(self, x, return_label=False):
        """
        Ref:　https://arxiv.org/pdf/0904.3664v1.pdf (p.39)
        """
        x = add_cons(x)
        
        gamma = self.gamma
        max_dim = self.inf_dim
        num_features = x.shape[1]
        norm_square = (x @ x.T)[range(len(x)), range(len(x))]

        label = list(combinations_with_replacement(range(num_features), max_dim))
        result = np.empty((len(x), len(label)))

        for i in range(len(label)):
            count = dict(Counter(label[i]))
            
            deg = sum(count.values()) - count.get(0, 0)
            poly_coef = get_poly_coef(terms=count)

            if deg == 0:
                coef = np.exp(-sqrt(gamma) * norm_square)
            else:
                coef = sqrt(2**deg / factorial(deg))**(1/deg) * \
                   sqrt(poly_coef) * \
                   np.exp(-sqrt(gamma) * norm_square)**(1/deg) 

            x_product = coef * reduce(np.multiply, [x[:, k]**v for k, v in count.items()])
            result[:, i] = x_product
        
        if return_label:
            return result, label
        else:
            return result

    def inner_product(self, x1, x2):
        gamma = self.gamma
        
        if len(x2.shape) == 1:
            x2 = x2.reshape((1, -1))
            
        x1_sqr = np.einsum('ij, ij -> i', x1, x1).reshape((-1, 1))
        x2_sqr = np.einsum('ij, ij -> i', x2, x2)
        norm_sqr = x1_sqr - 2*x1@(x2.T) + x2_sqr
        return np.exp(-gamma * norm_sqr)

