import numpy as np
from math import sqrt
from ..base.optimizers import Optimizer
from ..utils.data_utils import set_eval_data, add_bias_term
from ..utils.initialize_utils import initialize_weight

# @alias_class(GD)
# @alias_str(["gd", "gradientdescent", "GD", "GradientDescent"])
class GradientDescent(Optimizer):
    def __init__(self, learning_rate=None, lr=None):
        self.parameters = {}
        self.parameters['lr'] = learning_rate if lr is not None else lr 
        # lr is the abbreviation for "learning rate"

    def execute(self, x, y, loss, activation):
        x, y = set_eval_data((x, y))
        x = add_bias_term(x)
        w = initialize_weight(x=x, method="zeros")
        lr = self.parameters['lr']
        converged = False
        while not converged:
            gradient = loss.grad(x@w, y) @ activation.grad(x, w)
            update = w - lr * gradient
            if np.array_equal(w, update): 
                converged = True
            else:
                w = update
        return w

# @alias_class(SGD)
# @alias_str(["sgd", " stochasticgradientdescent", "SGD", "StochasticGradientDescent"])
class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate=None, lr=None):
        self.parameters = {}
        self.parameters['lr'] = learning_rate if lr is not None else lr 

    def execute(self, x, y, loss, activation):
        x, y = set_eval_data((x, y))
        x = add_bias_term(x)
        w = initialize_weight(x=x, method="zeros")
        lr = self.parameters['lr']
        converged = False
        while not converged:
            for x_i, y_i in zip(x, y):
                gradient = loss.grad(x_i@w, y_i) * activation.grad(x_i, w)
                update = w - lr * gradient
                if np.array_equal(w, update): 
                    converged = True
                    break
                else:
                    w = update
        return w

# @alias_str(["Adagrad", "adagrad"])
class Adagrad(Optimizer):
    def __init__(self, learning_rate=None, lr=None):
        self.parameters = {}
        self.parameters['lr'] = learning_rate if lr is not None else lr 
    
    def execute(self, x, y, loss, activation):
        x, y = set_eval_data((x, y))
        x = add_bias_term(x)
        w = initialize_weight(x=x, method="random")
        lr = self.parameters['lr']
        
        t = 0
        adaptive_denominator = np.zeros(w.shape)

        def time_decayed_lr(times):
            return lr / sqrt(times+1)

        def adaptive_term(times):
            nonlocal adaptive_denominator
            sum_of_former_gradients_square = times * (adaptive_denominator**2)
            adaptive_denominator = np.sqrt((gradient**2 + sum_of_former_gradients_square) / (times+1))
            return adaptive_denominator
        
        converged = False
        while not converged:
            gradient = loss.grad(x@w, y) @ activation.grad(x, w)
            update = w - (time_decayed_lr(t) / adaptive_term(t)) * gradient
            t = t + 1
            if np.array_equal(w, update): 
                converged = True
            else:
                w = update
        return w



    
