import numpy as np
import numba
from numba.experimental import jitclass
from numpy.linalg import inv, pinv

from ..base.optimizers import Optimizer
from ..regularizers import Null, L1, L2, Tikhonov, is_instance

from ..utils.data_utils import add_cons
from ..utils.decorator_utils import implementation
from ..utils.initialize_utils import init_weight, set_x_train, set_y_train
from ..utils.operation_utils import allclose



class AnalyticSolution(Optimizer):
    def execute(self, X, y, regularizer=Null(), **kwargs):
        x = set_x_train(X, **kwargs)
        y = set_y_train(y)
        kernelgram_X = kwargs.get("kernelgram_X", False)

        if is_instance(regularizer, Null):
            w = self.analytic_solution_linear_regression(x, y)

        elif is_instance(regularizer,  L1):
            raise Exception("L1 regularizer does not support analytic solution.")
        
        elif is_instance(regularizer, L2):
            w = self.analytic_solution_ridge_regression(x, y, regularizer)

        elif is_instance(regularizer, Tikhonov):
            if kernelgram_X == True:
                w = self.analytic_solution_gerneralized_ridge_regression(x, y, regularizer)
            else:
                w = self.analytic_solution_kernel_ridge_regression(x, y, regularizer)

        return w


    @implementation(numba_jit=True)
    def analytic_solution_linear_regression(x, y):
        w = pinv(x) @ y
        return w


    @implementation(numba_jit=True)
    def analytic_solution_ridge_regression(x, y, l2_regularizer):
        lambda_ = l2_regularizer.l
        w = inv((x.T @ x) + (lambda_ * np.identity(x.shape[1]))) @ (x.T @ y)
        return w


    @implementation(numba_jit=True)
    def analytic_solution_gerneralized_ridge_regression(x, y, tikhonov_regularizer):
        # In Tikhonov regularization, the regularized weights 
        #     are given by w_reg = (xᵀx + λΓᵀΓ)⁻¹(xᵀy)
        lambda_ = tikhonov_regularizer.l
        gamma_square = tikhonov_regularizer.Gamma_square
        w = inv((x.T @ x) + (lambda_ * gamma_square)) @ (x.T @ y)
        return w


    @implementation(numba_jit=True)
    def analytic_solution_kernel_ridge_regression(x, y, regularizer):
        # We have a analytic solution β = (λI + K)⁻¹(y) 
        #     for Kernel Ridge Regression. Although We can still obtain 
        #     the same solution using Tikhonov's solution, the formula
        #     here costs less in computation.
        lambda_ = regularizer.l
        gamma_square = regularizer.Gamma_square
        w = inv(gamma_square + (lambda_ * np.identity(x.shape[0]))) @ y
        return w
    


class GradientDescent(Optimizer):
    def __init__(self, lr=0.1, init_method="linear_reg"):
        self.lr = lr
        self.init_method = init_method


    def execute(self, X, y, loss, regularizer=Null(), **kwargs):
        lr = self.lr
        x = set_x_train(X, **kwargs)
        y = set_y_train(y)
        init_w = init_weight(x, y, method=self.init_method)

        w = self.gradient_descent(x, y, init_w, lr, loss, regularizer)

        if np.any(np.isnan(w)):
            msg = "Diverge. Current lr:{}, You may want to train with a smaller learning rate."
            raise Exception(msg.format(lr))
        else:
            return w


    @implementation(numba_jit=True)
    def gradient_descent(x, y, init_w, lr, loss, regularizer):
        w = init_w
        while True:
            grad = loss.grad(w, x, y)
            regularization = regularizer.grad(w)/len(x)
            
            prev_w = w
            w = w - lr * grad - regularization

            # leave iteration when weights have converged or diverged
            if allclose(w, prev_w) or np.any(np.isnan(w)): 
                break
        return w



class StocasticGradientDescent(Optimizer):
    def __init__(self, lr=0.1, init_method="linear_reg", epochs=100):
        self.lr = lr
        self.init_method = init_method
        self.epochs = epochs    # Set epochs using model.compile() is not recommended practice in 
                                #     most of the common situations. This option is left for setting epochs 
                                #     in ensemble meta-algorithm. Set epochs when you call fit().


    def execute(self, X, y, loss, epochs=None, regularizer=Null(), **kwargs):
        x = set_x_train(X, **kwargs)
        y = set_y_train(y)
        init_w = init_weight(x, y, method=self.init_method)
        lr = self.lr
        epochs = self.epochs if (epochs is None) else epochs
        self.epochs = epochs
        
        iters = len(x) * epochs
        w = self.stocastic_gradient_descent(x, y, init_w, lr, iters, loss, regularizer)

        if np.any(np.isnan(w)):
            msg = "Diverge. Current lr:{}, You may want to try with a smaller learning rate"
            raise Exception(msg.format(lr))
        else:
            return w 


    @implementation(numba_jit=True)
    def stocastic_gradient_descent(x, y, init_w, lr, iters, loss, regularizer):
        w = init_w
        for _ in range(iters):
            i = np.asarray([np.random.randint(0, x.shape[0])])
            x_i = x[i, :]
            y_i = y[i]
            
            grad = loss.grad(w, x_i, y_i)
            regularization = regularizer.grad(w)/len(x)
            
            w = w - lr*grad - regularization
        return w