import random
import numpy as np

from ..base.optimizers import Optimizer
from ..losses import ZeroOneError, MeanSquaredError

from ..utils.data_utils import add_cons
from ..utils.decorator_utils import implementation
from ..utils.initialize_utils import init_weight, set_x_train, set_y_train
from ..utils.operation_utils import sign, allclose



class LinearSeparable(Optimizer):
    def __init__(self, lr=1, init_method="zeros"):
        self.lr = lr
        self.init_method = init_method


    def execute(self, x_train, y_train, loss=ZeroOneError()):
        lr = self.lr
        x = set_x_train(x_train)
        y = set_y_train(y_train)
        init_w = init_weight(x, y, method=self.init_method)

        return self.linear_separable_pla(x, y, init_w, lr)


    @implementation(numba_jit=True)
    def linear_separable_pla(x, y, init_w, lr):
        w = init_w
        correct_counter = 0
        while True:
            for x_i, y_i in zip(x, y):
                if sign(x_i @ w) != y_i:
                    w = w + lr * y_i * x_i
                    correct_counter = 0
                else:
                    correct_counter = correct_counter + 1
            
            if correct_counter >= len(x):
                break
        return w



class Pocket(Optimizer):
    def __init__(self, lr=1, init_method="zeros", updates=50, return_pocket=True):
        self.lr = lr
        self.init_method = init_method
        self.updates = updates
        self.return_pocket = return_pocket


    def execute(self, x_train, y_train, loss=ZeroOneError(), updates=None):
        lr = self.lr
        x = set_x_train(x_train)
        y = set_y_train(y_train)
        init_w = init_weight(x, y, method=self.init_method)
        updates = self.updates if updates is None else updates
        return_pocket = self.return_pocket
        
        w_pocket, w = self.pocket(x, y, init_w, lr, updates)
        
        if return_pocket:
            return w_pocket
        else:
            return w


    @implementation(numba_jit=True)
    def pocket(x, y, init_w, lr, updates):
        w = init_w
       
        # Use Classification Error (0/1 Error)
        loss = ZeroOneError() 
        w_pocket = init_w
        pocket_loss = loss.eval(w_pocket, x, y)
        
        # Pocket algorithm iteration
        while updates > 0:
            i = random.randint(0, len(x)-1)
            x_i, y_i = x[i, ], y[i]
            
            if sign(x_i @ w) != y_i:
                w = w + lr * y_i * x_i
                current_loss = loss.eval(w, x, y)
                updates = updates - 1

                if current_loss < pocket_loss:
                    w_pocket = w
                    pocket_loss = current_loss

                # Converged before iteration ends
                if current_loss == 0:   
                    w_pocket = w
                    break

        return w_pocket, w



class GradientDescent(Optimizer):
    def __init__(self, lr=0.1, init_method="zeros", epochs=50):
        self.lr = lr
        self.init_method = init_method
        self.epochs = epochs


    def execute(self, x_train, y_train, loss=MeanSquaredError(), epochs=None):
        x = set_x_train(x_train)
        y = set_y_train(y_train)
        init_w = init_weight(x, y, method=self.init_method)
        lr = self.lr
        epochs = self.epochs if epochs is None else epochs

        iters = x.shape[0] * epochs
        w = self.gradient_descent(x, y, init_w, lr, loss, iters)

        if np.isnan(w).any():
            raise Exception("Diverge. Current lr:{}, You may want to train with a smaller learning rate.".format(lr))
        else:
            return w


    @implementation(numba_jit=True)
    def gradient_descent(x, y, init_w, lr, loss, iters):
        w = init_w
        while iters > 0:
            prev_w = w
            w = w - lr * loss.grad(w, x, y)
            iters -= 1
            if allclose(prev_w, w):
                break
        return w 
