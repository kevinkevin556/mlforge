import numpy as np

from ..base.optimizers import Optimizer
from ..utils.initialize_utils import set_x_train, set_y_train
from ..utils.decorator_utils import implementation


class NaiveBayesSolver(Optimizer):
    def __init__(self, distribution=None):
        self.distribution = distribution
    

    def execute(self, X, y):
        x = set_x_train(X, add_bias=False)
        y = set_y_train(y)
        return self.naive_bayes_solver(x, y, self.distribution)


    @implementation(numba_jit=True)
    def naive_bayes_solver(X, y, distribution):
        p_C1 = np.sum(y==1)/len(y)
        p_Cneg1 = np.sum(y==-1)/len(y)
        
        dist_C1 = distribution.fit(X[y==1, :]).copy()
        dist_Cneg1 = distribution.fit(X[y==-1, :]).copy()

        return p_C1, p_Cneg1, dist_C1, dist_Cneg1
