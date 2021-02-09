import numpy as np

from ..metrics import Accuracy
from ..distributions import Gaussian
from ..base.models import Model
from ..utils.decorator_utils import fit_method, predict_method
from .optimizers import NaiveBayesSolver



class NaiveBayes(Model):
    model_type = "binary-classifier"

    def __init__(
        self, 
        distribution = Gaussian(),
        optimizer = NaiveBayesSolver(), 
        metrics = [Accuracy()]
    ):
        self.p_C1_, self.dist_C1_ = None, None
        self.p_Cneg1_, self.dist_Cneg1_ = None, None

        self.compile(
            distribution = distribution,
            optimizer = optimizer,
            metrics = metrics
        )
    

    @fit_method
    def fit(self, X, y):
        res = self.optimizer.execute(X, y)
        self.p_C1_= res[0]
        self.p_Cneg1_ = res[1]
        self.dist_C1_ = res[2]
        self.dist_Cneg1_ = res[3]
        
        return self
    

    def predict_proba(self, X):
        p_C1_given_x = np.empty(X.shape[0])

        p_x_given_C1 = self.dist_C1_.proba(X)
        p_x_given_Cneg1 = self.dist_Cneg1_.proba(X)
        
        # Bayes Theorem
        p_C1_given_x = (p_x_given_C1 * self.p_C1_) /                                \
                       (p_x_given_C1 * self.p_C1_ + p_x_given_Cneg1 * self.p_Cneg1_)
            
        return p_C1_given_x


    @predict_method
    def predict(self, X):
        output = np.empty(X.shape[0])
        p_C1_given_x = self.predict_proba(X)
        output[p_C1_given_x >= 0.5] = 1
        output[p_C1_given_x < 0.5] = -1
        return output

