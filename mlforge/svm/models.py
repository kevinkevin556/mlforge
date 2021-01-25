import numpy as np

from .. import losses, metrics
from ..base.models import Model
from ..kernels import Gaussian, Linear
from ..regression import LogisticRegression, StocasticGradientDescent
from ..utils.data_utils import add_cons
from ..utils.decorator_utils import fit_method, predict_method
from ..utils.operation_utils import sign
from .optimizers import DualQpSolver, PrimalQpSolver


class HardMarginSVM(Model):

    __estimator_type__ = "binary_classification"

    def __init__(
        self,
        optimizer = PrimalQpSolver(), 
        metrics = [metrics.Accuracy()],
        kernel = Linear()
    ):
        self.coef_ = {
            "w": None,      # Coefficients in the primal problem
            "alpha": None,  # Coefficients in the dual problem
            "b": None       # Constants in decision function
        }
        self.support_vectors_ = {
            "x": None,      
            "y": None,
            "i": None       # Indices of support vectors
        }

        self.compile(optimizer=optimizer, metrics=metrics, kernel=kernel)


    @fit_method
    def fit(self, X, y):
        # Clear previous training result 
        self.coef_ = dict.fromkeys(iter(self.coef_.keys()), None)
        self.support_vectors_ = dict.fromkeys(iter(self.support_vectors_.keys()), None)

        if isinstance(self.optimizer, PrimalQpSolver):
            self.coef_["b"], self.coef_["w"] = self.optimizer.execute(X, y)
        if isinstance(self.optimizer, DualQpSolver):
            self.coef_["b"],  self.coef_["alpha"], self.support_vectors_= self.optimizer.execute(X, y)
        
        return self


    @predict_method
    def predict(self, X):
        w, alpha, b = tuple(self.coef_.get(key) for key in ("w", "alpha", "b"))
        trsf = self.kernel.transform

        if w is not None:
            return sign(trsf(X) @ w + b)
        if alpha is not None:
            sv_i = self.support_vectors_["i"]
            sv_x = self.support_vectors_["x"]
            sv_y = self.support_vectors_["y"]
            K = self.kernel.inner_product
            return sign(alpha[sv_i] * sv_y @ K(sv_x, X) + b)



class SoftMarginSVM(Model):

    __estimator_type__ = "binary_classification"

    def __init__(
        self,
        optimizer = DualQpSolver(), 
        metrics = [metrics.Accuracy()],
        kernel = Gaussian(),
        soft_margin_penalty = 1
    ):
        self.coef_ = {
            "w": None,      # Coefficients in the primal problem
            "alpha": None,  # Coefficients in the dual problem
            "b": None       # Constants in decision function
        }
        self.support_vectors_ = {
            "free":{
                "x": None,
                "y": None, 
                "i": None   # Indices of free support vectors
            }, 
            "bounded":{
                "x": None, 
                "y": None, 
                "i": None   # Indices of bounded support vectors
            }
        }

        self.compile(
            optimizer = optimizer,
            metrics = metrics,
            kernel = kernel,
            soft_margin_penalty=soft_margin_penalty
        )
    

    @fit_method
    def fit(self, X, y):        
        # Clear previous training result 
        self.coef_ = dict.fromkeys(iter(self.coef_.keys()), None)
        self.support_vectors_ = dict.fromkeys(iter(self.support_vectors_.keys()), {"x": None, "y": None, "i": None})
        
        if type(self.optimizer) is PrimalQpSolver:
            self.coef_["b"], self.coef_["w"] = self.optimizer.execute(X, y)
        if type(self.optimizer) is DualQpSolver:
            self.coef_["b"], self.coef_["alpha"], self.support_vectors_= self.optimizer.execute(X, y)
        
        return self


    @predict_method
    def predict(self, X):
        w, alpha, b = tuple(self.coef_.get(key) for key in ("w", "alpha", "b"))
        trsf = self.kernel.transform

        if w is not None:
            return sign(trsf(X) @ w + b)
        if alpha is not None:
            sv_i = np.hstack((self.support_vectors_["free"]["i"], self.support_vectors_["bounded"]["i"]))
            sv_x = np.vstack((self.support_vectors_["free"]["x"], self.support_vectors_["bounded"]["x"]))
            sv_y = np.hstack((self.support_vectors_["free"]["y"], self.support_vectors_["bounded"]["y"]))
            K = self.kernel.inner_product
            return sign((alpha[sv_i] * sv_y) @ K(sv_x, X) + b)



class ProbabilisticSVM(Model):

    __estimator_type__ = "binary_classification"

    svm = SoftMarginSVM()
    log_reg = LogisticRegression()

    def __init__(
        self,
        optimizer = [DualQpSolver(), StocasticGradientDescent()],
        metrics = [metrics.Accuracy()],
        **kwargs
    ):
        self.compile(optimizer=optimizer, metrics=metrics)


    @fit_method 
    def fit(self, X, y):
        self.svm = SoftMarginSVM(optimizer = self.optimizer[0])
        self.svm.fit(X, y)

        x, y = self.svm.predict(X).reshape(-1, 1), y # use fit decorator to omit reshape()

        self.log_reg = LogisticRegression(optimizer = self.optimizer[1])
        self.log_reg.fit(x, y)
        
        return self


    @predict_method
    def predict(self, X):
        return self.log_reg.predict_proba(self.svm.predict(X).reshape(-1, 1)) # use predict decorator to omit reshape()



class SVR(Model):

    __estimator_type__ = "regression"

    def __init__(
        self, 
        optimizer = DualQpSolver(), 
        metrics = [metrics.MeanSquaredError], 
        kernel = Gaussian(), 
        soft_margin_penalty = 1, # C
        tube_width = 0.05        # epsilon
    ):
        self.compile(
            optimizer=optimizer,
            metrics=metrics,
            kernel=kernel,
            soft_margin_penalty=soft_margin_penalty,
            tube_width=tube_width
        )
        self.coef_ = {"b":None, "w":None, "beta":None}
        self.support_vectors_ = {
            "x": None,      
            "y": None,
            "i": None       # Indices of support vectors
        }


    @fit_method
    def fit(self, X, y):
        # Clear previous training result 
        self.coef_ = dict.fromkeys(iter(self.coef_.keys()), None)
        self.support_vectors_ = dict.fromkeys(iter(self.support_vectors_.keys()), None)

        if type(self.optimizer) is PrimalQpSolver:
            self.coef_["b"], self.coef_["w"] = self.optimizer.execute(X, y)
        if type(self.optimizer) is DualQpSolver:
            self.coef_["b"], self.coef_["beta"], self.support_vectors_= self.optimizer.execute(X, y)
        
        return self


    @predict_method
    def predict(self, X):
        w, beta, b = tuple(self.coef_.get(key) for key in ("w", "beta", "b"))
        
        if w is not None:
            return X @ w + b
        if beta is not None:
            sv_i = np.hstack((self.support_vectors_["free"]["i"], self.support_vectors_["bounded"]["i"]))
            sv_x = np.vstack((self.support_vectors_["free"]["x"], self.support_vectors_["bounded"]["x"]))
            K = self.kernel.inner_product
            return beta[sv_i] @ K(sv_x, X) + b
