from .. import losses, metrics, regularizers
from ..base.models import Model
from ..kernels import Gaussian
from ..regularizers import Null, Tikhonov
from ..utils.data_utils import add_cons
from ..utils.decorator_utils import fit_method, predict_method
from ..utils.operation_utils import logistic, sign
from .optimizers import (AnalyticSolution, GradientDescent,
                         StocasticGradientDescent)


class LinearRegression(Model):
    
    __estimator_type__ = "regression"

    # Model Settings
    loss = losses.MeanSquaredError()
    
    def __init__(
        self,
        optimizer = AnalyticSolution(),
        metrics = [metrics.MeanSquaredError()],
        regularizer = Null()
    ):
        self.weight_ = None
        self.compile(optimizer = optimizer, metrics = metrics, regularizer = regularizer)
    

    @fit_method
    def fit(self, X, y, **kwargs): 
        loss = LinearRegression.loss
        regularizer = self.regularizer
        epochs = kwargs.get("epochs", 1)

        # Execute optimizer to fit data
        analytic = AnalyticSolution
        gd = GradientDescent
        sgd = StocasticGradientDescent

        if isinstance(self.optimizer, analytic):
            self.weight_ = self.optimizer.execute(X, y, regularizer)
        elif isinstance(self.optimizer, gd):
            self.weight_ = self.optimizer.execute(X, y, loss, regularizer)
        elif isinstance(self.optimizer, sgd):
            self.weight_ = self.optimizer.execute(X, y, loss, epochs, regularizer)
        else:
            raise ValueError("Invalid optimizer.")

        return self
 
    @predict_method
    def predict(self, X):
        x = add_cons(X)
        w = self.weight_
        return x@w



class LogisticRegression(Model):
    
    __estimator_type__ = "binary_classification"

    # Model Settings
    loss = losses.CrossEntropyError()

    def __init__(
        self,
        optimizer = StocasticGradientDescent(),
        metrics = [metrics.Accuracy()],
        regularizer = Null()
    ):
        self.weight_ = None
        self.compile(optimizer = optimizer, metrics = metrics, regularizer = regularizer)


    @fit_method
    def fit(self, X, y, **kwargs):
        loss = LogisticRegression.loss
        regularizer = self.regularizer
        epochs = kwargs.get("epochs", 1)

        # Execute optimizer to fit data
        gd = GradientDescent
        sgd = StocasticGradientDescent

        if isinstance(self.optimizer, gd):
            self.weight_ = self.optimizer.execute(X, y, loss, regularizer)
        elif isinstance(self.optimizer, sgd):
            self.weight_ = self.optimizer.execute(X, y, loss, epochs, regularizer)
        else:
            raise ValueError("Invalid optimizer.")

        return self


    def predict_proba(self, X):
        x = add_cons(X)
        w = self.weight_
        prob = logistic(x@w)
        return prob

    
    @predict_method
    def predict(self, X):
        prob = self.predict_proba(X)
        return sign(prob-0.5)



class RidgeRegression(Model):
    
    __estimator_type__ = "regression"

    # Model Settings
    loss = losses.MeanSquaredError
    regularizer = regularizers.L2

    def __init__(
        self,
        optimizer = AnalyticSolution(),
        metrics = [metrics.MeanSquaredError()],
        regul_param = 0.01
    ):
        self.weight_ = None
        self.compile(optimizer=optimizer, metrics=metrics, regul_param=regul_param)


    @fit_method
    def fit(self, X, y, **kwargs):
        loss = RidgeRegression.loss()
        regularizer = RidgeRegression.regularizer(l = self.regul_param)
        epochs = kwargs.get("epochs", 1)

        # Execute optimizer to fit data
        analytic = AnalyticSolution
        gd = GradientDescent
        sgd = StocasticGradientDescent

        if isinstance(self.optimizer, analytic):
            self.weight_ = self.optimizer.execute(X, y, regularizer)
        elif isinstance(self.optimizer, gd):
            self.weight_ = self.optimizer.execute(X, y, loss, regularizer)
        elif isinstance(self.optimizer, sgd):
            self.weight_ = self.optimizer.execute(X, y, loss, epochs, regularizer)
        else:
            raise ValueError("Invalid optimizer.")
        
        return self


    @predict_method
    def predict(self, X):
        x = add_cons(X)
        w = self.weight_
        return x@w



class KernelLogisticRegression(Model):

    __estimator_type__ = "binary_classification"
    
    loss = losses.CrossEntropyError
    regularizer = regularizers.Tikhonov # set Gamma_square = K(x,x) in fit function

    def __init__(
        self,
        optimizer = StocasticGradientDescent(), 
        metrics = [metrics.Accuracy()],
        kernel = Gaussian(),
        regul_param = 0.01      # C
    ):
        self.weight_func_, self.beta_ = None, None
        self.compile(optimizer=optimizer, metrics=metrics, kernel=kernel, regul_param=regul_param)

    @fit_method
    def fit(self, X, y, **kwargs):
        K = self.kernel.inner_product
        loss = KernelLogisticRegression.loss()
        regularizer = KernelLogisticRegression.regularizer(l=self.regul_param, Gamma_square=K(X, X))
        epochs = kwargs.get("epochs", 1)

        # Execute optimizer to fit data
        gd = GradientDescent
        sgd = StocasticGradientDescent

        if isinstance(self.optimizer, gd):
            self.beta_ = self.optimizer.execute(K(X,X), y, loss, regularizer, kernelgram_X=True)
        elif isinstance(self.optimizer, sgd):
            self.beta_ = self.optimizer.execute(K(X,X), y, loss, epochs, regularizer, kernelgram_X=True)
        else:
            raise ValueError("Invalid optimizer.")
        
        self.weight_func_ = lambda z: self.beta_ @ K(X, z)  # with embedded-in-kernel transform & L2 regularizer
                                                            # Since betas are typically non-zero, by representer thm,
                                                            # we need all x to obtain w
        return self

    def predict_proba(self, X):
        X_inner_product_w = self.weight_func_(X) # X@w
        prob = logistic(X_inner_product_w)
        return prob


    @predict_method
    def predict(self, X):
        prob = self.predict_proba(X)
        return sign(prob-0.5)



class KernelRidgeRegression(Model):

    __estimator_type__ = "regression"

    loss = losses.MeanSquaredError      # Linear regression of coef on K-based features
    regularizer = regularizers.Tikhonov # set Gamma_square = K(x,x) in fit function

    def __init__(
        self,
        optimizer = AnalyticSolution(), 
        metrics = [metrics.MeanSquaredError()],
        kernel = Gaussian(),
        regul_param = 0.01      # C
    ):
        self.weight_func_, self.beta_ = None, None
        self.compile(optimizer=optimizer, metrics=metrics, kernel=kernel, regul_param=regul_param)


    @fit_method
    def fit(self, X, y, **kwargs):
        K = self.kernel.inner_product
        loss = KernelRidgeRegression.loss()
        # regularization of coef on K-based regularizer
        regularizer = KernelRidgeRegression.regularizer(l = self.regul_param, Gamma_square = K(X, X))  
        epochs = kwargs.get("epochs", 1)

        # Execute optimizer to fit data
        analytic = AnalyticSolution
        gd = GradientDescent
        sgd = StocasticGradientDescent

        if isinstance(self.optimizer, analytic):
            self.beta_ = self.optimizer.execute(X, y, regularizer, add_bias=False)
        elif isinstance(self.optimizer, gd):
            self.beta_ = self.optimizer.execute(K(X, X), y, loss, regularizer, kernelgram_X=True)
        elif isinstance(self.optimizer, sgd):
            self.beta_ = self.optimizer.execute(K(X, X), y, loss, epochs, regularizer, kernelgram_X=True)

        self.weight_func_ = lambda z: self.beta_ @ K(X, z)  # with embedded-in-kernel transform & L2 regularizer
                                                            # Since betas are typically non-zero, by representer thm,
                                                            # we need all x to obtain w
        return self


    @predict_method
    def predict(self, X):
        return self.weight_func_(X) # X@w
