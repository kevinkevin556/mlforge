from .optimizers import LinearSeparable, GradientDescent
from ..base.models import Model
from ..losses import ZeroOneError, MeanSquaredError
from ..metrics import Accuracy

from ..utils.data_utils import add_cons
from ..utils.operation_utils import sign
from ..utils.decorator_utils import fit_method, predict_method

class Perceptron(Model): 
    model_type = "binary-classifier"

    loss = ZeroOneError()

    def __init__(self, optimizer=LinearSeparable(), metrics=[Accuracy()]):
        self.weight_ = None
        self.compile(optimizer=optimizer, metrics=metrics)
    
    @fit_method
    def fit(self, X, y, **kwargs):
        self.weight_ = self.optimizer.execute(X, y, loss=Perceptron.loss, **kwargs)
        return self

    @predict_method
    def predict(self, X):
        X = add_cons(X)
        w = self.weight_
        return sign(X@w)



class Adaline(Model):
    model_type = "binary-classifier"

    loss = MeanSquaredError()

    def __init__(self, optimizer=GradientDescent(), metrics=[Accuracy()]):
        self.weight_ = None
        self.compile(optimizer=optimizer, metrics=metrics)

    @fit_method
    def fit(self, X, y, **kwargs):
        self.weight_ = self.optimizer.execute(X, y, loss=Adaline.loss, **kwargs)
        return self

    @predict_method
    def predict(self, X):
        X = add_cons(X)
        w = self.weight_
        return sign(X@w)