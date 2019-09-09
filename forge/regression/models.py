from ..base.models import Model
from ..utils.data_utils import add_bias_term
from ..activations import Linear
from ..losses import MeanSquareError


class LinearReg(Model):
    def __init__(self, optimizer=None):
        self.optimizer = None
        self.weight = None
        self.metrics = [MeanSquareError()]
        self.loss = MeanSquareError()
        self.activation = Linear()
        self.compile(optimizer)

    def compile(self, optimizer):
        self.optimizer = optimizer
        
    def fit(self, x_train, y_train):
        self.weight = self.optimizer.execute(x_train, y_train,
                                            activation=self.activation,
                                            loss=self.loss)

    def predict(self, x_predict):
        x = add_bias_term(x_predict)
        return self.activation(x, self.weight)

    def evaluate(self, x_test, y_test):
        return self.metrics.eval(self.predict(x_test), y_test)	

# class LogisticReg(Model):
#     def __init__(self, optimizer=None):
#         self.optimizer = None
#         self.weight = None
#         self.metrics = ZeroOneError()
#         self.loss = ZeroOneError()
#         self.functional_form = lambda x, w: sign(x @ w) 
#         self.compile(optimizer)

#     def compile(self, optimizer):
#         self.optimizer = optimizer
        
#     def fit(self, x_train, y_train):
#         self.weight = self.optimizer.execute(x_train, y_train,
#                                             fn_form = self.functional_form,
#                                             loss = self.loss)

#     def predict(self, x_predict):
#         x = cons_augment(x_predict)
#         return self.functional_form(x, self.weight)

#     def evaluate(self, x_test, y_test):
#         return self.metrics.eval(self.predict(x_test), y_test)
