import numpy as numpy

from .. import losses
from ..base.models import Model
from ..utils.data_utils import check_data_consistent_length
from ..regression.models import LogisticRegression, LinearRegression, RidgeRegression
from ..regression.optimizers import AnalyticSolution, StocasticGradientDescent
from .meta_algorithms import OneVsOne, OneVsAll
from ..metrics import Accuracy
from ..regularizers import L2

class MulticlassLogisticRegression(Model):

    loss = losses.CrossEntropyError()

    def __init__(
        self,
        optimizer = StocasticGradientDescent(),
        meta_algorithm = OneVsAll(),
        metrics = [Accuracy()],
        regularizer = None
    ):
        self.compile(optimizer, meta_algorithm, metrics, regularizer)
    
    def compile(self, optimizer, meta_algorithm=None, metrics=None, regularizer=None):
        if metrics is not None: self.metrics = metrics
        if regularizer is not None: self.regularizer = regularizer

        self.optimizer = optimizer
        self.meta_classifier = meta_algorithm.__class__(
                                model = LogisticRegression(),
                                optimizer = self.optimizer,
                                metrics = self.metrics 
                            )
    
    def fit(self, x_train, y_train, **kwargs):
        self.meta_classifier.fit(x_train, y_train, **kwargs)
    
    def predict(self, x_predict):
        return self.meta_classifier.predict(x_predict)
    
    def evaluate(self, x_test, y_test):
        scores = []
        for metric in self.metrics:
            scores.append(metric.eval(y_test, self.predict(x_test)))
        return scores


class MulticlassLinearRegression(Model):
    
    loss = losses.MeanSquaredError()

    def __init__(
        self,
        optimizer = AnalyticSolution(),
        meta_algorithm = OneVsOne(),
        metrics = [Accuracy()],
        regularizer = None
    ):
        self.compile(optimizer, meta_algorithm, metrics, regularizer)
    
    def compile(self, optimizer, meta_algorithm=None, metrics=None, regularizer=None):
        if metrics is not None: self.metrics = metrics
        if regularizer is not None: self.regularizer = regularizer

        self.optimizer = optimizer
        self.meta_classifier = meta_algorithm.__class__(
                                model = LinearRegression(),
                                optimizer = self.optimizer,
                                metrics = self.metrics 
                            )
    
    def fit(self, x_train, y_train, **kwargs):
        self.meta_classifier.fit(x_train, y_train, **kwargs)
    
    def predict(self, x_predict):
        return self.meta_classifier.predict(x_predict)
    
    def evaluate(self, x_test, y_test):
        scores = []
        for metric in self.metrics:
            scores.append(metric.eval(y_test, self.predict(x_test)))
        return scores



class MulticlassRidgeRegression(Model):
    
    loss = losses.MeanSquaredError()
    regularizer = L2()

    def __init__(
        self,
        optimizer = AnalyticSolution(),
        meta_algorithm = OneVsOne(),
        metrics = [Accuracy()],
    ):
        self.compile(optimizer, meta_algorithm, metrics)
    
    def compile(self, optimizer, meta_algorithm=None, metrics=None):
        if metrics is not None: self.metrics = metrics

        self.optimizer = optimizer
        self.meta_classifier = meta_algorithm.__class__(
                                model = RidgeRegression(),
                                optimizer = self.optimizer,
                                metrics = self.metrics 
                            )
    
    def fit(self, x_train, y_train, **kwargs):
        self.meta_classifier.fit(x_train, y_train, **kwargs)
    
    def predict(self, x_predict):
        return self.meta_classifier.predict(x_predict)
    
    def evaluate(self, x_test, y_test):
        scores = []
        for metric in self.metrics:
            scores.append(metric.eval(y_test, self.predict(x_test)))
        return scores