import numpy as np
import copy

from .meta_algorithms import Bagging, AdaBoost, Voting

from ..metrics import Accuracy

from ..base.models import Model
from ..perceptron.models import Perceptron
from ..perceptron.optimizers import Pocket
from ..regression.models import LogisticRegression
from ..bayes.models import NaiveBayes

from ..tree.models import DecisionStump, RandomForest
from ..utils.operation_utils import sign
from ..utils.initialize_utils import new_instance
from ..utils.decorator_utils import fit_method, predict_method



class BaggingPerceptron(Model):

    __estimator_type__ = "binary_classification"

    meta_algorithm = Bagging(estimator=Perceptron())

    def __init__(
        self,
        n_estimators = 10,
        optimizer = Pocket(),
        metrics = [Accuracy()]
    ):  
        self.compile(n_estimators=n_estimators, optimizer=optimizer, metrics=metrics)

    @fit_method
    def fit(self, X, y):
        self.meta_algorithm = self.meta_algorithm.fit(X, y)
        return self

    @predict_method
    def predict(self, X):
        return self.meta_algorithm.predict(X)



class AdaBoostStump(Model):
    meta_algorithm = AdaBoost(estimator=DecisionStump())

    def __init__(
        self,
        n_estimators=5,
        metrics=[Accuracy()]
    ):
        self.compile(n_estimators=n_estimators, metrics=metrics)

    @fit_method
    def fit(self, X, y):
        self.meta_algorithm = self.meta_algorithm.fit(X, y)
        return self
    
    @predict_method
    def predict(self, X):
        return self.meta_algorithm.predict(X)



class VotingClassifier(Model):
    meta_algorithm = Voting()

    def __init__(
        self,
        base_models=[LogisticRegression(), NaiveBayes(), RandomForest()],
        metrics=[Accuracy()]
    ):
        self.compile(base_models=base_models, metrics=metrics)

    @fit_method
    def fit(self, X, y):
        self.meta_algorithm = self.meta_algorithm.fit(X, y)
        return self
    
    @predict_method
    def predict(self, X):
        return self.meta_algorithm.predict(X)