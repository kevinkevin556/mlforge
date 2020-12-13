import abc
import numpy as np
import copy

from  .. import metrics
from ..base.meta_algorithms import MetaAlgorithm
from ..regression.models import LinearRegression, LogisticRegression
from ..utils.initialize_utils import new_instance
from ..utils.data_utils import re_weight_data
from ..utils.operation_utils import sign

"""
ensemble type    |  blending            learning
-----------------+-------------------------------------
uniform          |  voting/averaging    Bagging
non-uniform      |  linear              AdaBoost
conditional      |  stacking            Decision Tree
"""

# Blending base Class
class Blending(MetaAlgorithm, metaclass = abc.ABCMeta):
    def __init__(self, base_models=[], meta_model=None, **model_configs):
        self.compile(base_models = base_models, meta_model=meta_model, **model_configs)


    def compile(self, **model_config):
        super().compile(**model_config)
        if getattr(self, "meta_model", None) is not None:
            self.estimator_type = self.meta_model.__estimator_type__


    def add(self, model):
        self.base_models.append(model)

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass


# Blending Models

class Voting(Blending):

    meta_model = None

    def __init__(self, base_models=[], metrics=[metrics.Accuracy()]):
        self.compile(
            base_models = base_models,
            meta_model = self.meta_model,
            metrics = metrics
        )

    def fit(self, X, y):
        # Simply train each base models
        for model in self.base_models:
            model.fit(X, y)
        
        return self


    def predict(self, X):
        polls = np.empty((len(X), len(self.base_models)))
        y_predict = np.empty(len(X))

        # obtain prediction from each base models
        for i in range(len(self.base_models)):
            polls[:, i] = self.base_models[i].predict(X)

        # classification: find the class with the most polls 
        if self.estimator_type == "binary_classification":
            def find_max_count(x):
                value, count = np.unique(x, return_counts=True)
                return value[count.argmax()]
            y_predict = np.apply_along_axis(find_max_count, axis=1, arr=polls)
        
        # regression: average the predictions to get the final fitting value
        if self.estimator_type == "regression":
            y_predict = np.mean(polls, axis=1)
        
        return y_predict



class LinearBlending(Blending):

    meta_model = LinearRegression()

    def __init__(
        self,
        base_models = [],
        linear_config = {},
        metrics = [metrics.MeanSquaredError()]
    ):
        super().__init__(
            base_models = base_models,
            meta_model = new_instance(self.meta_model).set_params(**linear_config),
            metrics = metrics
        )


    def fit(self, X, y):
        meta_X = np.empty((len(X), len(self.base_models)))
        
        # Train each base models and transform the features
        # with fitted models
        for i in range(len(self.base_models)):
            self.base_models[i].fit(X, y)
            meta_X[:, i] = self.base_models[i].predict(X)
        
        # Train meta model(linear regression) to get the optimal
        # weights between each base model
        self.meta_model.fit(meta_X, y)

        return self


    def predict(self, X):
        polls = np.empty((len(X), len(self.base_models)))
        y_predict = np.empty(len(X))

        # Transform the features with fitted base models
        for i in range(len(self.base_models)):
            polls[:, i] = self.base_models[i].predict(X)

        # Predict the value based on the transformed features
        # which are prediction of base models
        y_predict = self.meta_model.predict(polls)
        
        return y_predict



class Stacking(Blending):

    def __init__(
        self,
        base_models = [],
        meta_model = LogisticRegression(),
        metrics = [metrics.Accuracy()]
    ):
        super().__init__(
            base_models = base_models,
            meta_model = meta_model,
            metrics = metrics
        )


    def fit(self, X, y):
        meta_X = np.empty((len(X), len(self.base_models)))

        # Train each base models and transform the features
        # with fitted models        
        for i in range(len(self.base_models)):
            self.base_models[i].fit(X, y)
            meta_X[:, i] = self.base_models[i].predict(X)

        # Train meta model to get the optimal weights
        # between each base model        
        self.meta_model.fit(meta_X, y)
    
        return self


    def predict(self, X):
        polls = np.empty((len(X), len(self.base_models)))
        y_predict = np.empty(len(X))

        # Transform the features with fitted base models
        for i in range(len(self.base_models)):
            polls[:, i] = self.base_models[i].predict(X)

        # Predict the value based on the transformed features
        # which are prediction of base models
        y_predict = self.meta_model.predict(polls)
        return y_predict




# ensemble Learning Models

class Bagging(MetaAlgorithm):
    def __init__(
        self, 
        estimator,
        n_estimators = 5, 
        metrics = []
    ):
        self.compile(estimator = estimator,
                     n_estimators = n_estimators,
                     metrics = metrics)
        
        self.data_oob_status_ = None
        self.oob_metrics_ = None


    def fit(self, X, y):
        data_is_in_bag = np.empty((len(X), self.n_estimators))

        for i in range(self.n_estimators):
            model = self.base_models[i] 
            random_idx = np.random.randint(low=0, high=len(X), size=len(X))  # Bootstrap
            data_is_in_bag[:, i] = np.isin(np.arange(len(X)), random_idx)

            sampled_X = X[random_idx, :]  # Trained with bootstrappeed data
            sampled_y = y[random_idx]
            self.base_models[i] = model.fit(sampled_X, sampled_y)

        self.data_oob_status_ = np.logical_not(data_is_in_bag)
        self.oob_metrics_ = self.evaluate_oob_metrics(X, y)
    
        return self
                   
        
    def predict(self, X):
        polls = np.empty((len(X), self.n_estimators))
        y_predict = np.empty(len(X))

        for i in range(self.n_estimators):
            polls[:, i] = self.base_models[i].predict(X)

        if self.estimator_type == "binary_classification":
            def find_max_count(x):
                value, count = np.unique(x, return_counts=True)
                return value[count.argmax()]
            y_predict = np.apply_along_axis(lambda x:find_max_count(x), 1, polls)
        
        if self.estimator_type == "regression":
            y_predict = np.mean(polls, axis=1)
        
        return y_predict


    def evaluate_oob_metrics(self, X, y):
        not_trained_with_data = self.data_oob_status_
        base_models = np.array(self.base_models, dtype=object)
        y = y.astype('float')
        y_pred_in_validation = np.empty(len(X))

        for i in range(X.shape[0]):
            validated_models = base_models[not_trained_with_data[i]]
            if len(validated_models) == 0:
                y_pred_in_validation[i] = np.nan
                y[i] = np.nan
            else:
                validated_bag = Bagging(estimator=self.estimator, n_estimators=len(validated_models))
                validated_bag.compile(base_models=validated_models)
                y_pred_in_validation[i] = validated_bag.predict(X[i, None])
        
        y = y[np.logical_not(np.isnan(y))]
        y_pred_in_validation = y_pred_in_validation[np.logical_not(np.isnan(y_pred_in_validation))]

        scores = []
        for metric in self.base_models[0].metrics:
            scores.append(metric.eval(y, y_pred_in_validation))
        return scores



class AdaBoost(MetaAlgorithm):
    def __init__(self, estimator, n_estimators=5, metrics=[]):
        
        self.compile(
            estimator = estimator,
            n_estimators = n_estimators,
            metrics = metrics
        )
        
        self.model_weight_ = None


    def fit(self, X, y):
        """
        u: trainging sample weight
        g: model
        alpha: model weight
        
        Ref: https://www.csie.ntu.edu.tw/~htlin/mooc/doc/208_handout.pdf (p17)
        """
        n_samples = X.shape[0]
        alpha = np.empty(self.n_estimators)               # alpha = model weight
        u = np.ones(n_samples) * (1/n_samples)   # u = sample weight

        for i in range(self.n_estimators):
            g = self.base_models[i]              # g = base models
            
            # (1) Train model with re-weighted training data
            #     The subscript "rw" refers to "re-weighted"
            if i == 0:
                X_rw, y_rw = X, y
            else:
                X_rw, y_rw = re_weight_data((X, y), weight=u, method="sampling")
            
            y_pred = g.fit(X_rw, y_rw).predict(X)
            eps = np.sum(u * np.not_equal(y_pred, y))/np.sum(u)     # weighted classification error
            
            # (2) Update sample weight with previous weighted clf error
            scaling_factor = np.sqrt((1-eps) / eps)
            u = u * (scaling_factor ** sign(np.not_equal(y_pred, y), zero=-1))    # if y_pred !=  y, u * scaling factor
                                                                                  # if  ...   == .., u / scaling factor
            alpha[i] = np.log(scaling_factor)   # and record the model weight meanwhile
        
        self.model_weight_ = alpha
        return self


    def predict(self, X):
        polls = np.empty((X.shape[0], self.n_estimators))
        
        for i in range(self.n_estimators):
            polls[:, i] = self.base_models[i].predict(X)
        
        return sign(polls @ self.model_weight_)