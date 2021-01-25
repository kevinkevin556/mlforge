import numpy as np

from .. import impurities, metrics
from ..ensemble.meta_algorithms import Bagging
from ..base.models import Model
from ..regression.models import LinearRegression
from ..utils.data_utils import permutate
from ..utils.decorator_utils import fit_method, predict_method
from ..utils.operation_utils import sign
from .optimizers import DecisionStumpSolver, CART, RandomTree


class DecisionStump(Model):
    __estimator_type__ = "binary_classification"

    def __init__(self, metrics=[metrics.Accuracy()]):
        self.feature_ = None
        self.threshold_ = None
        self.sign_ = 0
        
        self.compile(optimizer = DecisionStumpSolver(), metrics=metrics)     


    @fit_method
    def fit(self, X, y):
        self.sign_, self.feature_, self.threshold_ = self.optimizer.execute(X, y)
        return self


    @predict_method
    def predict(self, X):
        x = X[:, self.feature_] 
        theta = self.threshold_
        s = self.sign_
        return s * sign(x - theta)




class DecisionTree(Model):
    __estimator_type__ = ["binary_classification", "regression"]

    def __init__(
        self,
        criterion = impurities.GiniIndex(),
        max_height = np.inf,
        optimizer = CART(),
        metrics = [metrics.Accuracy()]
    ):
        self.__estimator_type__ = criterion.__problem_type__
        self.tree_ = None
        
        self.compile(criterion = criterion,
                     optimizer = optimizer,
                     max_height = max_height,
                     metrics = metrics)

    @fit_method
    def fit(self, X, y):
        self.tree_ = self.optimizer.execute(X, y)
        return self

    @predict_method
    def predict(self, X):
        return self.tree_.predict(X)




class RandomForest(Model):
    __estimator_type__ = ["binary_classification", "regression"]

    meta_algorithm = Bagging(DecisionTree())

    def __init__(
        self,
        n_estimators = 5,
        optimizer = RandomTree(tree=CART(), criterion=impurities.GiniIndex(), random_subspace=(0.5, )),
        metrics = [metrics.Accuracy()],
        **kwargs
    ):
        self.__estimator_type__ = optimizer.criterion.__problem_type__
        self.compile(
            n_estimators = n_estimators,
            optimizer = optimizer,
            metrics = metrics,
            **kwargs
        )

    @fit_method
    def fit(self, X, y, raw_importance=False):
        self.meta_algorithm = self.meta_algorithm.fit(X, y)
        self.feature_importances_ = self.eval_feature_importance(X, y, raw_importance)
        return self


    @predict_method
    def predict(self, X):
        return self.meta_algorithm.predict(X)


    def eval_feature_importance(self, X, y, raw=False):
        """Evaluate raw feature importance

        The instructive annotation comes from the referance of random forest
        by Leo Breiman and Adele Cutler.
        See: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#varimp


        Parameters
        ----------
        X: 2-D array  (n_samples, n_features)
            The feature part of the training data

        y: 1-D array like   (n_samples, )
            The value or label from the training data
        
        raw: bool, default=False
            If `raw==True`, use the votes cast for right
            class as the evaluating metric in classification
            problem. If `raw==False`, use accuracy.
        

        Return
        ------
        importances: 1-D array (n_features, )
            Feature importance evaluated based on out-of-bag data.
        """
        bag = self.meta_algorithm
        oob_to_model = self.meta_algorithm.data_oob_status_
        correct_votes = np.empty(self.n_estimators)
        permuted_correct_votes = np.empty((X.shape[1], self.n_estimators))
        
        if self.__estimator_type__ == "binary_classification":
            if raw:
                metric = lambda y_pred, y_true: sum(y_pred == y_true)
            else:
                metric = metrics.Accuracy()
        elif self.__estimator_type__ == "regression":
            metric = metrics.R2()
        else:
            raise ValueError("Invalid __estimator_type__")
            
        # (1) In every tree grown in the forest, put down the oob cases
        # and evaluate the metrics (accuracy for classification and r^2 for
        # regression). 
        # The author use votes for the correct as the metric (raw == True),
        # so we use "correct_votes" the variable name to match the instruction.
        for i in range(self.n_estimators):
            X_oob = X[oob_to_model[:, i], :]
            y_oob = y[oob_to_model[:, i]]
            y_pred = bag.base_models[i].predict(X_oob)
            correct_votes[i] = metric(y_oob, y_pred)

            # (2) Now randomly permute the values of variable m
            # in the oob cases and put these cases down the tree.
            for m in range(X.shape[1]):
                X_oob_m_permuted = permutate(X_oob, feature_idx=m)
                y_pred_m_permuted = bag.base_models[i].predict(X_oob_m_permuted)
                permuted_correct_votes[m, i] = metric(y_oob, y_pred_m_permuted)
        
        # (3) Subtract the metric evaluated in the variable-m-permuted
        #  oob data from the metric evaluated in the untouched oob data.
        performance_in_trees = correct_votes - permuted_correct_votes        

        # (4) The average of this number over all trees in the forest
        # is the importance score for variable m.
        importances = np.mean(performance_in_trees, axis=1)

        return importances



class GradientBoostedDecisionTree(Model):
    __estimator_type__ = "regression"

    def __init__(
        self,
        n_estimators = 5,
        max_height = 5,
        metrics = [metrics.MeanSquaredError()]
    ):
        self.model_weight_ = np.empty(0)
        self.base_regression_trees_ = [DecisionTree(criterion = impurities.MeanSquaredError(),
                                                   max_height = max_height,
                                                   metrics = metrics) for _ in range(n_estimators)]
        self.compile(
            n_estimators = n_estimators,
            max_height = max_height,
            metrics = metrics
        )

    @fit_method
    def fit(self, X, y):

        residuals_i = y       
        for i in range(self.n_estimators):
            # (1) obtain base-model_i by A({(xn, yn − sn)}) where `A` 
            #     is regression tree algorithm, `sn` are predicting
            #     values evaluated by current boosting model. Therefore, 
            #     `yn - sn` are the residuals.
            model_i = self.base_regression_trees_[i].fit(X, residuals_i)
            y_pred_i = model_i.predict(X)[:, None]
            
            # (2) compute the model weight for model_i by regressing residuals
            #     `yn − sn` on the predicting values of current model `gt(xn)`.
            #     Note that it is an "single variable linear regression".
            coef_i = LinearRegression().fit(y_pred_i, residuals_i).weight_[1]
            self.model_weight_ = np.append(self.model_weight_, coef_i)

            # (3) After the update of model weight has been done, compute the
            #     prediction based on the updated boosting model, and then
            #     substract the prediction from y to set new residuals for
            #     the next round of training.
            residuals_i = y - self.predict(X)

        return self

    @predict_method
    def predict(self, X):
        
        n_estimators = len(self.model_weight_)
        polls = np.empty((X.shape[0], n_estimators))
        
        for i in range(n_estimators):
            polls[:, i] = self.base_regression_trees_[i].predict(X)
        
        return polls @ self.model_weight_
