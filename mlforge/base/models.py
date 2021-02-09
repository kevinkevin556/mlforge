import abc
import inspect
from copy import deepcopy

from .. import metrics
from .optimizers import Optimizer

from ..utils.data_utils import add_cons
from ..utils.print_utils import format_repr


class Model(metaclass = abc.ABCMeta):
    """ Base class of model to be used as a reference implementation.

    Parameters
    ----------
    optimizer: optimizer instance
        An optimizer is the implementation of the algorithm that
        minimizes the loss of the model.
    
    metrics: list of metric instance
        List of metrics to be evaluated by the model during testing. 
        Typically you will use `metrics=['accuracy']`.
    """

# class attributes #

    # Scikit-learn Estimator Tags
    model_type = None

    # Model settings
    loss = None

# API which needs implementation #

    @abc.abstractmethod
    def __init__(self, optimizer=None, metrics=[]):
        """A reference implementation of `__init__` function.
        
        There are 3 kinds of variables in a model.
        
        1. Model setting
        Model setting are the charaterics that determine 
        which model it is and how the model can be trained.
        The common settings are components of the argumented
        error -- loss and regularizer, and meta-algorithm for
        multiclass models or ensemble models. 
        
        Model settings should be declared as class attributes
        since they are shared by models of same type.


        2. Parameter
        Parameters are variables sended to the model initializer
        in order to specify the way to train the model and how 
        much regularization should be applied when undergoing 
        a training process.

        Parameters should be specified in the parameter list of 
        `__init__()` and created through `__compile()` by passing 
        them as `keyword-parameters`.


        3. Model (fitted) attributes
        Fitted attributes show that model is trained and present the
        result fitted with the current training data. 

        Typically model fitted attributes are assigned as `None` in
        `__init__()` and modified when `fit()` is called. The naming
        of fitted attributes should have trailing underscore`_`, which
        adheres to scikit-learn naming convention.
        """
        self.weight_ = None
        self.compile(optimizer=optimizer, metrics=metrics)


    @abc.abstractmethod
    def fit(self, X, y, **kwargs):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        
        **kwargs: optional data-dependent parameters


        Returns
        -------
        self : object
            Returns self.
        """
		
		# (1) Execute any particular optimizer to fit data
        self.weight = self.optimizer.execute(X, y, loss=Model.loss)
        
        # (2) Or a extensive way to call optimizer 
        #     when there are many of them with different arguments
        kwargs["X"],  kwargs["y"], kwargs["loss"]= X, y, Model.loss
        varnames = self.optimizer.execute.__code__.co_varnames
        using_kwargs = {v:kwargs.get(v, None) for v in varnames}
        self.weight = self.optimizer.execute(**using_kwargs)

        # `fit` should always return `self`, which makes pipeline easiler
        return self


    @abc.abstractmethod
    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The predciting input samples.

        Returns
        -------
        output : array-like, shape (n_samples,)
            The predicted values (class labels in classification, real 
            numbers in regression).
        """
        
        # Bias terms are commonly added before predicting
        x = add_cons(X)
        output = x @ self.weight_
        return output

    
# keras-like API #

    def compile(self, optimizer="None", metrics="None", **parameters):
        # Set optimizer and metrics
        if optimizer  != "None":
            self.optimizer = optimizer
        if metrics != "None":
            self.metrics = metrics
        
        # Set other model parameters
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        # If the model is an ensemble model,
        # create an copy of class meta_algorithm in object attributes
        if getattr(self.__class__, "meta_algorithm", None):
            setattr(self, "meta_algorithm", deepcopy(self.__class__.meta_algorithm))

        # Once you have set up optimizer,
        # if the parameter of model is also a parameter of the optimizer,
        # set it in the optimizer instance
        if getattr(self, "optimizer", None) and isinstance(self.optimizer, Optimizer):
            opt_params = self.optimizer.__init__.__code__.co_varnames
            for parameter, value in parameters.items():
                if parameter in opt_params:
                    setattr(self.optimizer, parameter, value)

        # If optimizer(s) are assigned in two-level training,
        # set the parameters iteratively.
        if getattr(self, "optimizer", None) and isinstance(self.optimizer, (list, tuple)):
            for opt in self.optimizer:
                if isinstance(opt, Optimizer):
                    opt_params = opt.__init__.__code__.co_varnames
                    for parameter, value in parameters.items():
                        if parameter in opt_params:
                            setattr(opt, parameter, value) 

        # If the model includes meta_algorithm attributes,
        # set the corresponding parameters in meta_algorithm as well
        parameters["optimizer"] = optimizer
        parameters["metrics"] = metrics
        if getattr(self, "meta_algorithm", None):
            meta_algo_params = self.meta_algorithm.__init__.__code__.co_varnames
            ma_kwargs = {ma_param: parameters[ma_param] for ma_param in meta_algo_params
                                             if ma_param in parameters}
            self.meta_algorithm.compile(**ma_kwargs)
            self.model_type = self.meta_algorithm.estimator_type



    def evaluate(self, X=None, y=None, oob=False):
        if oob and getattr(self, "meta_algorithm", None):
            return self.meta_algorithm.oob_metrics_
        else:
            scores = []
            for metric in self.metrics:
                scores.append(metric.eval(self.predict(X), y))
            return scores


# Scikit-learn compatible API #

    def set_params(self, **parameters):
        self.compile(**parameters)
        return self


    def get_params(self, deep=True):
        init_sig = inspect.signature(self.__init__)
        param_names = [param.name for param in init_sig.parameters.values()
                                  if  param.name != 'self' and 
                                      param.kind != param.VAR_KEYWORD ]
        output = {}
        for p_name in param_names:
            p_object = getattr(self, p_name)
            
            if deep and getattr(p_object, "get_params", None):
                p_object_params = p_object.get_params()
                output.update({p_name + "_" + key: value for key, value in p_object_params.items()})
            
            output[p_name] = p_object
        return output


    def score(self, X, y):
        if self.model_type == "binary-classifier":
            metric = metrics.Accuracy()
        if self.model_type == "regressor":
            metric = metrics.R2()
        return metric.eval(y, self.predict(X))
    
    
    def __repr__(self):
        return format_repr(self.__class__.__name__, self.get_params())