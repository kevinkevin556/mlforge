import abc
import inspect
from copy import deepcopy

from ..utils.print_utils import format_repr

class MetaAlgorithm(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self,
        estimator = None,  
        n_estimators = 0, 
        base_models = [],
        **model_config
    ):
        self.estimator_type = None
        self.estimator = estimator
        self.n_estimators = n_estimators

        self.base_models = base_models
        self.compile(**model_config)


    @abc.abstractmethod
    def fit(self, X, y):
        pass
    

    @abc.abstractmethod
    def predict(self, X):
        pass


# keras-like API #

    def compile(self, estimator=None, n_estimators=None, base_models=None, **model_config):
        if (estimator and base_models) or (n_estimators and base_models):
            raise ValueError("Do not set estimator/n_estimators and base_models at the same time.")

        # Set meta-algorithm attributes and  Initiate base models
        if base_models is not None:
            self.base_models = base_models
            ### TODO: set estimator and n_estimator###

        elif (estimator is not None) or (n_estimators is not None):
            if estimator is not None:
                self.estimator = estimator
            if  n_estimators is not None:
                self.n_estimators = n_estimators
            self.base_models = [deepcopy(self.estimator) for _ in range(self.n_estimators)]

        else:
            pass

        # Set base models attributes and collect model types
        types = []
        for model in self.base_models:
            model.compile(**model_config)

            if type(model.model_type) == str:
                types.append(model.model_type)
            elif hasattr(model.model_type, "__iter__"):
                types = types + list(model.model_type)
            else:
                raise ValueError("Invalid model_type")
        
        # Set meta-algorithm estimator_type 
        types = set(types)
        if len(types) == 0:
            self.estimator_type = "Undefined"
        elif len(types) == 1:
            self.estimator_type = types.pop()
        else:
            msg = "The applied models in the meta-algorithm should own identical model_type."
            raise Exception(msg)

# Scikit-learn compatible API #

    def set_params(self, **parameters):
        self.compile(**parameters)
        return self


    def get_params(self, deep=False):
        init_sig = inspect.signature(self.__init__)
        param_names = [param.name for param in init_sig.parameters.values()
                                  if  param.name != 'self' and 
                                      param.kind != param.VAR_KEYWORD ]
        output = {}
        for p_name in param_names:
            if p_name in dir(self):
                p_object = getattr(self, p_name)
            
                if deep and getattr(p_object, "get_params", None):
                    p_object_params = p_object.get_params()
                    output.update({p_name + "_" + key: value for key, value in p_object_params.items()})
                
                output[p_name] = p_object
        return output

    
    def __repr__(self):
        return format_repr(self.__class__.__name__, self.get_params())