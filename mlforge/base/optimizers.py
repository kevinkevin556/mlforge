import abc
import inspect

from ..utils.print_utils import format_repr

class Optimizer(metaclass = abc.ABCMeta):

    def __init__(self):
        pass
    
    @abc.abstractmethod
    def execute(self, x, y):
        """
        Algorithm implimentaation
        
        input: 2d-array, 1d-array 
        output: 1d-array  
        """
        pass 


# keras-like api 
 
    def get_config(self):
        """Returns the config of the optimizer.

        An optimizer config is a dictionary containing
        the configuration of an optimizer. The same
        optimizer can be reinstantiated from this
        configuration.
        
        Returns
        -------
        output: dict
            optimizer configuration
        """
        init_sig = inspect.signature(self.__init__)
        param_names = [param.name for param in init_sig.parameters.values()
                                  if  param.name != "self" and 
                                      param.kind != param.VAR_KEYWORD ]
        output = {}
        for p_name in param_names:
            p_object = getattr(self, p_name)
            output[p_name] = p_object
        return output


    @classmethod
    def from_config(cls, config=None, **kw_config):
        """Creates an optimizer from its config.
        
        This method is the reverse of `get_config`, capable of instantiating
        the same optimizer from the config dictionary.
                
        Parameters
        ----------
        config: dict
            typically the output of get_config.
        
        kw_config: 
            keywords argumemts to set configuration.

        Returns
        -------
        output: Optimizer
            An optimizer instance.
        """
        if config is None:
            config = {}
        if type(config) is not dict:
            raise ValueError("Invalid type for config. config should be a dcitionary.")
        
        config.update(kw_config)
        return cls(**config)


    def __repr__(self):
        return format_repr(self.__class__.__name__, self.get_config())


