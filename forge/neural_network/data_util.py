import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pyfoo.base.exception import * 


class OperationBase(object):
    
    inverse_dict = {}
    
    def __init__(self):
        pass
    
    @classmethod
    def operation(cls, func):
        cls.inverse_dict[func.__name__] = None
        def rdef_wrapper(self, *args):
            self.opr_seq.append((func.__name__, args))
            func(self, *args)
        return rdef_wrapper
    
    @classmethod
    def inverse(cls, opr):
        def pre_exec_wrapper(func):
            cls.inverse_dict[opr] = func
            return func
        return pre_exec_wrapper
           

class DataOperation(OperationBase):

    # operation decorators
    operation = OperationBase.operation    
    inverse = OperationBase.inverse
    
    def __init__(self):
        pass
  
    @operation
    def append(self, x):
        self.data.append(x)
    
    @inverse('append')
    def pop(self, x):
        self.data.pop()

class UsrOperation(OperationBase):
   
    #operation decorators
    operation = OperationBase.operation    
    inverse = OperationBase.inverse
    
    def __init__(self):
        pass

    
class DataContext(DataOperation, UsrOperation):
    
    def __init__(self, data):
        self.data = data
        self.opr_seq = []
    
    def inv(self, opr_arg_tuple):
        opr = opr_arg_tuple[0]
        arg = opr_arg_tuple[1]
        invs_opr = self.inverse_dict.get(opr)
        invs_opr(self, arg)
    
    def clean(self):
        for opr in self.opr_seq[::-1]:
            self.inv(opr)
    
    def undo(self):
        self.inv(self.opr_seq[-1])
        self.opr_seq.pop()
    
    def history(self):
        for i in range(len(self.opr_seq)):
            opr_arg = self.opr_seq[i]
            output = "[{}]    {}{}".format(i, opr_arg[0], opr_arg[1])
            print(output)
  
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exec_value, traceback):
        self.clean()

def using(data):
    return DataContext(data)