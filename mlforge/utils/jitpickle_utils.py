from numba.experimental import jitclass
import pickle
from functools import wraps, update_wrapper

from .print_utils import format_repr

jitclasses = {}

class jitpickler:
    '''
    pickler
    '''
    def __init__(self, jitobj):
        self.__dict__['obj'] = jitobj
        self.__dict__['__module__'] = jitobj.__module__
        self.__dict__['__doc__'] = jitobj.__doc__

    def __getstate__(self):
        obj = self.__dict__['obj']
        typ = obj._numba_type_
        fields = typ.struct

        return typ.classname, {k: getattr(obj, k) for k in fields}

    def __setstate__(self, state):
        name, value = state
        cls = jitclasses[name]
        value['_decorator'] = False
        jitobj = cls(**value)
        self.__init__(jitobj)

    def __getattr__(self, attr):
        return getattr(self.__dict__['obj'], attr)

    def __setattr__(self, attr, value):
        return setattr(self.__dict__['obj'], attr, value)

    def __delattr__(self, attr):
        return delattr(self.__dict__['obj'], attr)
    
    def get_config(self):
        obj = self.__dict__["obj"]
        typ = obj._numba_type_
        fields = typ.struct
        if getattr(self.obj, "get_config", None):
            return {k: getattr(obj, k) for k in fields}
        else:
            error_msg = "'{}' object has no attribute 'get_config'"
            raise AttributeError(msg.format(typ.classname))
    
    def __repr__(self):
        obj = self.__dict__["obj"]
        typ = obj._numba_type_
        if getattr(self.obj, "get_config", None):
            return format_repr(typ.classname, self.get_config())
        else:
            obj.__repr__()



def jitpickle(cls):
    decoratorkw = '_decorator'
    
    @wraps(cls)
    def decorator(*args, **kwargs):
        if kwargs.get(decoratorkw, True):
            kwargs.pop(decoratorkw, None)
            return jitpickler(cls(*args, **kwargs))
        else:
            kwargs.pop(decoratorkw, None)
            return cls(*args, **kwargs)
    
    global jitclasses
    jitclasses[cls.class_type.class_name] = decorator
    return decorator


def is_instance(pickler_obj, pickled_type):
    if not isinstance(pickler_obj, jitpickler):
        raise TypeError("is_instance() arg 1 must be a jitpickler object")

    cls = pickled_type.class_type.class_name
    instance_type = pickler_obj.__dict__['obj']._numba_type_.classname

    if instance_type == cls:
        return True
    else:
        return False
    