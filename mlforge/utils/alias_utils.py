alias_dict = {}


def register(alias_str, obj_class):
    global alias_dict
    class_name  = obj_class.__name__
    class_dict = {class_name: obj_class}

    if not alias_dict.get(alias_str, False): 
        alias_dict[alias_str] = class_dict
    else:
        alias_dict[alias_str].update(class_dict)
        

def add_alias(alias, obj_class):
    if type(alias) == str:
        register(alias, obj_class)
    if type(alias) in [list, tuple]:
        for a in alias:
            register(a, obj_class)


def init_with_alias(alias, obj_class=None, **kwargs):
    if obj_class is None:
        class_func = alias_dict[alias]
        if len(class_func.values) > 1:
            raise ValueError("obj_class should be assigned.")
        else:
            return class_func(**kwargs)
    else:
        class_func = alias_dict[alias][obj_class]
        return class_func(**kwargs)