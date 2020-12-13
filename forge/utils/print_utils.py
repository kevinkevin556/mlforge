import re

def format_repr(cls_name, config_dict):
    repr_format = "{}({})"
    config_str = re.sub(r"\'([a-zA-Z_][a-zA-Z_0-9]*)\'\: ", 
                        r"\1=",
                        str(config_dict)[1:-1])
    return repr_format.format(cls_name, config_str)