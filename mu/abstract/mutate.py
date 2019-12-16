import functools
from typing import Any, Dict, Tuple


def execute_method(cls: Any, method: str, args: Tuple, kwargs: Dict) -> Any:
    return getattr(cls, method)(*args, **kwargs)


def mutate_class(cls):
    def adapt_result(self, cls, res):
        if type(res) == cls:
            return type(self)(res)
        else:
            return res

    def method_decorator(func):
        def wrap(*args, **kwargs):
            return adapt_result(
                    args[0], cls,
                    execute_method(cls, func, args, kwargs))
        return wrap

    def property_decorator(func):
        def wrap(*args, **kwargs):
            self = args[0]
            return adapt_result(self, cls, func.fget(self))
        return property(wrap)

    def getitem(self, idx):
        try:
            length = len(self)
            indices = idx.indices(length)
            copied = self.copy()
            start, stop = indices[0], indices[1]
            del_end = length - stop
            if del_end > 0:
                for i in range(del_end):
                    del copied[-1]
            del_start = start
            if del_start > 0:
                for i in range(del_start):
                    del copied[0]
            return copied
        except AttributeError:
            return cls.__getitem__(self, idx)

    c_name = "mu_{0}".format(cls.__name__)
    new_class = type(c_name, (cls,), {})

    forbidden = ("__init__", "__class__", "__init_subclass__")
    keys = [function for function in dir(new_class) if not functools.reduce(
        lambda x, y: x or y, (function == d for d in forbidden))]
    functions = [getattr(new_class, k) for k in keys]
    old_method = tuple((key, func) for key, func in zip(keys, functions)
                       if callable(func))
    old_property = tuple((key, func) for key, func in zip(keys, functions)
                         if type(func) == property)
    methods = {key: method_decorator(key) for key, func in old_method}
    properties = {key: property_decorator(func)
                  for key, func in old_property}
    for k in methods:
        if k == "__getitem__":
            setattr(new_class, k, getitem)
        else:
            setattr(new_class, k, methods[k])
    for k in properties:
        setattr(new_class, k, property(methods[k]))
    return new_class
