from mu.abstract import mutate

import abc
import math


def is_private(string: str) -> bool:
    if string[:2] == "__" or string[:1] == "_":
        return True
    else:
        return False


class AbstractPitch(abc.ABC):
    @abc.abstractmethod
    def calc(self) -> float:
        raise NotImplementedError

    @property
    def freq(self) -> float:
        return self.calc()

    @classmethod
    def mk_iterable(cls, template) -> abc.ABCMeta:
        def adapt_result(self, cls, res):
            if (cls,) * len(res) == tuple(map(lambda x: type(x), res)):
                return type(self)(res)
            elif (None,) * len(res) != res:
                return res
            else:
                return None

        def method_decorator(func):
            def wrap(*args, **kwargs):
                res = tuple(
                    mutate.execute_method(f, func, args[1:], kwargs)
                    for f in args[0]
                    if f is not None
                )
                return adapt_result(args[0], cls, res)

            return wrap

        def property_decorator(func):
            def wrap(*args, **kwargs):
                self = args[0]
                res = tuple(func.fget(f) for f in self)
                return adapt_result(self, cls, res)

            return property(wrap)

        c_name = "{0}_{1}".format(cls.__name__, template.__name__)
        bases = (template,)
        keys = [function for function in dir(cls) if not is_private(function)]
        functions = [getattr(cls, k) for k in keys]
        old_method = tuple(
            (key, func) for key, func in zip(keys, functions) if callable(func)
        )
        old_property = tuple(
            (key, func) for key, func in zip(keys, functions) if type(func) == property
        )
        methods = {key: method_decorator(key) for key, func in old_method}
        properties = {key: property_decorator(func) for key, func in old_property}
        return type(c_name, bases, {**methods, **properties})

    def __eq__(self, other: "AbstractPitch") -> bool:
        try:
            return self.freq == other.freq
        except AttributeError:
            return False

    def __lt__(self, other: "AbstractPitch") -> bool:
        return self.freq < other.freq

    def __gt__(self, other: "AbstractPitch") -> bool:
        return self.freq > other.freq

    def __hash__(self) -> int:
        return hash(self.freq)

    @staticmethod
    def hz2ct(freq0, freq1):
        return 1200 * math.log(freq1 / freq0, 2)
