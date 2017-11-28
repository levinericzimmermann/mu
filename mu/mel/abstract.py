from abc import ABC, abstractmethod, ABCMeta


def is_private(string: str) -> bool:
    if string[:2] == "__" or string[:1] == "_":
        return True
    else:
        return False


class AbstractTone(ABC):
    @abstractmethod
    def calc(self) -> float:
        raise NotImplementedError

    @property
    def freq(self) -> float:
        return self.calc()

    @classmethod
    def mk_iterable(cls, template) -> ABCMeta:
        def adapt_result(self, cls, res):
            if (cls,) * len(res) == tuple(map(lambda x: type(x), res)):
                return type(self)(res)
            elif (None,) * len(res) != res:
                return res
            else:
                return None

        def method_decorator(func):
            def wrap(*args, **kwargs):
                def execute(f, args, kwargs):
                    if args and kwargs:
                        return getattr(f, func)(*args, **kwargs)
                    elif args and not kwargs:
                        return getattr(f, func)(*args)
                    elif kwargs and not args:
                        return getattr(f, func)(**kwargs)
                    else:
                        return getattr(f, func)()
                res = tuple(execute(f, args[1:], kwargs) for f in args[0])
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
        old_method = tuple((key, func) for key, func in zip(keys, functions)
                           if callable(func))
        old_property = tuple((key, func) for key, func in zip(keys, functions)
                             if type(func) == property)
        methods = {key: method_decorator(key) for key, func in old_method}
        properties = {key: property_decorator(func)
                      for key, func in old_property}
        return type(c_name, bases, {**methods, **properties})

    def __eq__(self, other):
        try:
            return self.freq == other.freq
        except AttributeError:
            return False

    def __hash__(self):
        return hash(self.freq)

    @staticmethod
    def hz2ct(freq):
        pass

    @staticmethod
    def ct2hz(ct):
        pass


class AbstractMelody(list):
    def reverse(self):
        return type(self)(reversed(self))

    def __and__(self, other):
        """merge two Melody-objects"""
        return type(self)(list.__add__(self, other))

    def __hash__(self):
        return hash(tuple(hash(t) for t in self))

    def __getitem__(self, idx):
        if type(idx) == slice:
            return type(self)(list.__getitem__(self, idx))
        else:
            return list.__getitem__(self, idx)


class AbstractHarmony(set):
    def __or__(self, other: "JIMel") -> "JIMel":
        """merge two Harmony-objects"""
        return type(self)(set.__or__(self, other))

    def __hash__(self):
        return hash((hash(t) for t in self))
