from mu.abstract import mutate

import abc
import bisect
import os
import math


__directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__directory, "", "12edo"), "r") as f:
    _12edo_freq = tuple(float(line[:-1]) for line in f.readlines())


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
    def hz2ct(freq0, freq1) -> float:
        return 1200 * math.log(freq1 / freq0, 2)

    def convert2midi_hex(self) -> tuple:
        """calculates the MIDI Tuning Standard of the pitch
        (http://www.microtonal-synthesis.com/MIDItuning.html)
        """
        freq = self.freq
        closest = bisect.bisect_right(_12edo_freq, freq) - 1
        diff = self.hz2ct(_12edo_freq[closest], freq)
        size0 = 0.78125
        size1 = 0.0061
        steps0 = int(diff // size0)
        steps1 = int((diff - (steps0 * size0)) // size1)
        assert steps0 < 128
        assert steps1 < 128
        return hex(closest), hex(steps0), hex(steps1)
