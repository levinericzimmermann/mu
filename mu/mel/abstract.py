from mu.abstract import mutate

import abc
import bisect
import math
import operator
import os
import warnings

try:
    import quicktions as fractions
except ImportError:
    import fractions


__directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__directory, "", "12edo"), "r") as f:
    _12edo_freq = tuple(float(line[:-1]) for line in f.readlines())


def is_private(string: str) -> bool:
    if string[:2] == "__" or string[:1] == "_":
        return True
    else:
        return False


class AbstractPitch(abc.ABC):
    _cent_calculation_constant = 1200 / (math.log10(2))
    _midi_tuning_table0 = tuple(i * 0.78125 for i in range(128))
    _midi_tuning_table1 = tuple(i * 0.0061 for i in range(128))

    @abc.abstractmethod
    def calc(self) -> float:
        raise NotImplementedError

    @property
    def freq(self) -> float:
        return self.calc()

    @abc.abstractproperty
    def cents(self) -> float:
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        """Return True if pitch equals a Rest. Otherwise False."""
        return False

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

    def __ge__(self, other: "AbstractPitch") -> bool:
        return self.freq >= other.freq

    def __le__(self, other: "AbstractPitch") -> bool:
        return self.freq <= other.freq

    def __hash__(self) -> int:
        return hash(self.freq)

    @staticmethod
    def hz2ct(freq0: float, freq1: float) -> float:
        return 1200 * math.log(freq1 / freq0, 2)

    @staticmethod
    def ratio2ct(ratio: fractions.Fraction) -> float:
        return AbstractPitch._cent_calculation_constant * math.log10(ratio)

    @staticmethod
    def ct2ratio(ct: float) -> fractions.Fraction:
        return fractions.Fraction(10 ** (ct / AbstractPitch._cent_calculation_constant))

    def convert2midi_tuning(self) -> tuple:
        """calculates the MIDI Tuning Standard of the pitch

        (http://www.microtonal-synthesis.com/MIDItuning.html)
        """

        def detect_steps(difference):
            def find_lower_and_higher(table, element):
                closest = bisect.bisect_right(table, element)
                if closest < len(table):
                    indices = (closest - 1, closest)
                    differences = tuple(abs(element - table[idx]) for idx in indices)
                else:
                    idx = closest - 1
                    difference = abs(table[idx] - element)
                    indices, differences = (idx, idx), (difference, difference)
                return tuple(zip(indices, differences))

            closest_s0 = find_lower_and_higher(
                AbstractPitch._midi_tuning_table0, difference
            )
            closest_s1 = find_lower_and_higher(
                AbstractPitch._midi_tuning_table1, closest_s0[0][1]
            )
            closest_s1 = min(closest_s1, key=operator.itemgetter(1))
            difference0 = closest_s1[1]
            difference1 = closest_s0[1][1]
            if difference0 <= difference1:
                return closest_s0[0][0], closest_s1[0], difference0
            else:
                return closest_s0[1][0], 0, difference1

        freq = self.freq
        if freq:
            closest = bisect.bisect_right(_12edo_freq, freq) - 1
            diff = self.hz2ct(_12edo_freq[closest], freq)
            steps0, steps1, diff = detect_steps(diff)
            if diff >= 5:
                msg = "Closest midi-pitch of {0} ({1} Hz) ".format(self, freq)
                msg += "is still {0} cents apart!".format(diff)
                warnings.warn(msg)
            return closest, steps0, steps1
        else:
            return tuple([])
