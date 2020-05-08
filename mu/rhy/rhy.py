from mu.abstract import muobjects
from mu.time import time
from mu.utils import prime_factors

import abc
import functools
import itertools
import operator
from typing import Union


class AbstractRhythm(abc.ABC):
    """A Rhythm may be a structure to organise time."""

    @abc.abstractmethod
    def flat(self):
        raise NotImplementedError

    @abc.abstractproperty
    def delay(self):
        raise NotImplementedError

    @abc.abstractproperty
    def stretch(self, arg):
        raise NotImplementedError


class Unit(AbstractRhythm, time.Time):
    def __repr__(self):
        return "Unit({})".format(str(self))

    def flat(self):
        return Compound((self,))

    @property
    def delay(self):
        return time.Time(self)

    def stretch(self, arg):
        return type(self)(self * arg)

    def copy(self):
        return type(self)(self)


class Compound(AbstractRhythm, muobjects.MUList):
    def __init__(self, iterable):
        iterable = list(iterable)
        for i, n in enumerate(iterable):
            if not isinstance(n, AbstractRhythm):
                try:
                    iterable[i] = Unit(n)
                except TypeError:
                    raise ValueError("Unknown element {0}.".format(n))
        muobjects.MUList.__init__(self, iterable)

    def flat(self):
        if self:
            return functools.reduce(
                lambda x, y: x + y, tuple(u.flat() for u in self))
        else:
            return type(self)([])

    @property
    def delay(self):
        return time.Time(sum(u.delay for u in self))

    def stretch(self, arg):
        return type(self)(tuple(u.stretch(arg) for u in self))

    def append(self, arg: Union[int, Unit]) -> None:
        if id(arg) == id(self):
            arg = arg.copy()
        list.append(self, arg)

    def convert2absolute(self, skiplast=True):
        new = self.copy()
        d = 0
        for i, r in enumerate(self):
            new[i] = type(new[i])(d)
            d += r.delay
        if skiplast is False:
            new.append(type(new[0])(d))
        return new

    def convert2relative(self):
        new = self.copy()[:-1]
        for i, r0, r1 in zip(range(len(self)), self, self[1:]):
            diff = r1 - r0
            new[i] = type(new[i])(diff)
        return new


class PulseChroma(int):
    def __init__(self, length: int):
        muobjects.MUInt.__init__(length)
        primes = prime_factors.factorise(length)
        length_primes = len(primes)
        if length_primes > 1:
            primes_unique = tuple(set(primes))
            if length_primes == len(primes_unique):
                combinations_num = len(primes_unique) - 1
                subpulse = itertools.combinations(primes, combinations_num)
                subpulse = (PulseChroma(functools.reduce(operator.mul, sub))
                            for sub in subpulse)
                self.subpulse = tuple(subpulse)
            else:
                subpulse = functools.reduce(operator.mul, primes_unique)
                self.subpulse = (PulseChroma(subpulse),)
        else:
            if length != 1:
                self.subpulse = PulseChroma(1),
            else:
                self.subpulse = 0,

    def __hash__(self):
        return hash((int(self), self.subpulse))

    def __eq__(self, other):
        try:
            sub0 = set(self.subpulse)
            sub1 = set(other.subpulse)
            return int.__eq__(self, other) and sub0 == sub1
        except AttributeError:
            return False

    def count_subpulse(self):
        try:
            return tuple(self / sub for sub in self.subpulse)
        except ZeroDivisionError:
            return 0,

    def specify(self):
        try:
            subpath = tuple(sub.specify() for sub in self.subpulse)
            subpath = tuple(functools.reduce(operator.add, subpath))
        except AttributeError:
            return SpecifiedPulseChroma(self, 0),
        return tuple(SpecifiedPulseChroma(self, sub) for sub in subpath)


class SpecifiedPulseChroma(int):
    def __init__(self, length, subpulse):
        muobjects.MUInt.__init__(length)
        self.subpulse = subpulse

    def __new__(cls, *args, **kwargs):
        return muobjects.MUInt.__new__(cls, *args[:-1], **kwargs)

    def __eq__(self, other):
        try:
            test0 = int.__eq__(self, other)
            test1 = self.subpulse == other.subpulse
            return test0 and test1
        except AttributeError:
            return False

    @property
    def primes(self):
        return tuple(set(prime_factors.factorise(self)))

    def count_subpulse(self):
        try:
            return self / self.subpulse
        except ZeroDivisionError:
            return 0
