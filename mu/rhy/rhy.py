from mu.time import time
from mu.abstract import muobjects
from mu.utils import music21
from mu.utils import prime_factors
import abc
import functools
import operator
import itertools
from typing import Union


class AbstractRhythm(abc.ABC):
    """
    A Rhythm may be a structure to organise time.
    """
    @abc.abstractmethod
    def flat(self):
        raise NotImplementedError

    @abc.abstractproperty
    def delay(self):
        raise NotImplementedError

    @abc.abstractproperty
    def stretch(self, arg):
        raise NotImplementedError


class RhyUnit(AbstractRhythm, time.Time):
    def __repr__(self):
        return time.Time.__repr__(self)

    def flat(self):
        return RhyCompound((self,))

    @property
    def delay(self):
        return time.Time(self)

    def stretch(self, arg):
        return RhyUnit(self * arg)

    @music21.decorator
    def convert2music21(self):
        return music21.m21.duration.Duration(self)


class RhyCompound(AbstractRhythm, muobjects.MUList):
    def flat(self):
        if self:
            return functools.reduce(
                lambda x, y: x + y, tuple(u.flat() for u in self))
        else:
            return RhyCompound([])

    @property
    def delay(self):
        return time.Time(sum(u.delay for u in self))

    def stretch(self, arg):
        return RhyCompound(tuple(u.stretch(arg) for u in self))

    def append(self, arg: Union[int, RhyUnit]) -> None:
        if id(arg) == id(self):
            arg = arg.copy()
        list.append(self, arg)


class PulseChroma(muobjects.MUInt):
    def __init__(self, length):
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
        """
        """
        try:
            subpath = tuple(sub.specify() for sub in self.subpulse)
            subpath = tuple(functools.reduce(operator.add, subpath))
        except AttributeError:
            return SpecifiedPulseChroma(self, 0),
        return tuple(SpecifiedPulseChroma(self, sub) for sub in subpath)


class SpecifiedPulseChroma:
    def __init__(self, length, subpulse):
        self._length = int(length)
        self.subpulse = subpulse

    def __repr__(self):
        return repr(self._length)

    def __eq__(self, other):
        try:
            test0 = self._length == other._length
            test1 = self.subpulse == other.subpulse
            return test0 and test1
        except AttributeError:
            return False

    def count_subpulse(self):
        try:
            return self._length / self.subpulse
        except ZeroDivisionError:
            return 0
