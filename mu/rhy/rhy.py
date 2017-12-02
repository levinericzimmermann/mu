from mu.time import time
from mu.abstract import muobjects
import abc
import functools


class Rhythm(abc.ABC):
    @abc.abstractmethod
    def flat(self):
        raise NotImplementedError

    @abc.abstractproperty
    def delay(self):
        raise NotImplementedError

    @abc.abstractproperty
    def stretch(self, arg):
        raise NotImplementedError


class RhyUnit(Rhythm, time.Time):
    def __repr__(self):
        return time.Time.__repr__(self)

    def flat(self):
        return RhyCompound((self,))

    @property
    def delay(self):
        return float(self)

    def stretch(self, arg):
        return RhyUnit(self * arg)


class RhyCompound(Rhythm, muobjects.MUList):
    def flat(self):
        if self:
            return functools.reduce(
                    lambda x, y: x + y, tuple(u.flat() for u in self))
        else:
            return RhyCompound([])

    @property
    def delay(self):
        return sum(u.delay for u in self)

    def stretch(self, arg):
        return RhyCompound(tuple(u.stretch(arg) for u in self))

    def append(self, arg):
        if id(arg) == id(self):
            arg = arg.copy()
        list.append(self, arg)
