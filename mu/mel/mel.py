from mu.abstract import muobjects
from mu.mel import abstract
from typing import Any
import collections


class SimplePitch(abstract.AbstractPitch):
    """A very simple pitch / interval implementation.

    SimplePitch - objects are specified via a concert_pitch frequency
    and a cent value, that describe the distance to the concert_pitch.
    """

    def __init__(self, concert_pitch_freq: float, cents: float = 0):
        self.concert_pitch_freq = concert_pitch_freq
        self.__cents = cents

    def calc(self) -> float:
        return self.concert_pitch_freq * (2 ** (self.cents / 1200))

    @property
    def cents(self) -> float:
        return self.__cents

    def copy(self) -> "SimplePitch":
        return type(self)(self.concert_pitch_freq, self.cents)


class EmptyPitch(abstract.AbstractPitch):
    def calc(self, factor=0):
        return None

    def __repr__(self):
        return "NoPitch"

    def copy(self):
        return type(self)()

    @property
    def cents(self) -> None:
        return None


TheEmptyPitch = EmptyPitch()


class Mel(muobjects.MUList):
    def __init__(self, iterable: Any, multiply: int = 260) -> None:
        muobjects.MUList.__init__(self, iterable)
        self.multiply = multiply

    def copy(self):
        iterable = tuple(item.copy() for item in self)
        return type(self)(iterable, multiply=self.multiply)

    def __hash__(self) -> int:
        return hash(tuple(hash(t) for t in self))

    def calc(self, factor: int = 1) -> tuple:
        return tuple(p.calc(self.multiply * factor) for p in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def cents(self) -> tuple:
        return tuple(item.cents for item in self)

    def uniqify(self):
        unique = collections.OrderedDict((x, True) for x in self).keys()
        return type(self)(unique)


class Harmony(muobjects.MUSet):
    def __hash__(self) -> int:
        return hash(tuple(hash(t) for t in self))

    def sorted(self):
        return sorted(self.calc())

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def cents(self) -> tuple:
        return tuple(item.cents for item in self)


class Cadence(muobjects.MUList):
    def __hash__(self):
        return hash(tuple(hash(h) for h in self))

    def calc(self, factor=1) -> tuple:
        return tuple(h.calc(self.multiply * factor) for h in self)

    @property
    def freq(self) -> tuple:
        return self.calc()


class Scale(muobjects.MUOrderedSet):
    _period_cls = Mel

    def __init__(self, period, periodsize):
        if not type(period) == self._period_cls:
            period = self._period_cls(period)
        period = period.sort().uniqify()
        muobjects.MUOrderedSet.__init__(self, period + self._period_cls((periodsize,)))
        self.period = period
        self.periodsize = periodsize

    def __add__(self, other):
        return type(self)(
            tuple(self.period) + tuple(other.period),
            max((self.periodsize, other.periodsize)),
        )
