# @Author: Levin Eric Zimmermann
# @Date:   2018-04-07T15:12:18+02:00
# @Email:  levin-eric.zimmermann@folkwang-uni.de
# @Project: mu
# @Last modified by:   uummoo
# @Last modified time: 2018-04-07T15:39:01+02:00


from mu.abstract import muobjects
from mu.mel import abstract
from typing import Any
import collections


class EmptyPitch(abstract.AbstractPitch):
    def calc(self, factor=0):
        return None

    def __repr__(self):
        return "NoPitch"

    def copy(self):
        return type(self)()


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

    def uniqify(self):
        return type(self)(
                collections.OrderedDict((x, True) for x in self).keys())


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
        muobjects.MUOrderedSet.__init__(
                self, period + self._period_cls((periodsize, )))
        self.period = period
        self.periodsize = periodsize

    def __add__(self, other):
        return type(self)(
                tuple(self.period) + tuple(other.period),
                max((self.periodsize, other.periodsize)))
