import abc
import bisect
import functools
import itertools
import numpy as np
import operator

from mu.abstract import muobjects
from mu.rhy import rhy


class Interpolation(abc.ABC):
    """Abstract superclass for all interpolation subclasses."""

    @abc.abstractmethod
    def __call__(self, x0, x1, n, dtype=float) -> tuple:
        raise NotImplementedError

    def copy(self) -> "Interpolation":
        return type(self)()


class Linear(Interpolation):
    def __call__(self, x0, x1, n, dtype=float) -> tuple:
        return tuple(np.linspace(x0, x1, n, dtype=dtype))

    def __hash__(self) -> int:
        return hash("LinearInterpolation")


class Logarithmic(Interpolation):
    def __call__(self, x0, x1, n, dtype=float) -> tuple:
        if x0 == 0:
            x0_pre = 0.00001
            return (0,) + tuple(np.geomspace(x0_pre, x1, n, dtype=dtype))[1:]
        else:
            return tuple(np.geomspace(x0, x1, n, dtype=dtype))

    def __hash__(self) -> int:
        return hash("LogarithmicInterpolation")


class Proportional(Interpolation):
    """The slope of this interpolation changes depending on the proportion.

    For a proportion > 1 the slope becomes sharper at the end of the interpolation,
    while a proportion < 1 means that the slope is bigger at the beginning.
    Proportion = 1 leads to linear interpolation.
    """

    def __init__(self, proportion: float):
        self.__interpolate = self.mk_interpolation_function(proportion)
        self.__proportion = proportion

    @staticmethod
    def mk_interpolation_function(proportion):
        def interpolate(start, end, length) -> tuple:
            length -= 1
            diff = end - start
            avg_change = diff / (length)
            halved = length // 2
            a, b = [], []
            for prop in tuple(np.linspace(proportion, 1, halved + 1))[:-1]:
                n, m = solve(prop, avg_change)
                a.append(n)
                b.append(m)
            b = reversed(b)
            if length % 2 != 0:
                a.append(avg_change)
            changes = [start] + a + list(b)
            return tuple(itertools.accumulate(changes))

        def solve(proportion, change) -> tuple:
            complete_change = change * 2
            n = complete_change / (proportion + 1)
            m = complete_change - n
            return n, m

        return interpolate

    def __call__(self, x0, x1, n, dtype=float) -> tuple:
        res = self.__interpolate(x0, x1, n)
        if dtype != float:
            res = tuple(dtype(n) for n in res)
        return res

    def __hash__(self) -> int:
        return hash((hash("ProportionalInterpolation"), hash(self.__proportion)))


class InterpolationEvent(object):
    def __init__(self, delay: rhy.Unit, interpolation_type=Linear()):
        if isinstance(delay, rhy.Unit) is False:
            delay = rhy.Unit(delay)
        self.__delay = delay
        self.__interpolation_type = interpolation_type

    @property
    def delay(self):
        return self.__delay

    @property
    def interpolation_type(self):
        return self.__interpolation_type

    @abc.abstractproperty
    def interpolate(self, other, steps):
        raise NotImplementedError


class InterpolationLine(muobjects.MUList):
    """Container class to describe interpolations between states.

    They are expected to contain InterpolationEvent - objects.
    InterpolationLine - objects can be called to generate the interpolation.
    Input arguments are only Gridsize that describe how small
    one interpolation step is. InterpolationLine - objects are
    unchangeable objects.

    The last event is expected to have delay == 0.
    """

    def __init__(self, iterable):
        iterable = tuple(iterable)
        try:
            assert iterable[-1].delay == 0
        except AssertionError:
            raise ValueError("The last element has to have delay = 0")
        muobjects.MUList.__init__(self, iterable)

    def __call__(self, gridsize: float):
        def find_closest_point(points, time):
            pos = bisect.bisect_right(points, time)
            try:
                return min(
                    (
                        (abs(time - points[pos]), pos),
                        (abs(time - points[pos - 1]), pos - 1),
                    ),
                    key=operator.itemgetter(0),
                )[1]
            except IndexError:
                # if pos is len(points) + 1
                return pos

        points = Linear()(0, float(self.duration), int(self.duration / gridsize))
        absolute_delays = self.delay.convert2absolute()
        positions = tuple(
            find_closest_point(points, float(delay)) for delay in absolute_delays
        )
        interpolation_size = tuple(b - a for a, b in zip(positions, positions[1:]))
        interpolations = (
            item0.interpolate(item1, steps + 1)[:-1]
            for item0, item1, steps in zip(self, self[1:], interpolation_size)
        )
        return tuple(functools.reduce(operator.add, interpolations))

    def copy(self):
        return type(self)(item.copy() for item in self)

    @property
    def delay(self) -> rhy.Compound:
        return rhy.Compound(obj.delay.copy() for obj in self)

    @property
    def duration(self) -> float:
        return sum(self.delay)
