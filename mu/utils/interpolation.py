import abc
import itertools
import numpy as np


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


class Logarithmic(Interpolation):
    def __call__(self, x0, x1, n, dtype=float) -> tuple:
        return tuple(np.geomspace(x0, x1, n, dtype=dtype))


class Proportional(Interpolation):
    """The slope of this interpolation changes depending on the proportion.

    For a proportion > 1 the slope becomes sharper at the end of the interpolation,
    while a proportion < 1 means that the slope is bigger at the beginning.
    Proportion = 1 leads to linear interpolation.
    """

    def __init__(self, proportion: float):
        self.__interpolate = self.mk_interpolation_function(proportion)

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
