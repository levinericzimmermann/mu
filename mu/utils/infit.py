"""This module implements infinite generators.

With 'infinite generators' the author is refering to objects that infinitely
support a next - call.
"""

import abc
import itertools
import operator

from mu.utils import activity_levels


class InfIt(abc.ABC):
    """Abstract class for infinite generators."""

    @abc.abstractmethod
    def __next__(self) -> object:
        raise NotImplementedError


class Cycle(InfIt):
    def __init__(self, iterable: tuple):
        self.__cycle = itertools.cycle(iterable)
        self.__repr = "Cycle({})".format(iterable)

    def __repr__(self) -> str:
        return self.__repr

    def __next__(self) -> object:
        return next(self.__cycle)


class Value(Cycle):
    def __init__(self, item):
        super().__init__((item,))
        self.__repr = "Value({})".format(item)


class NestedCycle(Cycle):
    """Infinite cycle that contains other InfIt objects that can be called."""

    def __init__(self, *infinite_iterable: InfIt):
        super().__init__(infinite_iterable)

    def __repr__(self) -> str:
        return "Nested{}".format(super().__repr__())

    def __next__(self) -> object:
        return next(super().__next__())


class MetaCycle(Cycle):
    """Infinite cycle that dynamically builds new InfIt objects when it get called."""

    def __init__(self, *object_argument_pair: tuple):
        super().__init__(object_argument_pair)

    def __repr__(self) -> str:
        return "Meta{}".format(super().__repr__())

    def __next__(self) -> InfIt:
        obj, arguments = next(super().__next__())
        return obj(*arguments)


class MathOperation(InfIt):
    def __init__(self, operator, start: float, n: float):
        self.__operator = operator
        self.__value = start
        self.__n = n

    def __repr__(self) -> str:
        return "InfitMath({}, {}, {})".format(self.__value, self.__n, self.__operator)

    def __next__(self) -> object:
        new_value = self.__operator(self.__n, self.__value)
        self.__value = new_value
        return new_value


class Addition(MathOperation):
    def __init__(self, start: float = 0, n: float = 1):
        super().__init__(operator.add, start, n)


class Multiplication(MathOperation):
    def __init__(self, start: float = 1, n: float = 2):
        super().__init__(operator.mul, start, n)


class Power(MathOperation):
    def __init__(self, start: float = 2, n: float = 2):
        super().__init__(lambda n, value: value ** n, start, n)


class Random(InfIt):
    def __init__(self, seed: int):
        import random as random_module

        random_module.seed(seed)
        self.__random_module = random_module

    @property
    def random_module(self):
        return self.__random_module


class Uniform(Random):
    def __init__(self, border0: float = 0, border1: float = 1, seed: int = 1):
        super().__init__(seed)
        self.__lower_border = border0
        self.__uppper_border = border1

    def __repr__(self) -> str:
        return "Uniform({}, {})".format(self.__lower_border, self.__uppper_border)

    def __next__(self) -> float:
        return self.random_module.uniform(self.__lower_border, self.__uppper_border)


class Gaussian(Random):
    def __init__(self, center: float = 0, deviation: float = 1, seed: int = 1):
        super().__init__(seed)
        self.__center = center
        self.__deviation = deviation
        self.__border0 = center - deviation
        self.__border1 = center + deviation
        self.__standard_deviation = deviation / 3

    def __repr__(self) -> str:
        return "Gaussian({} +/- {})".format(self.__center, self.__deviation)

    def __next__(self) -> float:
        def next_gaussian() -> float:
            return self.random_module.gauss(self.__center, self.__standard_deviation)

        result = next_gaussian()

        # making sure the result isn't going too high or too low
        while result < self.__border0 or result > self.__border1:
            result = next_gaussian()

        return result


class ActivityLevel(InfIt):
    """infit module implementation of Activity Levels.

    See mu.utils.activity_levels.ActivityLevel for a more detailed explanation.
    """

    def __init__(
        self, activity_level: int, start_at: int = 0, is_inverse: bool = False
    ):
        self.__al = activity_levels.ActivityLevel(
            start_at=start_at, is_inverse=is_inverse
        )
        self.__activity_level = activity_level

    def __repr__(self) -> str:
        return "InfIt_ActivityLevel({})".format(self.__activity_level)

    def __next__(self) -> int:
        return self.__al(self.__activity_level)
