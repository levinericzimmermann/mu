import abc
import itertools
import operator


"""This module implements objects that infinitely support a next - call."""


class InfIt(abc.ABC):
    @abc.abstractmethod
    def __next__(self) -> object:
        raise NotImplementedError


class Cycle(InfIt):
    def __init__(self, iterable: tuple):
        self.__cycle = itertools.cycle(iterable)

    def __next__(self) -> object:
        return next(self.__cycle)


class MathOperation(InfIt):
    def __init__(self, operator, start: float, n: float):
        self.__operator = operator
        self.__value = start
        self.__n = n

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
    def __init__(self, start: float = 1, n: float = 2):
        super().__init__(lambda n, value: value ** n, start, n)


class Random(InfIt):
    def __init__(self, seed: int):
        import random as random_module

        random_module.seed(seed)
        self.__random_module = random_module

    @property
    def random_module(self):
        return self.__random_module
