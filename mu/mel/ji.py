from mu.mel import abstract
from fractions import Fraction
import pyprimes
from pyprimes import factors
import functools
import itertools


def comparable_bool_decorator(func):
    def wrap(*args, **kwargs):
        if Monzo.is_comparable(args[0], args[1]):
            return func(*args, **kwargs)
        else:
            return False
    return wrap


def comparable_monzo_decorator(func):
    def wrap(*args, **kwargs):
        if Monzo.is_comparable(args[0], args[1]):
            return func(*args, **kwargs)
        else:
            return Monzo([])
    return wrap


class Monzo:
    _val_shift = 0

    def __init__(self, iterable, val_border=1):
        self._vector = Monzo._shift_vector(
            tuple(iterable), pyprimes.prime_count(val_border))
        self.val_border = val_border

    def __getitem__(self, idx):
        res = self._vec[idx]
        if type(idx) == slice:
            return type(self)(res, self.val_border)
        else:
            return res

    def __iter__(self):
        return iter(self._vec)

    @property
    def _vec(self):
        return self._vector[self._val_shift:]

    def __repr__(self):
        return repr(self._vec)

    def __len__(self):
        return len(self._vec)

    def index(self, arg):
        return self._vec.index(arg)

    @staticmethod
    def adjusted_monzos(m0, m1) -> tuple:
        m0, m1 = list(m0), list(m1)
        while len(m1) < len(m0):
            m1.append(0)
        while len(m0) < len(m1):
            m0.append(0)
        return tuple(m0), tuple(m1)

    @staticmethod
    def is_comparable(m0: "Monzo", m1: "Monzo") -> bool:
        try:
            return m0._val_shift == m1._val_shift
        except AttributeError:
            return False

    @staticmethod
    def calc_iterables(iterable0, iterable1, operation) -> iter:
        return (operation(x, y) for x, y in zip(iterable0, iterable1))

    @staticmethod
    def adjust_ratio(r: Fraction, val_border=int) -> Fraction:
        if val_border > 1:
            while r > val_border:
                r /= val_border
            while r < 1:
                r *= val_border
            return r
        else:
            return r

    @staticmethod
    def monzo2ratio(monzo: tuple, val: tuple, val_border: int) -> Fraction:
        numerator = 1
        denominator = 1
        for number, exponent in zip(val, monzo):
            if exponent > 0:
                numerator *= pow(number, exponent)
            elif exponent < 0:
                exponent *= -1
                denominator *= pow(number, exponent)
        return Monzo.adjust_ratio(Fraction(numerator, denominator), val_border)

    @staticmethod
    def ratio2monzo(ratio: Fraction, val_shift=0) -> "Monzo":
        gen_pos = factors.factors(ratio.numerator)
        gen_neg = factors.factors(ratio.denominator)

        biggest_prime = max(factors.factorise(
            ratio.numerator) + factors.factorise(ratio.denominator))
        monzo = [0] * pyprimes.prime_count(biggest_prime)

        for num, fac in gen_pos:
            if num > 1:
                monzo[pyprimes.prime_count(num) - 1] += fac

        for num, fac in gen_neg:
            if num > 1:
                monzo[pyprimes.prime_count(num) - 1] -= fac

        return Monzo(monzo[val_shift:])

    @staticmethod
    def _shift_vector(vec, shiftval) -> tuple:
        if shiftval > 0:
            m = [0] * shiftval + list(vec)
        else:
            m = vec[abs(shiftval):]
        return tuple(m)

    @property
    def val(self) -> tuple:
        return tuple(pyprimes.nprimes(
            len(self) + self._val_shift))[self._val_shift:]

    @property
    def val_border(self) -> int:
        if self._val_shift == 0:
            return 1
        else:
            return tuple(pyprimes.nprimes(
                len(self) + self._val_shift))[self._val_shift - 1]

    @val_border.setter
    def val_border(self, v):
        difference = pyprimes.prime_count(
            v) - pyprimes.prime_count(self.val_border)
        self._val_shift += difference

    @property
    def ratio(self) -> Fraction:
        return Monzo.monzo2ratio(self, self.val, self.val_border)

    @property
    def float(self) -> float:
        return float(self.ratio)

    @property
    def gender(self) -> int:
        maxima = max(self)
        minima = min(self)
        if (maxima > 0 and minima >= 0) or (
                maxima > 0 and self.index(maxima) > self.index(minima)):
            return 1
        elif maxima <= 0 and minima < 0 or (
                minima < 0 and self.index(minima) > self.index(maxima)):
            return -1
        else:
            return 0

    @property
    def harmonic(self) -> int:
        if self.ratio.denominator % 2 == 0:
            return self.ratio.numerator
        elif self.ratio.numerator % 2 == 0:
            return - self.ratio.denominator
        elif self.ratio == Fraction(1, 1):
            return 1
        else:
            return 0

    @property
    def primes(self) -> tuple:
        p = factors.factorise(self.ratio.numerator * self.ratio.denominator)
        return tuple(sorted(tuple(set(p))))[self._val_shift:]

    @property
    def components(self) -> tuple:
        vectors = [[0] * c + [x] for c, x in enumerate(self) if x != 0]
        return tuple(type(self)(
                vec, val_border=self.val_border) for vec in vectors)

    @comparable_bool_decorator
    def is_related(self: "Monzo", other: "Monzo") -> bool:
        for p in self.primes:
            if p in other.primes:
                return True
        return False

    @comparable_bool_decorator
    def is_congeneric(self: "Monzo", other: "Monzo") -> bool:
        if self.primes == other.primes:
            return True
        else:
            return False

    @comparable_bool_decorator
    def __eq__(self: "Monzo", other: "Monzo") -> bool:
        return tuple.__eq__(self._vector, other._vector)

    def summed(self) -> int:
        return sum(map(lambda x: abs(x), self))

    def subvert(self) -> list:
        def ispos(num):
            if num > 0:
                return 1
            else:
                return -1
        sep = [tuple(type(self)([0] * counter + [ispos(vec)], self.val_border)
                     for i in range(abs(vec)))
               for counter, vec in enumerate(self) if vec != 0]
        res = [a for sub in sep for a in sub]
        if len(res) == 0:
            res.append(type(self)([0]))
        return res

    def copy(self) -> "Monzo":
        return Monzo(self, self.val_border)

    @comparable_monzo_decorator
    def __math(self, other, operation) -> "Monzo":
        m0, m1 = Monzo.adjusted_monzos(self, other)
        return Monzo(Monzo.calc_iterables(m0, m1, operation), self.val_border)

    def __add__(self, other: "Monzo") -> "Monzo":
        return self.__math(other, lambda x, y: x + y)

    def __sub__(self, other: "Monzo") -> "Monzo":
        return self.__math(other, lambda x, y: x - y)

    def __mul__(self, other) -> "Monzo":
        return self.__math(other, lambda x, y: x * y)

    def __div__(self, other) -> "Monzo":
        return self.__math(other, lambda x, y: x / y)

    def scalar(self, factor):
        """Return the scalar-product of a Monzo and its factor."""
        return self * type(self)((factor,) * len(self), self.val_border)

    def dot(self, other) -> float:
        """Return the dot-product of two Monzos."""
        return sum(a * b for a, b in zip(self, other))

    def matrix(self, other):
        """Return the matrix-product of two Monzos."""
        m0 = tuple(type(self)(tuple(arg * arg2 for arg2 in other),
                              self.val_border) for arg in self)
        m1 = tuple(type(self)(tuple(arg * arg2 for arg2 in self),
                              self.val_border) for arg in other)
        return m0 + m1

    def inverse(self) -> "Monzo":
        return type(self)(list(map(lambda x: -x, self)), self.val_border)

    def shift(self, shiftval: int) -> "Monzo":
        return type(self)(Monzo._shift_vector(
            self, shiftval), self.val_border)


class JIPitch(Monzo, abstract.AbstractPitch):
    multiply = 1

    def __init__(self, iterable, val_border=1, multiply=1):
        self._vector = tuple(Monzo._shift_vector(
            iterable, pyprimes.prime_count(val_border)))
        self.val_border = val_border
        self.multiply = multiply

    def __eq__(self, other) -> bool:
        return abstract.AbstractPitch.__eq__(self, other)

    def __repr__(self) -> str:
        return str(self.ratio)

    @property
    def monzo(self) -> tuple:
        return tuple(self)

    def __add__(self, other) -> "JIPitch":
        return JIPitch(Monzo.__add__(self, other), self.val_border)

    def __sub__(self, other) -> "JIPitch":
        return JIPitch(Monzo.__sub__(self, other), self.val_border)

    def __mul__(self, other) -> "JIPitch":
        return JIPitch(Monzo.__mul__(self, other), self.val_border)

    def __hash__(self):
        return abstract.AbstractPitch.__hash__(self)

    def calc(self, factor=1) -> float:
        return float(self.ratio * self.multiply * factor)

    def copy(self) -> "JIPitch":
        return JIPitch(self, self.val_border, self.multiply)

    @classmethod
    def from_ratio(cls, num: int, den: int, val_border=1, multiply=1
                   ) -> "JIPitch":
        obj = cls(JIPitch.ratio2monzo(Fraction(num, den), cls._val_shift))
        obj.val_border = val_border
        obj.multiply = multiply
        return obj

    @classmethod
    def from_monzo(cls, *arg, val_border=1, multiply=1) -> "JIPitch":
        obj = cls(arg, val_border)
        obj.multiply = multiply
        return obj


class JIContainer:
    def __init__(self, iterable, multiply=260):
        super(type(self), self).__init__(iterable)
        self.multiply = multiply
        self._val_border = 1

    @classmethod
    def mk_line(cls, reference, count):
        return cls([reference.scalar(i + 1) for i in range(count)])

    @classmethod
    def mk_line_and_inverse(cls, reference, count):
        m0 = cls.mk_line(reference, count)
        return m0 & m0.inverse()

    def set_multiply(self, arg):
        for t in self:
            t.multiply = arg

    def show(self) -> tuple:
        r = tuple((r, p, round(f, 2))
                  for r, p, f in zip(self, self.primes, self.freq))
        return tuple(sorted(r, key=lambda t: t[2]))

    def dot_sum(self):
        """Return the sum of the dot-product of each Monzo
        with each other Monzo in the Container"""
        d = 0
        for m_out in self:
            for m_in in self:
                if m_in != m_out:
                    d += m_out.dot(m_in)
        return d


class JIMel(JIPitch.mk_iterable(abstract.AbstractMelody), JIContainer):
    def __init__(self, iterable, multiply=260):
        return JIContainer.__init__(self, iterable, multiply)

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def intervals(self):
        """return intervals between single notes"""
        return self[1:] - self[:-1]

    def __getitem__(self, idx):
        res = abstract.AbstractMelody.__getitem__(self, idx)
        if type(res) == type(self):
            res.multiply = self.multiply
            res.val_border = self.val_border
        return res

    def __add__(self, other: "JIMel"):
        return JIMel((m0 + m1 for m0, m1 in zip(self, other)))

    def __sub__(self, other: "JIMel"):
        return JIMel((m0 - m1 for m0, m1 in zip(self, other)))

    def __mul__(self, other: "JIMel"):
        return JIMel((m0 * m1 for m0, m1 in zip(self, other)))

    def __div__(self, other: "JIMel"):
        return JIMel((m0 / m1 for m0, m1 in zip(self, other)))

    @property
    def val_border(self) -> int:
        return self[0].val_border

    @val_border.setter
    def val_border(self, arg) -> None:
        for f in self:
            f.val_border = arg
        self._val_border = arg

    def subvert(self):
        return type(self)(functools.reduce(
            lambda x, y: x + y, tuple(t.subvert() for t in self)),
            self.multiply)

    def accumulate(self):
        return type(self)(tuple(itertools.accumulate(self)),
                          self.multiply)

    def separate(self):
        subverted = JIMel((self[0],)) & self.intervals.subvert()
        return type(self)(subverted, self.multiply).accumulate()


class JIHarmony(JIPitch.mk_iterable(abstract.AbstractHarmony), JIContainer):
    def __init__(self, iterable, multiply=260):
        return JIContainer.__init__(self, iterable, multiply)

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def val_border(self) -> int:
        return self._val_border

    @val_border.setter
    def val_border(self, arg) -> None:
        for f in self:
            f.val_border = arg
        self._val_border = arg


"""
    syntactic sugar for creating JIPitch - Objects:
"""


def r(num, den, val_border=1, multiply=1):
    return JIPitch.from_ratio(num, den, val_border, multiply)


def m(*num, val_border=1, multiply=1):
    return JIPitch.from_monzo(*num, val_border=val_border, multiply=multiply)
