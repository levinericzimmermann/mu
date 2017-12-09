from mu.mel import abstract
from mu.mel import mel
from mu.abstract import muobjects
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
        # TODO: replace ugly implementation
        vec = self._vector[self._val_shift:]
        c = 0
        for i in reversed(vec):
            if i == 0:
                c += 1
            else:
                break
        if c:
            vec = vec[:-c]
        return vec

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

    @staticmethod
    def gcd(*args):
        def _gcd(a, b):
            """Return greatest common divisor using Euclid's Algorithm
            https://stackoverflow.com/questions/147515/least-common-multiple-for-3-or-more-numbers"""
            while b:
                a, b = b, a % b
            return a
        return functools.reduce(_gcd, args)

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
    def gender(self) -> bool:
        if self:
            maxima = max(self)
            minima = min(self)
            if (maxima > 0 and minima >= 0) or (
                    maxima > 0 and self.index(maxima) > self.index(minima)):
                return True
            elif maxima <= 0 and minima < 0 or (
                    minima < 0 and self.index(minima) > self.index(maxima)):
                return False
        else:
            return True

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
    def quantity(self) -> int:
        return len(self.primes)

    @property
    def components(self) -> tuple:
        vectors = [[0] * c + [x] for c, x in enumerate(self) if x != 0]
        return tuple(type(self)(
            vec, val_border=self.val_border) for vec in vectors)

    @property
    def lv(self):
        if self:
            return abs(Monzo.gcd(*tuple(filter(lambda x: x != 0, self))))
        else:
            return 1

    @property
    def identity(self):
        if self:
            filtered = type(self)([1 / self.lv] * len(self), self.val_border)
            monzo = tuple(int(x) for x in self * filtered)
            return type(self)(monzo, self.val_border)
        else:
            return type(self)([], self.val_border)

    @property
    def past(self) -> tuple:
        return tuple(type(self)(
            self.identity.scalar(i), self.val_border) for i in range(self.lv))

    @property
    def is_root(self) -> bool:
        test = Monzo(self._vector, 1)
        test.val_border = 2
        if test:
            return False
        else:
            return True

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
        return tuple.__eq__(self._vec, other._vec)

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

    def __pow__(self, other) -> "Monzo":
        return self.__math(other, lambda x, y: x ** y)

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

    def filter(self, *prime):
        return type(self)(MonzoFilter(
                *prime, val_border=self.val_border) * self, self.val_border)


class MonzoFilter(Monzo):
    @staticmethod
    def mk_filter_vec(*prime):
        numbers = tuple(pyprimes.prime_count(p) for p in prime)
        iterable = [1] * max(numbers)
        for n in numbers:
            iterable[n-1] = 0
        return iterable

    def __init__(self, *prime, val_border=1):
        Monzo.__init__(self, MonzoFilter.mk_filter_vec(*prime), 1)
        self.val_border = val_border

    def __mul__(self, other):
        iterable = list(self._vector)[self._val_shift:]
        while len(other._vec) > len(iterable):
            iterable.append(1)
        return Monzo.__mul__(Monzo(iterable, self.val_border), other)


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

    def __div__(self, other) -> "JIPitch":
        return JIPitch(Monzo.__div__(self, other), self.val_border)

    def __pow__(self, val) -> "JIPitch":
        return JIPitch(Monzo.__pow__(self, val), self.val_border)

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

    @property
    def identity_adjusted(self):
        """unstable, experimental method.
        works only for 2/1 as a frame and for val_border=2"""
        id = self.identity
        if id:
            id.val_border = 1
            id -= Monzo([id[0]])
            if id.gender:
                while id.float < 1:
                    id += Monzo([1])
                while id.float > 2:
                    id -= Monzo([1])
                if id.float > 1.7:
                    id.multiply *= 0.25
                elif id.float > 1.5:
                    id.multiply *= 0.5
            else:
                while id.float > 1:
                    id -= Monzo([1])
                while id.float < 0.5:
                    id += Monzo([1])
                if id.float * 2 < 1.2:
                    id.multiply *= 4
                elif id.float * 2 < 1.5:
                    id.multiply *= 2
            return id
        else:
            return id

    @property
    def adjusted_register(self):
        return type(self)(
                self.identity_adjusted.scalar(self.lv),
                1, self.identity_adjusted.multiply)


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
        return m0 + m0.inverse()

    @property
    def average_gender(self):
        return sum(map(lambda b: 1 if b is True else -1,
                       self.gender)) / len(self)

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

    @property
    def summed_summed(self):
        return sum(self.summed())

    def count_root(self):
        return sum(map(lambda p: 1 if p.is_root else 0, self))


class JIMel(JIPitch.mk_iterable(mel.Mel), JIContainer):
    def __init__(self, iterable, multiply=260):
        JIContainer.__init__(self, iterable, multiply)

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def intervals(self):
        """return intervals between single notes"""
        return self[1:].sub(self[:-1])

    def __getitem__(self, idx):
        res = muobjects.MUList.__getitem__(self, idx)
        if type(res) == type(self):
            res.multiply = self.multiply
            res.val_border = self.val_border
        return res

    def add(self, other: "JIMel"):
        return JIMel((m0 + m1 for m0, m1 in zip(self, other)))

    def sub(self, other: "JIMel"):
        return JIMel((m0 - m1 for m0, m1 in zip(self, other)))

    def mul(self, other: "JIMel"):
        return JIMel((m0 * m1 for m0, m1 in zip(self, other)))

    def div(self, other: "JIMel"):
        return JIMel((m0 / m1 for m0, m1 in zip(self, other)))

    @property
    def val_border(self) -> int:
        return self[0].val_border

    @val_border.setter
    def val_border(self, arg) -> None:
        for f in self:
            f.val_border = arg
        self._val_border = arg

    @property
    def different_pitches(self):
        container = []
        for t in self:
            if t not in container:
                container.append(t)
        return tuple(container)

    @property
    def is_ordered(self):
        if self.order() == 1:
            return True
        else:
            return False

    def order(self, val_border=2):
        intervals = self.intervals
        intervals.val_border = val_border
        return sum(intervals.lv) / len(intervals)

    def count_repeats(self):
        repeats = 0
        for p0, p1 in zip(self, self[1:]):
            if p0 == p1:
                repeats += 1
        return repeats

    def count_different_pitches(self):
        return len(self.different_pitches)

    def subvert(self):
        return type(self)(functools.reduce(
            lambda x, y: x + y, tuple(t.subvert() for t in self)),
            self.multiply)

    def accumulate(self):
        return type(self)(tuple(itertools.accumulate(self)),
                          self.multiply)

    def separate(self):
        subverted = JIMel((self[0],)) + self.intervals.subvert()
        return type(self)(subverted, self.multiply).accumulate()


class JIHarmony(JIPitch.mk_iterable(mel.Harmony), JIContainer):
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

    def operator_harmony(self, func):
        # TODO: replace ugly implementation by a better one
        new_har = type(self)([], self.multiply)
        for c, p in enumerate(self):
            for c2, p2 in enumerate(self):
                if c2 > c:
                    new_har.add(func(p, p2))
        new_har.val_border = self.val_border
        return new_har

    def add_harmony(self):
        return self.operator_harmony(lambda x, y: x + y)

    def sub_harmony(self):
        return self.operator_harmony(lambda x, y: x - y)

    def mul_harmony(self):
        return self.operator_harmony(lambda x, y: x * y)


class JICadence(JIPitch.mk_iterable(mel.Cadence), JIContainer):
    def __init__(self, iterable, multiply=1):
        super(type(self), self).__init__(iterable)
        self.multiply = multiply
        self._val_border = 1

    def calc(self, factor=1) -> tuple:
        return tuple(h.calc(self.multiply * factor) for h in self)

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

    def dot_sum(self):
        return tuple(h.dot_sum() for h in self)


"""
    syntactic sugar for the creation of JIPitch - Objects:
"""


def r(num, den, val_border=1, multiply=1):
    return JIPitch.from_ratio(num, den, val_border, multiply)


def m(*num, val_border=1, multiply=1):
    return JIPitch.from_monzo(*num, val_border=val_border, multiply=multiply)
