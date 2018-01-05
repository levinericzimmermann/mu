from mu.mel import abstract
from mu.mel import mel
from mu.abstract import muobjects
from fractions import Fraction
import pyprimes
from pyprimes import factors
import functools
import itertools
import math
from typing import (Callable, List, Type)


def comparable_bool_decorator(func: Callable) -> Callable:
    def wrap(*args, **kwargs):
        if Monzo.is_comparable(args[0], args[1]):
            return func(*args, **kwargs)
        else:
            return False
    return wrap


def comparable_monzo_decorator(func: Callable) -> Callable:
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

    @classmethod
    def from_ratio(cls, num: int, den: int, val_border=1, multiply=1
                   ) -> Type["Monzo"]:
        obj = cls(cls.ratio2monzo(Fraction(num, den), cls._val_shift))
        obj.val_border = val_border
        obj.multiply = multiply
        return obj

    @classmethod
    def from_monzo(cls, *arg, val_border=1, multiply=1) -> Type["Monzo"]:
        obj = cls(arg, val_border)
        obj.multiply = multiply
        return obj

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
    def adjust_ratio(r: Fraction, val_border: int) -> Fraction:
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
    def ratio2monzo(ratio: Fraction, val_shift=0) -> Type["Monzo"]:
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
    def gcd(*args) -> int:
        return functools.reduce(math.gcd, args)

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
    def val_border(self, v: int):
        difference = pyprimes.prime_count(
            v) - pyprimes.prime_count(self.val_border)
        self._val_shift += difference

    def set_val_border(self, val_border: int) -> Type["Monzo"]:
        """Return a new Monzo-Object
        with a new val_border"""
        copied = self.copy()
        copied.val_border = val_border
        return copied

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
    def lv(self) -> int:
        if self:
            return abs(Monzo.gcd(*tuple(filter(lambda x: x != 0, self))))
        else:
            return 1

    @property
    def identity(self) -> Type["Monzo"]:
        if self:
            filtered = type(self)([1 / self.lv] * len(self), self.val_border)
            monzo = tuple(int(x) for x in self * filtered)
            return type(self)(monzo, self.val_border)
        else:
            return type(self)([], self.val_border)

    @property
    def past(self) -> tuple:
        identity = self.identity
        lv = self.lv
        return tuple(type(self)(
            identity.scalar(i), self.val_border) for i in range(lv))

    @property
    def is_root(self) -> bool:
        test = Monzo(self._vector, 1)
        test.val_border = 2
        if test:
            return False
        else:
            return True

    @property
    def virtual_root(self) -> Type["Monzo"]:
        return type(self).from_ratio(1, self.ratio.denominator)

    @property
    def is_symmetric(self):
        absolute = abs(self)
        maxima = max(absolute)
        return all(x == maxima for x in filter(lambda x: x != 0, absolute))

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

    def __abs__(self):
        monzo = tuple(abs(v) for v in iter(self))
        return type(self)(monzo, val_border=self.val_border)

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
            iterable[n - 1] = 0
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

    def differential(self, other):
        """calculates differential tone between pitch and other pitch"""
        diff_ratio = abs((self.ratio * self.multiply) - (
            other.ratio * other.multiply))
        return type(self).from_ratio(
            diff_ratio.numerator, diff_ratio.denominator)


class JIContainer:
    def __init__(self, iterable, multiply=260):
        super(type(self), self).__init__(iterable)
        self.multiply = multiply
        self._val_border = 1

    @classmethod
    def mk_line(cls, reference, count):
        return cls([reference.scalar(i + 1) for i in range(count)])

    @property
    def avg_gender(self):
        if self:
            return sum(map(lambda b: 1 if b is True else -1,
                           self.gender)) / len(self)
        else:
            return 0

    @property
    def avg_lv(self):
        if self:
            return sum(p.lv for p in self) / len(self)
        else:
            return 0

    def set_multiply(self, arg):
        """set the multiply - argument of
        every containing pitch - element to
        the input argument."""
        for p in self:
            if p is not None:
                p.multiply = arg

    def set_muliplied_multiply(self, arg):
        """set the multiply - argument of
        every containing pitch - element to itself
        multiplied with the input argument."""
        for p in self:
            if p is not None:
                p.multiply *= arg

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

    def summed_summed(self):
        return functools.reduce(lambda x, y: x + y, self.summed())

    def count_root(self):
        return sum(map(lambda p: 1 if p.is_root else 0, self))

    def count_identities(self):
        am = 0
        container = []
        for x in self.identity:
            if x not in container:
                container.append(x)
                am += 1
        return am

    @property
    def dominant_prime(self):
        summed = abs(functools.reduce(lambda x, y: x + y, self))
        if summed:
            return summed.val[summed.index(max(summed))]
        else:
            return 1

    def remove(self, pitch):
        data = [p for p in self if p != pitch]
        copied = type(self)(data, self.multiply)
        copied.val_border = self.val_border
        return copied

    def find_by(self, pitch, compare_function) -> Type["JIPitch"]:
        """This method compares every pitch of a Container-Object
        with the arguments pitch through the compare_function.
        The compare_function shall return a float or integer number.
        This float or integer number represents the fitness of the specific
        pitch. The return-value is the pitch with the lowest fitness."""
        result = tuple((p, compare_function(p, pitch)) for p in self)
        if result:
            result_sorted = sorted(result, key=lambda el: el[1])
            return result_sorted[0][0]

    def find_by_walk(self, pitch, compare_function) -> Type["JIContainer"]:
        """Iterative usage of the find_by - method. The input pitch
        is the first argument, the resulting pitch is next input Argument etc.
        until the Container might be empty."""
        test_container = self.copy()
        result = [pitch]
        while len(test_container):
            pitch = test_container.find_by(pitch, compare_function)
            test_container = test_container.remove(pitch)
            result.append(pitch)
        return type(self)(result)

    def map(self, function):
        return type(self)((function(x) for x in self))

    def add_map(self, pitch):
        return self.map(lambda x: x + pitch)

    def sub_map(self, pitch):
        return self.map(lambda x: x - pitch)


class JIMel(JIPitch.mk_iterable(mel.Mel), JIContainer):
    def __init__(self, iterable, multiply=260):
        JIContainer.__init__(self, iterable, multiply)

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def intervals(self) -> "JIMel":
        """return intervals between single notes"""
        return self[1:].sub(self[:-1])

    @classmethod
    def mk_line_and_inverse(cls, reference, count):
        m0 = cls.mk_line(reference, count)
        return m0 + m0.inverse()

    def __getitem__(self, idx):
        res = muobjects.MUList.__getitem__(self, idx)
        if type(res) == type(self):
            res.multiply = self.multiply
            res.val_border = self.val_border
        return res

    def add(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 + m1 for m0, m1 in zip(self, other)))

    def sub(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 - m1 for m0, m1 in zip(self, other)))

    def mul(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 * m1 for m0, m1 in zip(self, other)))

    def div(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 / m1 for m0, m1 in zip(self, other)))

    def remove(self, pitch):
        return JIContainer.remove(self, pitch)

    @property
    def val_border(self) -> int:
        return self[0].val_border

    @val_border.setter
    def val_border(self, arg) -> None:
        for f in self:
            f.val_border = arg
        self._val_border = arg

    @property
    def pitch_rate(self):
        container = []
        frequency = []
        for t in self:
            if t in container:
                frequency[container.index(t)] += 1
            else:
                container.append(t)
                frequency.append(1)
        return tuple(zip(container, frequency))

    @property
    def pitch_rate_sorted(self):
        return tuple(sorted(self.pitch_rate, key=lambda obj: obj[1]))

    @property
    def different_pitches(self):
        return tuple(zip(*self.pitch_rate))[0]

    @property
    def lv_difference(self):
        return tuple(abs(t0.lv - t1.lv)
                     for t0, t1 in zip(self, self[1:]))

    @property
    def most_common_pitch(self):
        return self.pitch_rate_sorted[-1][0]

    @property
    def least_common_pitch(self):
        return self.pitch_rate_sorted[0][0]

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

    def count_property(self, test):
        counter = 0
        for t0, t1 in zip(self, self[1:]):
            if test(t0, t1):
                counter += 1
        return counter

    def count_repeats(self):
        return self.count_property(lambda x, y: x == y)

    def count_related(self):
        return self.count_property(lambda x, y: x.is_related(y))

    def count_congeneric(self):
        return self.count_property(lambda x, y: x.is_congeneric(y))

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

    @property
    def intervals(self):
        """return all present intervals between single notes"""
        intervals = JIHarmony([])
        for p0 in self:
            for p1 in self:
                if p0 != p1:
                    if p0 > p1:
                        interval = p0 - p1
                    else:
                        interval = p1 - p0
                        intervals.add(interval)
        return intervals

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    def remove(self, pitch):
        return JIContainer.remove(self, pitch)

    @classmethod
    def mk_line_and_inverse(cls, reference, count):
        m0 = cls.mk_line(reference, count)
        return m0 | m0.inverse()

    @classmethod
    def mk_harmonic_series(cls, p, max):
        root = p.virtual_root
        return cls([root + JIPitch.from_ratio(i, 1) for i in range(max)])

    @property
    def root(self):
        ls = list(self)
        distance = []
        for t in ls:
            local_distance = 0
            for t_comp in ls:
                if t != t_comp:
                    local_distance += (t - t_comp).summed()
            distance.append(local_distance)
        minima = (c for c, d in enumerate(distance) if d == min(distance))
        return tuple(ls[c] for c in minima)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def val_border(self) -> int:
        for x in self:
            return x.val_border

    @val_border.setter
    def val_border(self, arg) -> None:
        for f in self:
            f.val_border = arg
        self._val_border = arg

    def converted2root(self):
        root = self.root
        if root:
            root = self.root[0]
            return JIHarmony(t - root for t in self)
        else:
            return JIHarmony([])

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

    def components_harmony(self):
        return JIHarmony(functools.reduce(
            lambda x, y: x + y, tuple(
                tone.components + (tone,) for tone in self)))

    def inverse_harmony(self):
        return self.inverse() | self

    def symmetric_harmony(self, *shift):
        return JIHarmony(functools.reduce(
            lambda x, y: x | y, tuple(self.shift(s) for s in shift)))

    def past_harmony(self):
        return JIHarmony(functools.reduce(
            lambda x, y: x + y, tuple(tone.past + (tone,) for tone in self)))


class JICadence(JIPitch.mk_iterable(mel.Cadence), JIContainer):
    def __init__(self, iterable: List[JIHarmony], multiply: int = 1) -> None:
        super(type(self), self).__init__(iterable)
        self.multiply = multiply
        self._val_border = 1

    @property
    def val_border(self) -> int:
        return self[0].val_border

    @val_border.setter
    def val_border(self, arg) -> None:
        for f in self:
            f.val_border = arg
        self._val_border = arg

    @property
    def root(self):
        return tuple(h.root for h in self)

    @property
    def intervals(self):
        return tuple(h.intervals for h in self)

    def calc(self, factor=1) -> tuple:
        return tuple(h.calc(self.multiply * factor) for h in self)

    def summed(self):
        return tuple(h.summed() for h in self)

    def summed_summed(self):
        return tuple(h.summed_summed() for h in self)

    @property
    def float(self):
        return tuple(h.float for h in self)

    @property
    def gender(self):
        return tuple(h.gender for h in self)

    @property
    def identity(self):
        return tuple(h.identity for h in self)

    @property
    def identity_adjusted(self):
        return tuple(h.identity_adjusted for h in self)

    @property
    def lv(self):
        return tuple(h.lv for h in self)

    @property
    def past(self):
        return tuple(h.past for h in self)

    @property
    def ratio(self):
        return tuple(h.ratio for h in self)

    @property
    def empty_chords(self):
        return tuple(i for i, chord in enumerate(self) if not chord)

    @property
    def has_empty_chords(self):
        if self.empty_chords:
            return True
        else:
            return False

    @property
    def length(self):
        return tuple(len(h) for h in self)

    @property
    def virtual_root(self):
        return tuple(h.virtual_root for h in self)

    @property
    def adjusted_register(self):
        return type(self)([h.adjusted_register for h in self])

    def without_nulls(self):
        return type(self)(h for h in self if h)

    def dot_sum(self):
        return tuple(h.dot_sum() for h in self)

    def count_repeats(self) -> int:
        repeats = 0
        for h0, h1 in zip(self, self[1:]):
            if h0 == h1:
                repeats += 1
        return repeats


class JIScale(JIPitch.mk_iterable(mel.Scale), JIContainer):
    _period_cls = JIMel

    def __init__(self, period, periodsize):
        mel.Scale.__init__(self, period, periodsize)

    @property
    def intervals(self):
        plus_period = JIMel(self)
        return JIMel.sub(plus_period[1:], plus_period[:-1])

    def map(self, function):
        return type(self)((function(x) for x in self.period),
                          function(self.periodsize))

    def index(self, item):
        for c, x in enumerate(self):
            if x == item:
                return c
        raise ValueError("x not in tuple")


"""
    syntactic sugar for the creation of JIPitch - Objects:
"""


def r(num, den, val_border=1, multiply=1):
    return JIPitch.from_ratio(num, den, val_border, multiply)


def m(*num, val_border=1, multiply=1):
    return JIPitch.from_monzo(*num, val_border=val_border, multiply=multiply)
