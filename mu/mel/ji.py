import collections
import functools
import itertools
import json
import math
import operator
import string

from typing import Callable
from typing import List
from typing import Type

import primesieve

from mu.mel import abstract
from mu.mel import mel
from mu.utils import prime_factors
from mu.utils import tools

try:
    from quicktions import Fraction
except ImportError:
    from fractions import Fraction


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


CONCERT_PITCH = 440  # default concert pitch on a=440 Hz


class Monzo(object):
    r"""A Monzo is a representation or notation of a musical interval in just intonation.

    It may benamed after the American composer Joe Monzo
    (http://xenharmonic.wikispaces.com/Monzos).

    A Monzo could be understood as a vector, which contains
    exponents for prime numbers. The corresponding prime numbers
    are saved in a similar vector called "val". Hence, every Monzo
    object contains a "val" - property.
    If the val of a Monzo - Object would be (2, 3, 5) and
    the Monzo of the Object would be (-2, 0, 1) the resulting
    interval in p/q - notation would be 5/4, since
    2^-2 * 3^0 * 5^1 = 1/4 * 1 * 5 = 5/4.
    If you write the monzo-vector straight over the val-vector
    the relationship between both becomes clear:
    (-2, 0, 1)
    (+2, 3, 5)
    You might generate a Monzo - Object through passing an
    iterable (tuple, list) containing the wished exponents
    to the Monzo class:
    >>> m0 = Monzo((-2, 0,, 1))
    >>> m0.ratio
    Fraction(5, 4)

    In some music, a couple of Primes are ignored, meaining
    two pitches are considered as belonging to the same pitch class,
    no matter whether these ignored primes are contained
    by one of theses pitches or not.
    For instance in most western music two pitches are considered
    equal if there is an octave difference between both
    (same pitch class in different registers). For this
    music the prime number 2 makes no difference in meaning.
    For a proper representation of intervals in such a tuning
    Monzo - Objects contain a val_border - property. The
    val_border property marks the first prime, which shall be
    ignored by the object. You could pass the val_border as
    a second argument to the Monzo - class.
    >>> m0 = Monzo((0, 1), val_border=2)
    >>> m0.ratio
    Fraction(5, 4)
    """

    _val_shift = 0

    def __init__(self, iterable, val_border=1):
        self._vector = Monzo._init_vector(iterable, val_border)
        self.val_border = val_border

    def __hash__(self):
        return hash((self._val_shift, self._vec))

    @classmethod
    def from_ratio(
        cls, num: int, den: int, val_border: int = 1, multiply: float = None
    ) -> Type["Monzo"]:
        if multiply is None:
            multiply = CONCERT_PITCH
        obj = cls(cls.ratio2monzo(Fraction(num, den), cls._val_shift))
        obj.val_border = val_border
        obj.multiply = multiply
        return obj

    @classmethod
    def from_str(cls, string) -> Type["Monzo"]:
        if string == "NoPitch":
            return mel.EmptyPitch()
        else:
            num, den = string.split("/")
            return cls.from_ratio(int(num), int(den))

    @classmethod
    def from_monzo(
        cls, *arg, val_border: int = 1, multiply: float = None
    ) -> Type["Monzo"]:
        if multiply is None:
            multiply = CONCERT_PITCH
        obj = cls(arg, val_border)
        obj.multiply = multiply
        return obj

    @classmethod
    def from_json(cls, data) -> Type["Monzo"]:
        arg = data[0]
        val_border = data[1]
        obj = cls(arg, val_border)
        return obj

    @classmethod
    def load_json(cls, name: str) -> Type["Monzo"]:
        with open(name, "r") as f:
            encoded = json.loads(f.read())
        return cls.from_json(encoded)

    def convert2json(self):
        return json.dumps((self._vec, self.val_border))

    def export2json(self, name: str):
        with open(name, "w") as f:
            f.write(self.convert2json())

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
        return self._vector[self._val_shift :]

    def __repr__(self):
        return repr(self._vec)

    def __len__(self):
        return len(self._vec)

    def __bool__(self):
        # return bool(self._vec)
        return True

    def index(self, arg):
        return self._vec.index(arg)

    @staticmethod
    def _init_vector(iterable, val_border):
        return Monzo.discard_nulls(
            Monzo._shift_vector(tuple(iterable), Monzo.count_primes(val_border))
        )

    @staticmethod
    def adjusted_monzos(m0, m1):

        r"""Adjust two Monzos, e.g. makes their length equal.

        The length of the longer Monzo is the reference.

        Arguments:
            * m0: first monzo to adjust
            * m1: second monzo to adjust
        >>> m0 = (1, 0, -1)
        >>> m1 = (1,)
        >>> m0_adjusted, m1_adjusted = Monzo.adjusted_monzos(m0, m1)
        >>> m0_adjusted
        (1, 0, -1)
        >>> m1_adjusted
        (1, 0, 0)
        """

        m0 = m0._vec
        m1 = m1._vec
        l0 = len(m0)
        l1 = len(m1)
        if l0 > l1:
            return m0, m1 + (0,) * (l0 - l1)
        else:
            return m0 + (0,) * (l1 - l0), m1

    @staticmethod
    def is_comparable(m0: "Monzo", m1: "Monzo") -> bool:
        r"""Check whether two Monzo - Objects are comparable.

        Arguments:
            * m0: first monzo to compare
            * m1: second monzo to compare
        >>> m0 = Monzo((1, 0, -1), val_border=1)
        >>> m1 = Monzo((1,), val_border=2)
        >>> m2 = Monzo((1, 0, -1), val_border=2)
        >>> Monzo.is_comparable(m0, m1)
        False
        >>> Monzo.is_comparable(m1, m2)
        True
        """

        try:
            return m0._val_shift == m1._val_shift
        except AttributeError:
            return False

    @staticmethod
    def calc_iterables(iterable0: iter, iterable1: iter, operation: callable) -> iter:
        r"""Return a new generator - object

        Its elements are the results of a input function ('operation'), applied on
        every pair of zip(iterable0, iterable1).

        Arguments:
            * iterable0: first iterable to calculate
            * iterable1: second iterable to calculate
            * operation: function with two input arguments,
              to call on the first element of iterable0 and the first element
              of iterable1, on the seconds element of iterable0
              and the second element on iterable1, ...

        >>> tuple0 = (1, 0, 2, 3)
        >>> tuple1 = (2, 1, -3, 3)
        >>> tuple2 = (4,)
        >>> plus = lambda x, y: x + y
        >>> minus = lambda x, y: x - y
        >>> Monzo.calc_iterables(tuple0, tuple1, plus)
        <generator object <genexpr> at 0x7fb74d087468>
        >>> tuple(Monzo.calc_iterables(tuple0, tuple1, plus))
        (3, 1, -1, 6)
        >>> tuple(Monzo.calc_iterables(tuple0, tuple1, minus))
        (-1, -1, 5, 0)
        >>> tuple(Monzo.calc_iterables(tuple0, tuple2, plus))
        (5,)
        """

        return (operation(x, y) for x, y in zip(iterable0, iterable1))

    @staticmethod
    def adjust_ratio(r: Fraction, val_border: int) -> Fraction:
        r"""Multiply or divide a Fraction - Object with the val_border - argument,

        until it is equal or bigger than 1 and smaller than val_border.

        Arguments:
            * r: The Ratio, which shall be adjusted
            * val_border
        >>> ratio0 = Fraction(1, 3)
        >>> ratio1 = Fraction(8, 3)
        >>> val_border = 2
        >>> Monzo.adjust_ratio(ratio0, val_border)
        Fraction(4, 3)
        >>> Monzo.adjust_ratio(ratio1, val_border)
        Fraction(4, 3)

        """

        if val_border > 1:
            while r >= val_border:
                r /= val_border
            while r < 1:
                r *= val_border
        return r

    @staticmethod
    def adjust_float(f: float, val_border: int) -> float:
        r"""Multiply float with val_border, until it is <= 1 and > than val_border.

        Arguments:
            * r: The Ratio, which shall be adjusted
            * val_border
        >>> float0 = 0.5
        >>> float1 = 2
        >>> val_border = 2
        >>> Monzo.adjust_ratio(float0, val_border)
        1
        >>> Monzo.adjust_ratio(float1, val_border)
        1
        """

        if val_border > 1:
            while f > val_border:
                try:
                    f /= val_border
                except OverflowError:
                    f //= val_border
            while f < 1:
                f *= val_border
        return f

    @staticmethod
    def adjust_monzo(vector: tuple, val: tuple, val_border: int) -> tuple:
        r"""Adjust a vector and its val depending on the val_border.

        Arguments:
            * vector: The monzo, which shall be adjusted
            * val: Its corresponding val
            * val_border
        >>> vector0 = (1,)
        >>> val0 = (3,)
        >>> val_border = 2
        >>> Monzo.adjust_monzo(vector0, val0, val_border)
        ((-1, 1), (2, 3))

        """  # TODO(DOCSTRING) Make proper description what actually happens

        if vector:
            if val_border > 1:
                multiplied = functools.reduce(
                    operator.mul, (p ** e for p, e in zip(val, vector))
                )
                res = math.log(val_border / multiplied, val_border)
                if res < 0:
                    res -= 1
                res = int(res)
                val = (val_border,) + val
                vector = (res,) + vector
            return vector, val
        return (1,), (1,)

    @staticmethod
    def discard_nulls(iterable):
        r"""Discard all zeros after the last not 0 - element of an arbitary iterable.

        Return a tuple.
        Arguments:
            * iterable: the iterable, whose 0 - elements shall
              be discarded
        >>> tuple0 = (1, 0, 2, 3, 0, 0, 0)
        >>> ls = [1, 3, 5, 0, 0, 0, 2, 0]
        >>> Monzo.discard_nulls(tuple0)
        (1, 0, 2, 3)
        >>> Monzo.discard_nulls(ls)
        (1, 3, 5, 0, 0, 0, 2)
        """

        iterable = tuple(iterable)
        c = 0
        for i in reversed(iterable):
            if i != 0:
                break
            c += 1
        if c != 0:
            return iterable[:-c]
        return iterable

    @staticmethod
    def monzo2pair(monzo: tuple, val: tuple) -> tuple:
        r"""Transform a Monzo to a (numerator, denominator) - pair (two element tuple).

        Arguments are:
            * Monzo -> The exponents of prime numbers
            * Val -> the referring prime numbers
        >>> myMonzo0 = (1, 0, -1)
        >>> myMonzo1 = (0, 2, 0)
        >>> myVal0 = (2, 3, 5)
        >>> myVal1 = (3, 5, 7)
        >>> Monzo.monzo2pair(myMonzo0, myVal0)
        (2, 5)
        >>> Monzo.monzo2pair(myMonzo0, myVal1)
        (3, 7)
        >>> Monzo.monzo2pair(myMonzo1, myVal1)
        (25, 1)
        """

        numerator = 1
        denominator = 1
        for number, exponent in zip(val, monzo):
            if exponent > 0:
                numerator *= pow(number, exponent)
            elif exponent < 0:
                denominator *= pow(number, -exponent)
        return numerator, denominator

    @staticmethod
    def monzo2ratio(monzo: tuple, _val: tuple, _val_shift: int) -> Fraction:
        r"""Transform a Monzo to a Fraction - Object

        (if installed to a quicktions.Fraction - Object,
        otherwise to a fractions.Fraction - Object).

        Arguments are:
            * Monzo -> The exponents of prime numbers
            * _val -> the referring prime numbers for the underlying
                      ._vector - Argument (see Monzo._vector).
            * _val-shift -> how many prime numbers shall be skipped
                            (see Monzo._val_shift)
        >>> myMonzo0 = (1, 0, -1)
        >>> myMonzo1 = (0, 2, 0)
        >>> myVal0 = (2, 3, 5)
        >>> myVal1 = (2, 3, 5, 7)
        >>> myValShift0 = 0
        >>> myValShift1 = 1
        >>> Monzo.monzo2ratio(myMonzo0, myVal0, myValShift0)
        2/5
        >>> Monzo.monzo2ratio(myMonzo0, myVal1, myValShift1)
        12/7
        >>> Monzo.monzo2ratio(myMonzo1, myVal1, myValShift1)
        25/16
        """

        if _val_shift > 0:
            val_border = _val_shift - 1
            try:
                val_border = _val[val_border]
            except IndexError:
                val_border = 1
        else:
            val_border = 1
        num, den = Monzo.monzo2pair(monzo, _val[_val_shift:])
        return Monzo.adjust_ratio(Fraction(num, den), val_border)

    @staticmethod
    def monzo2float(monzo: tuple, _val: tuple, _val_shift: int) -> float:
        r"""Transform a Monzo to a float.

        Arguments are:
            * Monzo -> The exponents of prime numbers
            * _val -> the referring prime numbers for the underlying
                      ._vector - Argument (see Monzo._vector).
            * _val-shift -> how many prime numbers shall be skipped
                            (see Monzo._val_shift)
        >>> myMonzo0 = (1, 0, -1)
        >>> myMonzo1 = (0, 2, 0)
        >>> myVal0 = (2, 3, 5)
        >>> myVal1 = (2, 3, 5, 7)
        >>> myValShift0 = 0
        >>> myValShift1 = 1
        >>> Monzo.monzo2ratio(myMonzo0, myVal0, myValShift0)
        0.4
        >>> Monzo.monzo2ratio(myMonzo0, myVal1, myValShift1)
        1.7142857142857142
        >>> Monzo.monzo2ratio(myMonzo1, myVal1, myValShift1)
        1.5625
        """

        if _val_shift > 0:
            val_border = _val_shift - 1
            try:
                val_border = _val[val_border]
            except IndexError:
                val_border = 1
        else:
            val_border = 1
        num, den = Monzo.monzo2pair(monzo, _val[_val_shift:])
        try:
            calc = num / den
        except OverflowError:
            calc = num // den
        return Monzo.adjust_float(calc, val_border)

    @staticmethod
    def ratio2monzo(ratio: Fraction, val_shift=0) -> Type["Monzo"]:
        r"""Transform a Fraction - Object to a Monzo.

        Arguments are:
            * ratio -> The Fraction, which shall be transformed
            * val_shift -> how many prime numbers shall be skipped
                           (see Monzo._val_shift)
        >>> try:
        >>>     from quicktions import Fraction
        >>> except ImportError:
        >>>     from fractions import Fraction
        >>> myRatio0 = Fraction(3, 2)
        >>> myRatio1 = Fraction(7, 6)
        >>> myValShift0 = 0
        >>> myValShift1 = 1
        >>> Monzo.ratio2monzo(myRatio0, myValShift0)
        (-1, 1)
        >>> Monzo.ratio2monzo(myRatio0, myValShift1)
        (1,)
        >>> Monzo.monzo2ratio(myRatio1, myValShift1)
        (-1, 0, 1)
        """

        gen_pos = prime_factors.factors(ratio.numerator)
        gen_neg = prime_factors.factors(ratio.denominator)

        factorised_num = prime_factors.factorise(ratio.numerator)
        factorised_den = prime_factors.factorise(ratio.denominator)

        biggest_prime = max(factorised_num + factorised_den)
        monzo = [0] * Monzo.count_primes(biggest_prime)

        for num, fac in gen_pos:
            if num > 1:
                monzo[Monzo.count_primes(num) - 1] += fac

        for num, fac in gen_neg:
            if num > 1:
                monzo[Monzo.count_primes(num) - 1] -= fac

        return Monzo(monzo[val_shift:])

    @staticmethod
    def _shift_vector(vec, shiftval) -> tuple:
        r"""Add Zeros to the beginning of a tuple / discard elements from a tuple.

        Arguments are:
            * vec -> The tuple, which shall be modified
            * shiftval -> how many elements shall be shifted
        >>> myVec0 = (0, 1, -1)
        >>> myVec1 = (1, -1, 1)
        >>> Monzo._shift_vector(myVec0, 0)
        (0, 1, -1)
        >>> Monzo._shift_vector(myVec0, 1)
        (0, 0, 1, -1)
        >>> Monzo._shift_vector(myVec0, -1)
        (1, -1)
        >>> Monzo._shift_vector(myVec1, -2)
        (1,)
        """

        if shiftval > 0:
            m = (0,) * shiftval + tuple(vec)
        else:
            m = tuple(vec[abs(shiftval) :])
        return m

    @staticmethod
    def gcd(*args) -> int:
        r"""Calculate the greatest common denominator of many numbers.

        Arguments:
            * arg -> numbers, whose greatest common denominator shall
                     be found

        >>> Monzo.gcd(4, 8)
        4
        >>> Monzo.gcd(4, 8, 3)
        1
        >>> Monzo.gcd(64, 100, 400)
        4
        """

        return functools.reduce(math.gcd, args)

    @staticmethod
    def nth_prime(arg):
        r"""Find the nth - prime.

        More efficient version than primesieve.nth_primes,
        since it uses saved Primes for n < 50.

        Arguments:
            * n -> number, which Prime shall be found

        >>> Monzo.nth_prime(3)
        5
        >>> Monzo.nth_prime(10)
        29
        """

        try:
            primes = (
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                97,
                101,
                103,
                107,
                109,
                113,
                127,
                131,
                137,
                139,
                149,
                151,
                157,
                163,
                167,
                173,
                179,
                181,
                191,
                193,
                197,
                199,
                211,
                223,
                227,
                229,
            )
            return primes[arg]
        except IndexError:
            return primesieve.nth_prime(arg)

    @staticmethod
    def n_primes(arg):
        r"""List the first n - primes.

        More efficient version than primesieve.n_primes,
        since it uses saved Primes for n < 50.

        Arguments:
            * n -> number, which Prime shall be found

        >>> Monzo.nth_prime(3)
        (2, 3, 5)
        >>> Monzo.nth_prime(10)
        (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
        """

        if arg <= 50:
            return Monzo.nth_prime(slice(0, arg))
        else:
            return primesieve.n_primes(arg)

    @staticmethod
    def count_primes(arg):
        r"""Count prime numbers.

        More efficient version than primesieve.count_primes,
        since it uses saved Primes for n < 70.

        Arguments:
            * n -> number, which Prime shall be found

        >>> Monzo.nth_prime(3)
        (2, 3, 5)
        >>> Monzo.nth_prime(10)
        (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
        """

        if arg <= 70:
            data = (
                0,
                0,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                4,
                4,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                8,
                8,
                8,
                8,
                9,
                9,
                9,
                9,
                9,
                9,
                10,
                10,
                11,
                11,
                11,
                11,
                11,
                11,
                12,
                12,
                12,
                12,
                13,
                13,
                14,
                14,
                14,
                14,
                15,
                15,
                15,
                15,
                15,
                15,
                16,
                16,
                16,
                16,
                16,
                16,
                17,
                17,
                18,
                18,
                18,
                18,
                18,
                18,
                19,
                19,
                19,
            )
            return data[arg]
        else:
            return primesieve.count_primes(arg)

    @staticmethod
    def indigestibility(num: int) -> float:
        """Calculate indigestibility of a number

        The implementation follows Clarence Barlows definition
        given in 'The Ratio Book' (1992).
        Arguments:
            * num -> integer, whose indigestibility value shall be calculated

        >>> Monzo.indigestibility(1)
        0
        >>> Monzo.indigestibility(2)
        1
        >>> Monzo.indigestibility(3)
        2.6666666666666665
        """

        decomposed = prime_factors.factorise(num)
        return Monzo.indigestibility_of_factorised(decomposed)

    @staticmethod
    def indigestibility_of_factorised(decomposed):
        decomposed = collections.Counter(decomposed)
        decomposed = zip(decomposed.values(), decomposed.keys())
        summed = ((power * pow(prime - 1, 2)) / prime for power, prime in decomposed)
        return 2 * sum(summed)

    @staticmethod
    def mk_filter_vec(*prime):
        numbers = tuple(Monzo.count_primes(p) for p in prime)
        iterable = [1] * max(numbers)
        for n in numbers:
            iterable[n - 1] = 0
        return tuple(iterable)

    @property
    def _val(self) -> tuple:
        r"""Return ascending list of primes, until the highest contained Prime.

        This Method ignores the val_border / _val_shift
        property of an Object.
        >>> m0 = Monzo((0, 1, 2), 1)
        >>> m0._val
        (2, 3, 5)
        >>> m0.val_border = 2
        >>> m0._val
        (2, 3, 5)
        >>> m1 = Monzo((0, -1, 0, 0, 1), 1)
        >>> m1._val
        (2, 3, 5, 7, 11)
        """

        return Monzo.n_primes(len(self) + self._val_shift)

    @property
    def val(self) -> tuple:
        r"""Return complete list of primes (e.g. 'val'),

        until the
        highest Prime, which the Monzo / JIPitch contains in respect
        to the current val_border or _val_shift - property of the object.

        >>> m0 = Monzo((0, 1, 2), 1)
        >>> m0.val
        (2, 3, 5)
        >>> m0.val_border = 2
        >>> m0.val
        (3, 5)
        >>> m0.val_border = 3
        >>> m0.val
        (5,)
        >>> m1 = Monzo((0, -1, 0, 0, 1), 1)
        >>> m1._val
        (2, 3, 5, 7, 11)
        """

        return self._val[self._val_shift :]

    @property
    def val_border(self) -> int:
        r"""The val - border property denotes the first Prime, which shall be ignored.

        Speaking in terms of scales, it actually descripes the period
        of a scale; in most known scales the period may be an octave,
        e.g. after an octave the same pitch classes are repeating.
        For example D3 and a D4 are seen as the same pitch class,
        just in different registers. Intervals of such a scale would
        set their val_border to 2, since they are octavely repeating.
        If pitch classes are never repeating, meaning that every pitch class of
        every register is understood as an indidual pitch,
        the correct val_border - argument may be 1.
        In scales with a period of an octave plus a fifth (for example
        the famous Bohlen-Pierce-Scale) the proper value for the
        val_border would be 3 (third Harmonic Tone is an octave plus
        a fifth).

        >>> m0 = Monzo((0, 1), 1)
        >>> m0.ratio
        Fraction(3, 1)
        >>> m0.val_border = 2
        >>> m0.ratio
        Fraction(3, 2)
        >>> m0.val_border = 3
        >>> m0.ratio
        Fraction(1, 1)
        """

        if self._val_shift == 0:
            return 1
        else:
            return Monzo.nth_prime(self._val_shift - 1)

    @val_border.setter
    def val_border(self, v: int):
        self._val_shift = Monzo.count_primes(v)

    def set_val_border(self, val_border: int) -> Type["Monzo"]:
        """Return a copied version of the Monzo / JIPitch -Object with a new val_border.

        The Object itself stay unchanged.

        Arguments:
            * val_border: The val_border for the new Monzo

        >>> m0 = Monzo((0, 1,), val_border=2)
        >>> m1 = m0.set_val_border(3)
        >>> m1.val_border
        3
        >>> m0.val_border
        2
        """

        copied = self.copy()
        copied.val_border = val_border
        return copied

    @property
    def factorised(self) -> tuple:
        """Return factorised / decomposed version of itsef.

        >>> m0 = Monzo((0, 1,), val_border=2)
        >>> m0.factorised
        (2, 2, 5)
        >>> m1 = Monzo.from_ratio(7, 6)
        >>> m1.factorised
        (2, 3, 7)
        """

        vec = self._vec
        val = self.val
        border = self.val_border
        vec_adjusted, val_adjusted = type(self).adjust_monzo(vec, val, border)
        decomposed = ([p] * abs(e) for p, e in zip(val_adjusted, vec_adjusted))
        return tuple(functools.reduce(operator.add, decomposed))

    @property
    def factorised_numerator_and_denominator(self) -> tuple:
        vec = self._vec
        val = self.val
        border = self.val_border
        vec_adjusted, val_adjusted = type(self).adjust_monzo(vec, val, border)
        num_den = [[[]], [[]]]
        for p, e in zip(val_adjusted, vec_adjusted):
            if e > 0:
                idx = 0
            else:
                idx = 1
            num_den[idx].append([p] * abs(e))
        return tuple(
            functools.reduce(operator.add, decomposed) for decomposed in num_den
        )

    @property
    def blueprint(self, ignore: tuple = (2,)) -> tuple:
        blueprint = []
        for factorised in self.factorised_numerator_and_denominator:
            factorised = tuple(fac for fac in factorised if fac not in ignore)
            counter = collections.Counter(collections.Counter(factorised).values())
            if counter:
                maxima = max(counter.keys())
                blueprint.append(tuple(counter[idx + 1] for idx in range(maxima)))
            else:
                blueprint.append(tuple([]))
        return tuple(blueprint)

    @property
    def ratio(self) -> Fraction:
        """Return the Monzo transformed to a Ratio (Fraction-Object).

        >>> m0 = Monzo((0, 1,), val_border=2)
        >>> m0.ratio
        Fraction(5, 4)
        >>> m0 = Monzo.from_ratio(3, 2)
        >>> m0.ratio
        Fraction(3, 2)
        """

        return Monzo.monzo2ratio(self, self._val, self._val_shift)

    @property
    def numerator(self) -> int:
        """Return the numerator of a Monzo or JIPitch - object.

        This metod ignores the val_border - property of the object,
        meaning that myMonzo.ratio.numerator != myMonzo.numerator.

        >>> m0 = Monzo((0, -1,), val_border=2)
        >>> m0.numerator
        1
        >>> m0.ratio
        Fraction(8, 5)
        >>> m0.ratio.numerator
        8
        """

        numerator = 1
        for number, exponent in zip(self.val, self):
            if exponent > 0:
                numerator *= pow(number, exponent)
        return numerator

    @property
    def denominator(self) -> int:
        """Return the denominator of a Monzo or JIPitch - object.

        This metod ignores the val_border - property of the object,
        meaning that myMonzo.ratio.denominator != myMonzo.denominator.

        >>> m0 = Monzo((0, 1,), val_border=2)
        >>> m0.denominator
        1
        >>> m0.ratio
        Fraction(5, 4)
        >>> m0.ratio.denominator
        4
        """

        denominator = 1
        for number, exponent in zip(self.val, self):
            if exponent < 0:
                denominator *= pow(number, -exponent)
        return denominator

    @property
    def float(self) -> float:
        """Return the float of a Monzo or JIPitch - object.

        These are the same: float(myMonzo.ratio) == myMonzo.float. Note the
        difference that the second version might be slightly
        more performant.

        >>> m0 = Monzo((1,), val_border=2)
        >>> m0.float
        1.5
        >>> float(m0.ratio)
        1.5
        """

        return Monzo.monzo2float(self, self._val, self._val_shift)

    @property
    def cents(self) -> float:
        return self.ratio2ct(self.ratio)

    def __float__(self) -> float:
        return float(self.float)

    def simplify(self):
        """Set all elements in self._vector to 0 with a bigger index than self._val_shift.

        >>> monzo0 = Monzo((1, -1), val_border=1)
        >>> monzo0.val_border = 2
        >>> monzo0._vector
        (1, -1)
        >>> monzo0_simplified = monzo0.simplify()
        >>> monzo0._vector
        (0, -1)

        """

        v_shift = self._val_shift
        new = self.copy()
        if v_shift > 0:
            new._vector = (0,) * v_shift + self._vec
        return new

    def adjust_register(
        self,
        limitup: float = 2 ** 6,
        startperiod: int = 3,
        concert_pitch_period: int = 3,
    ):
        """Adjust register of the Interval. Change the val_border to 1.

        Arguments:
            * startperiod
            * limitup
            * concert_pitch_period

        >>> monzo0 = Monzo((1, -1), val_border=2)
        """  # TODO(adjustR): add proper description with example

        def period_generator(val_border):
            result = val_border ** startperiod
            while True:
                yield result
                result *= val_border

        v_shift = self._val_shift
        v_border = self.val_border
        if v_border == 1:
            v_border = 2
        i = 1
        periods = [i]
        while i < limitup:
            i *= v_border
            periods.append(i)
        amount_periods = len(periods)
        identity_pitch = self.identity
        if identity_pitch.gender is True:
            ratio = identity_pitch.ratio
            id_num, id_den = ratio.numerator, ratio.denominator
        else:
            identity_simplified = identity_pitch.simplify()
            identity_simplified._val_shift = 0
            id_num = identity_simplified.numerator
            id_den = identity_simplified.denominator
            while id_num * v_border < id_den:
                id_num *= v_border
        id_pitch = type(self).from_ratio(id_num, id_den, val_border=1)
        id_pitch_scaled = id_pitch.scalar(self.lv)
        id_pitch_scaled_float = id_pitch_scaled.float
        resulting_period = 0
        for i, per in enumerate(period_generator(v_border)):
            if per > id_pitch_scaled_float:
                break
            resulting_period = i
        for i, per in enumerate(period_generator(1 / v_border)):
            if per < id_pitch_scaled_float:
                if i > 0:
                    resulting_period = -i
                break
        adjusted_period = resulting_period % (amount_periods - 1)
        diff = resulting_period - adjusted_period
        sub_pitch_monzo = ((0,) * (v_shift - 1)) + (diff,)
        sub_pitch = type(self).from_monzo(*sub_pitch_monzo)
        resulting_pitch = id_pitch_scaled - sub_pitch
        concert_pitch_adjustment_diff = concert_pitch_period - startperiod
        concert_pitch_adjustment_monzo = ((0,) * (v_shift - 1)) + (
            concert_pitch_adjustment_diff,
        )
        concert_pitch_adjustment = type(self).from_monzo(
            *concert_pitch_adjustment_monzo
        )
        res_pitch_for_concert_pitch_adjusted = (
            resulting_pitch - concert_pitch_adjustment
        )
        return res_pitch_for_concert_pitch_adjusted

    def register(self, octave: int) -> "Monzo":
        n = self.normalize()
        factor = 2 ** abs(octave)
        if octave < 1:
            added = r(1, factor)
        else:
            added = r(factor, 1)
        p = n + added
        p.multiply = self.multiply
        return p

    def move_to_closest_register(self, reference: "Monzo") -> "Monzo":
        reference_register = reference.octave

        best = None
        for adaption in range(-1, 2):
            candidate = self.register(reference_register + adaption)
            difference = abs((candidate - reference).cents)
            set_best = True
            if best and difference > best[1]:
                set_best = False

            if set_best:
                best = (candidate, difference)

        best[0].multiply = self.multiply
        return best[0]

    @property
    def gender(self) -> bool:
        """Return the gender (bool) of a Monzo or JIPitch - object.

        The gender of a Monzo or JIPitch - may be True if
        the exponent of the highest occurring prime number is a
        positive number and False if the exponent is a
        negative number.
        special case: The gender of a Monzo or JIPitch - object
        containing an empty val is True.

        >>> m0 = Monzo((-2. 1), val_border=2)
        >>> m0.gender
        True
        >>> m1 = Monzo((-2, -1), val_border=2)
        >>> m1.gender
        False
        >>> m2 = Monzo([], val_border=2)
        >>> m2.gender
        True
        """

        if self.val:
            maxima = max(self)
            minima = min(self)
            test = (
                maxima <= 0 and minima < 0,
                minima < 0 and self.index(minima) > self.index(maxima),
            )
            if any(test):
                return False
        return True

    @property
    def strict_gender(self) -> bool:
        """Strict version of the gender property.

        Return True if all exponents are positive.

        Return False if all exponents are negative.

        Return None if the pitch contains positive
        and negative exponents.
        """
        gender = None
        for exponent in self:
            if exponent != 0:
                if gender is not None:
                    return
                else:
                    gender = exponent > 0
        return gender

    @property
    def harmonic(self) -> int:
        """Return the nth - harmonic / subharmonic the pitch may represent.

        May be positive for harmonic and negative for
        subharmonic pitches. If the return - value is 0,
        the interval may occur neither between the first harmonic
        and any other pitch of the harmonic scale nor
        between the first subharmonic in the and any other
        pitch of the subharmonic scale.

        >>> m0 = Monzo((1,), val_border=2)
        >>> m0.ratio
        Fraction(3, 2)
        >>> m0.harmonic
        3
        >>> m1 = Monzo((-1,), 2)
        >>> m1.harmonic
        -3
        """

        if self.ratio.denominator % 2 == 0:
            return self.ratio.numerator
        elif self.ratio.numerator % 2 == 0:
            return -self.ratio.denominator
        elif self.ratio == Fraction(1, 1):
            return 1
        else:
            return 0

    @property
    def primes(self) -> tuple:
        """Return all occurring prime numbers of a Monzo or JIPitch - object.

        >>> m0 = Monzo((1,), val_border=2)
        >>> m0.ratio
        Fraction(3, 2)
        >>> m0.harmonic # a fifth may be the 3th pitch of the harmonic scale
        3
        >>> m1 = Monzo((-1,), 2)
        >>> m1.harmonic # a fourth may be the 3th pitch of the subharmonic scale
        -3
        """

        p = prime_factors.factorise(self.numerator * self.denominator)
        return tuple(sorted(tuple(set(p))))

    @property
    def primes_for_numerator_and_denominator(self) -> tuple:
        return tuple(
            tuple(sorted(set(prime_factors.factorise(n))))
            for n in (self.numerator, self.denominator)
        )

    @property
    def quantity(self) -> int:
        """Count how many different prime numbers are occurring in the Object.

        >>> m0 = Monzo((1,), val_border=2)
        >>> m0.quantity
        1
        >>> m1 = Monzo((1, 1), val_border=2)
        >>> m1.quantity
        2
        >>> m2 = Monzo((1, 1, -1), val_border=2)
        >>> m2.quantity
        3
        >>> m3 = Monzo((0, 1, -1), val_border=2)
        >>> m3.quantity # one exponent is zero now
        2
        """

        return len(self.primes)

    @property
    def components(self) -> tuple:
        r"""Seperate a monzo object to its different primes.

        >>> m0 = (1, 0, -1)
        >>> m1 = (1,)
        >>> m0_adjusted, m1_adjusted = Monzo.adjusted_monzos(m0, m1)
        >>> m0
        (1, 0, -1)
        >>> m1
        (1, 0, 0)
        """

        vectors = [[0] * c + [x] for c, x in enumerate(self) if x != 0]
        return tuple(type(self)(vec, val_border=self.val_border) for vec in vectors)

    @property
    def harmonicity_wilson(self) -> int:
        decomposed = self.factorised
        return int(sum(filter(lambda x: x != 2, decomposed)))

    @property
    def harmonicity_vogel(self) -> int:
        decomposed = self.factorised
        decomposed_filtered = tuple(filter(lambda x: x != 2, decomposed))
        am_2 = len(decomposed) - len(decomposed_filtered)
        return int(sum(decomposed_filtered) + am_2)

    @property
    def harmonicity_euler(self) -> int:
        """Return the 'gradus suavitatis' of euler.

        A higher number means a less consonant interval /
        a more complicated harmony.
        euler(1/1) is definied as 1.
        >>> m0 = Monzo((1,), val_border=2)
        >>> m1 = Monzo([], val_border=2)
        >>> m2 = Monzo((0, 1,), val_border=2)
        >>> m3 = Monzo((0, -1,), val_border=2)
        >>> m0.harmonicity_euler
        4
        >>> m1.harmonicity_euler
        1
        >>> m2.harmonicity_euler
        7
        >>> m3.harmonicity_euler
        8
        """

        decomposed = self.factorised
        return 1 + sum(x - 1 for x in decomposed)

    @property
    def harmonicity_barlow(self) -> float:
        r"""Calculate the barlow-harmonicity of an interval.

        This implementation follows Clarence Barlows definition, given
        in 'The Ratio Book' (1992).

        A higher number means a more consonant interval / a less
        complicated harmony.

        barlow(1/1) is definied as infinite.

        >>> m0 = Monzo((1,), val_border=2)
        >>> m1 = Monzo([], val_border=2)
        >>> m2 = Monzo((0, 1,), val_border=2)
        >>> m3 = Monzo((0, -1,), val_border=2)
        >>> m0.harmonicity_barlow
        0.27272727272727276
        >>> m1.harmonicity_barlow # 1/1 is infinite harmonic
        inf
        >>> m2.harmonicity_barlow
        0.11904761904761904
        >>> m3.harmonicity_barlow
        -0.10638297872340426
        """

        def sign(x):
            return (1, -1)[x < 0]

        num_den_decomposed = self.factorised_numerator_and_denominator
        ind_num = Monzo.indigestibility_of_factorised(num_den_decomposed[0])
        ind_de = Monzo.indigestibility_of_factorised(num_den_decomposed[1])
        if ind_num == 0 and ind_de == 0:
            return float("inf")
        return sign(ind_num - ind_de) / (ind_num + ind_de)

    @property
    def harmonicity_simplified_barlow(self) -> float:
        r"""Calculate a simplified barlow-harmonicity of an interval.

        This implementation follows Clarence Barlows definition, given
        in 'The Ratio Book' (1992), with the difference that
        only positive numbers are returned and that (1/1) is
        defined as 1 instead of infinite.

        >>> m0 = Monzo((1,), val_border=2)
        >>> m1 = Monzo([], val_border=2)
        >>> m2 = Monzo((0, 1,), val_border=2)
        >>> m3 = Monzo((0, -1,), val_border=2)
        >>> m0.harmonicity_simplified_barlow
        0.27272727272727276
        >>> m1.harmonicity_simplified_barlow # 1/1 is not infinite but 1
        1
        >>> m2.harmonicity_simplified_barlow
        0.11904761904761904
        >>> m3.harmonicity_simplified_barlow # positive return value
        0.10638297872340426
        """

        barlow = abs(self.harmonicity_barlow)
        if barlow == float("inf"):
            return 1
        return barlow

    @property
    def harmonicity_tenney(self) -> float:
        r"""Calculate Tenneys harmonic distance of an interval

        A higher number
        means a more consonant interval / a less
        complicated harmony.

        tenney(1/1) is definied as 0.

        >>> m0 = Monzo((1,), val_border=2)
        >>> m1 = Monzo([], val_border=2)
        >>> m2 = Monzo((0, 1,), val_border=2)
        >>> m3 = Monzo((0, -1,), val_border=2)
        >>> m0.harmonicity_tenney
        2.584962500721156
        >>> m1.harmonicity_tenney
        0.0
        >>> m2.harmonicity_tenney
        4.321928094887363
        >>> m3.harmonicity_tenney
        -0.10638297872340426
        """

        ratio = self.ratio
        num = ratio.numerator
        de = ratio.denominator
        return math.log(num * de, 2)

    @property
    def lv(self) -> int:
        if self.val:
            return abs(Monzo.gcd(*tuple(filter(lambda x: x != 0, self))))
        else:
            return 1

    @property
    def identity(self) -> Type["Monzo"]:
        if self._val:
            val_border = self.val_border
            filtered = type(self)([1 / self.lv] * len(self), val_border)
            monzo = tuple(int(x) for x in self * filtered)
            return type(self)(monzo, val_border)
        else:
            return type(self)([], self.val_border)

    @property
    def past(self) -> tuple:
        identity = self.identity
        lv = self.lv
        return tuple(type(self)(identity.scalar(i), self.val_border) for i in range(lv))

    @property
    def is_root(self) -> bool:
        test = Monzo(self._vector, 1)
        test.val_border = 2
        return not any(test.val)

    @property
    def virtual_root(self) -> Type["Monzo"]:
        return type(self).from_ratio(1, self.ratio.denominator)

    @property
    def is_symmetric(self):
        absolute = type(self)(tuple(abs(v) for v in self), val_border=self.val_border)
        maxima = max(absolute)
        return all(x == maxima for x in filter(lambda x: x != 0, absolute))

    @property
    def sparsity(self):
        zero = 0
        for i in self:
            if i == 0:
                zero += 1
        try:
            return zero / len(self._vec)
        except ZeroDivisionError:
            return 0

    @property
    def density(self):
        return 1 - self.sparsity

    @comparable_bool_decorator
    def is_related(self: "Monzo", other: "Monzo") -> bool:
        """Two JIPitch - Objects are related if they share at least one common prime.

        Arguments:
            * Monzo or JIPitch to compare with

        >>> monzo0 = Monzo((1, -1), val_border=2)
        >>> monzo1 = Monzo((-1, 1), val_border=2)
        >>> monzo2 = Monzo((2, -2), val_border=2)
        >>> monzo3 = Monzo((0, -1), val_border=2)
        >>> monzo4 = Monzo((4,), val_border=2)
        >>> monzo5 = Monzo((0, 0, 2), val_border=2)
        >>> monzo0.is_related(monzo1)
        True
        >>> monzo0.is_related(monzo2)
        True
        >>> monzo0.is_related(monzo3)
        True
        >>> monzo0.is_related(monzo4)
        True
        >>> monzo0.is_related(monzo5)
        False
        """

        for p in self.primes:
            if p in other.primes:
                return True
        return False

    @comparable_bool_decorator
    def is_congeneric(self: "Monzo", other: "Monzo") -> bool:
        """Two JIPitch - Objects are congeneric if their primes are equal.

        Arguments:
            * Monzo or JIPitch to compare with

        >>> monzo0 = Monzo((1, -1), val_border=2)
        >>> monzo1 = Monzo((-1, 1), val_border=2)
        >>> monzo2 = Monzo((2, -2), val_border=2)
        >>> monzo3 = Monzo((0, -1), val_border=2)
        >>> monzo4 = Monzo((2, -1), val_border=2)
        >>> monzo0.is_congeneric(monzo1)
        True
        >>> monzo0.is_congeneric(monzo2)
        True
        >>> monzo0.is_congeneric(monzo3)
        False
        >>> monzo0.is_congeneric(monzo4)
        True
        """

        if self.primes == other.primes:
            return True
        else:
            return False

    @comparable_bool_decorator
    def is_sibling(self: "Monzo", other: "Monzo") -> bool:
        """Two JIPitch - Objects are siblings if their primes and their gender are equal.

        Arguments:
            * Monzo or JIPitch to compare with

        >>> monzo0 = Monzo((1, -1), val_border=2)
        >>> monzo1 = Monzo((-1, 1), val_border=2)
        >>> monzo2 = Monzo((2, -2), val_border=2)
        >>> monzo3 = Monzo((3, -3), val_border=2)
        >>> monzo4 = Monzo((0, -1), val_border=2)
        >>> monzo5 = Monzo((2, -1), val_border=2)
        >>> monzo0.is_sibling(monzo1)
        False
        >>> monzo0.is_sibling(monzo2)
        True
        >>> monzo0.is_sibling(monzo3)
        True
        >>> monzo0.is_sibling(monzo4)
        False
        >>> monzo0.is_sibling(monzo5)
        True
        """

        if self.primes == other.primes and self.gender == other.gender:
            return True
        else:
            return False

    def summed(self) -> int:
        return sum(map(lambda x: abs(x), self))

    def normalize(self, prime=2) -> "Monzo":
        ratio = self.ratio
        adjusted = type(self).adjust_ratio(ratio, prime)
        return type(self).from_ratio(
            adjusted.numerator, adjusted.denominator, multiply=self.multiply
        )

    def subvert(self) -> list:
        def ispos(num):
            if num > 0:
                return 1
            else:
                return -1

        sep = [
            tuple(
                type(self)([0] * counter + [ispos(vec)], self.val_border)
                for i in range(abs(vec))
            )
            for counter, vec in enumerate(self)
            if vec != 0
        ]
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

    def __eq__(self: "Monzo", other: "Monzo") -> bool:
        try:
            test0 = tuple.__eq__(self._vec, other._vec)
            test1 = self._val_shift == other._val_shift
            return test0 and test1
        except AttributeError:
            return False

    def __add__(self, other: "Monzo") -> "Monzo":
        return self.__math(other, operator.add)

    def __sub__(self, other: "Monzo") -> "Monzo":
        return self.__math(other, operator.sub)

    def __mul__(self, other) -> "Monzo":
        return self.__math(other, operator.mul)

    def __div__(self, other) -> "Monzo":
        return self.__math(other, operator.div)

    def __pow__(self, other) -> "Monzo":
        return self.__math(other, lambda x, y: x ** y)

    def __abs__(self):
        if self.numerator > self.denominator:
            return self.copy()
        else:
            monzo = tuple(-v for v in iter(self))
            return type(self)(monzo, val_border=self.val_border)

    def scalar(self, factor):
        """Return the scalar-product of a Monzo and its factor."""
        return self * type(self)((factor,) * len(self), self.val_border)

    def dot(self, other) -> float:
        """Return the dot-product of two Monzos."""
        return sum(a * b for a, b in zip(self, other))

    def intersection(self, other) -> "Monzo":
        m0, m1 = tuple(self), tuple(other)
        while len(m1) < len(m0):
            m1 += (0,)

        return type(self)(
            [p if m1[idx] != 0 else 0 for idx, p in enumerate(m0)], self.val_border
        )

    def difference(self, other) -> "Monzo":
        m0, m1 = tuple(self), tuple(other)
        while len(m1) < len(m0):
            m1 += (0,)
        return type(self)(
            [p if m1[idx] == 0 else 0 for idx, p in enumerate(m0)], self.val_border
        )

    def matrix(self, other):
        """Return the matrix-product of two Monzos."""
        m0 = tuple(
            type(self)(tuple(arg * arg2 for arg2 in other), self.val_border)
            for arg in self
        )
        m1 = tuple(
            type(self)(tuple(arg * arg2 for arg2 in self), self.val_border)
            for arg in other
        )
        return m0 + m1

    def inverse(self, axis=None) -> "Monzo":
        if axis is None:
            return type(self)(list(map(lambda x: -x, self)), self.val_border)
        else:
            distance = self - axis
            return axis - distance

    def shift(self, shiftval: int) -> "Monzo":
        return type(self)(Monzo._shift_vector(self, shiftval), self.val_border)

    def filter(self, *prime):
        iterable0 = Monzo.mk_filter_vec(*prime)[self._val_shift :]
        iterable1 = tuple(self._vec)
        while len(iterable0) < len(iterable1):
            iterable0 += (1,)
        while len(iterable1) < len(iterable0):
            iterable1 += (0,)
        iterable = Monzo.calc_iterables(iterable0, iterable1, operator.mul)
        return type(self)(iterable, self.val_border)


class JIPitch(Monzo, abstract.AbstractPitch):
    multiply = 1

    def __init__(self, iterable, val_border: int = 1, multiply: float = None):
        if multiply is None:
            multiply = CONCERT_PITCH

        Monzo.__init__(self, iterable, val_border)
        self.multiply = multiply

    def __eq__(self, other) -> bool:
        try:
            return all(
                (
                    self.multiply == other.multiply,
                    self._vec == other._vector[self._val_shift :],
                )
            )
        except AttributeError:
            return abstract.AbstractPitch.__eq__(self, other)

    def __repr__(self) -> str:
        return str(self.ratio)

    @property
    def monzo(self) -> tuple:
        return tuple(self)

    def __add__(self, other) -> "JIPitch":
        return JIPitch(
            Monzo.__add__(self, other), self.val_border, multiply=self.multiply
        )

    def __sub__(self, other) -> "JIPitch":
        return JIPitch(
            Monzo.__sub__(self, other), self.val_border, multiply=self.multiply
        )

    def __mul__(self, other) -> "JIPitch":
        return JIPitch(
            Monzo.__mul__(self, other), self.val_border, multiply=self.multiply
        )

    def __div__(self, other) -> "JIPitch":
        return JIPitch(
            Monzo.__div__(self, other), self.val_border, multiply=self.multiply
        )

    def __pow__(self, val) -> "JIPitch":
        return JIPitch(
            Monzo.__pow__(self, val), self.val_border, multiply=self.multiply
        )

    def __hash__(self):
        return hash(self._vec)
        # return abstract.AbstractPitch.__hash__(self)
        # TODO(HASH_PROBLEM): find valid solution for this problem

    def calc(self, factor=1) -> float:
        return float(self.float * self.multiply * factor)

    def copy(self) -> "JIPitch":
        return JIPitch(self, self.val_border, self.multiply)

    @classmethod
    def from_json(cls, data) -> Type["JIPitch"]:
        arg = data[0]
        val_border = data[1]
        try:
            multiply = data[2]
        except IndexError:
            multiply = 1
        obj = cls(arg, val_border)
        obj.multiply = multiply
        return obj

    @classmethod
    def from_cents(
        cls,
        cents: float,
        limit_denominator: int = 32,
        val_border: int = 1,
        multiply: float = 1,
    ) -> "JIPitch":
        ratio = cls.ct2ratio(cents)
        if limit_denominator is not None:
            ratio = ratio.limit_denominator(limit_denominator)
        return cls.from_ratio(ratio.numerator, ratio.denominator, val_border, multiply)

    def convert2json(self):
        return json.dumps((self._vec, self.val_border, self.multiply))

    def differential(self, other):
        """calculates differential tone between pitch and other pitch"""
        tests = (self != mel.EmptyPitch(), other != mel.EmptyPitch())
        if all(tests) is True:
            diff_ratio = abs(self.ratio - other.ratio)
            return type(self).from_ratio(diff_ratio.numerator, diff_ratio.denominator)
        else:
            if tests[0] is True:
                return self
            elif tests[1] is True:
                return other
            else:
                return mel.EmptyPitch()


class JIContainer(object):
    def __init__(self, iterable, multiply=260):
        super(type(self), self).__init__(iterable)
        self.multiply = multiply
        self._val_border = 1

    @staticmethod
    def from_str(cls, string: str) -> "JIContainer":
        ratios = string.split(", ")
        return cls(JIPitch.from_str(r) for r in ratios)

    @classmethod
    def from_scl(cls, name: str) -> "JIContainer":
        """Generating JIContainer from the scl file format.

        See: http://huygens-fokker.org/scala/scl_format.html
        """

        def convert2pitch(scl_line: str) -> JIPitch:
            p = scl_line.split(" ")[0]
            if p[-1] == ".":
                msg = "Can't translate cent values to JIPitch"
                raise NotImplementedError(msg)
            else:
                ratio = p.split("/")
                ratio_size = len(ratio)
                if ratio_size == 2:
                    num, den = tuple(int(n) for n in ratio)
                elif ratio_size == 1:
                    num = int(ratio[0])
                    den = 1
                else:
                    msg = "Can't read ratio {0}.".format(ratio)
                    raise NotImplementedError(msg)

                try:
                    assert num > den
                except AssertionError:
                    msg = "ERROR: Invalide ratio {0}. ".format(ratio)
                    msg += "Ratios have to be positiv (numerator "
                    msg += "has to be bigger than denominator)."
                    raise ValueError(msg)

                return r(num, den)

        with open(name, "r") as f:
            lines = f.read().splitlines()
            # deleting comments
            lines = tuple(line for line in lines if line and line[0] != "!")
            description = lines[0]
            pitches = lines[2:]
            estimated_amount_pitches = int(lines[1])
            real_amount_pitches = len(pitches)

            try:
                assert estimated_amount_pitches == real_amount_pitches
            except AssertionError:
                msg = "'{}' contains {} pitches ".format(
                    description, real_amount_pitches
                )
                msg += "while {} pitches are expected.".format(estimated_amount_pitches)
                raise ValueError(msg)

        return cls((r(1, 1),) + tuple(convert2pitch(p) for p in pitches))

    @property
    def val_border(self) -> int:
        i = 0
        v = None
        while v is None:
            try:
                v = self[i].val_border
            except AttributeError:
                i += 1
            except IndexError:
                v = 1
        return v

    @val_border.setter
    def val_border(self, arg: int) -> None:
        self._val_border = arg
        shift_val = Monzo.count_primes(arg)
        for f in self:
            f._val_shift = shift_val

    @classmethod
    def mk_line(cls, reference, count):
        return cls([reference.scalar(i + 1) for i in range(count)])

    @property
    def avg_gender(self):
        if self:
            return sum(map(lambda b: 1 if b is True else -1, self.gender)) / len(self)
        else:
            return 0

    @property
    def avg_lv(self):
        if self:
            return sum(p.lv for p in self) / len(self)
        else:
            return 0

    @staticmethod
    def from_json(cls, data):
        iterable = data[0]
        multiply = data[1]
        obj = cls(tuple(JIPitch.from_json(p) for p in iterable))
        obj.multiply = multiply
        return obj

    def convert2json(self):
        return json.dumps(
            (tuple((p._vec, p.val_border, p.multiply) for p in self), self.multiply)
        )

    @staticmethod
    def load_json(cls, name: str):
        with open(name, "r") as f:
            encoded = json.loads(f.read())
        return cls.from_json(encoded)

    def export2json(self, name: str):
        with open(name, "w") as f:
            f.write(self.convert2json())

    def set_multiply(self, arg: float):
        """Set the multiply - argument of every containing pitch - element."""
        for p in self:
            if p is not None:
                p.multiply = arg

    def set_muliplied_multiply(self, arg: float):
        """Multiply the multiply-property of every containing pitch with the input."""
        for p in self:
            if p is not None:
                p.multiply *= arg

    def show(self) -> tuple:
        r = tuple((r, p, round(f, 2)) for r, p, f in zip(self, self.primes, self.freq))
        return tuple(sorted(r, key=lambda t: t[2]))

    def dot_sum(self):
        """Return the sum of every dot-product between two Monzos in the Container"""
        d = 0
        for m_out in self:
            for m_in in self:
                if m_in != m_out:
                    d += m_out.dot(m_in)
        return d

    def summed_summed(self):
        summed = self.summed()
        if summed:
            return functools.reduce(lambda x, y: x + y, summed)
        else:
            return 0

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
    def dominant_prime(self) -> tuple:
        summed = abs(functools.reduce(lambda x, y: x + y, self))
        if summed:
            maxima = max(summed)
            return tuple(summed.val[i] for i, exp in enumerate(summed) if exp == maxima)
        else:
            return 1

    def remove(self, *pitch):
        def test_allowed(p, pitches):
            for p1 in pitches:
                if p1 == p:
                    return False
            return True

        data = [p for p in self if test_allowed(p, pitch)]
        copied = type(self)(data, self.multiply)
        copied.val_border = self.val_border
        return copied

    def find_by(self, pitch, compare_function) -> Type["JIPitch"]:
        """Compares every pitch of a Container with the input pitch using the compare_function.

        The compare_function shall return a float or integer number.
        This float or integer number represents the fitness of the specific
        pitch. The return-value is the pitch with the lowest fitness (Minima).
        """
        result = tuple((p, compare_function(p, pitch)) for p in self)
        if result:
            result_sorted = sorted(result, key=lambda el: el[1])
            return result_sorted[0][0]

    def map(self, function):
        return type(self)((function(x) for x in self))

    def add_map(self, pitch):
        return self.map(
            lambda x: x + pitch if x != mel.EmptyPitch() else mel.EmptyPitch()
        )

    def sub_map(self, pitch):
        return self.map(lambda x: x - pitch)

    def adjust_register_of_identities(
        self,
        *identity,
        limitup: float = 2 ** 6,
        concert_pitch_period: int = 3,
        not_listed_startperiod: int = 3
    ):
        """Adjust register of different pitches in the Container differently.

        This happens in respect to their identity. For
        every Identity there has to be an extra argument.
        By default not listed identities will
        be adjusted by startperiod: int=3.
        The Syntax of every Input - Arguments is:
        (identity (Pitch or Monzo Object), startperiod: int = 3)
        Arguments:
            * identities, as many as wished
        Keyword-Arugments:
            * limitup
            * concert_pitch_period
            * not_listed_startperiod
        >>> p0 = r(4, 3, val_border=2)
        >>> p1 = r(16, 9, val_border=2)
        >>> p2 = r(7, 4, val_border=2)
        >>> p3 = r(49, 32, val_border=2)
        >>> p4 = r(5, 4, val_border=2)
        >>> h0 = JIHarmony((p0, p1, p2, p3, p4))
        >>> identities = ((p0.identity, 1), (p1.identity, 1),
                          (p2.identity, 4))
        >>> h0.adjust_register_of_identities(*identities)
        JIHarmony([]) #TODO write proper solution
        """
        new_container = []
        for p in self:
            found = False
            for p_id, startperiod_id in identity:
                if p.identity == p_id:
                    p_new = p.adjust_register(
                        limitup=limitup,
                        concert_pitch_period=concert_pitch_period,
                        startperiod=startperiod_id,
                    )
                    found = True
                    break
            if found is False:
                p_new = p.adjust_register(
                    limitup=limitup,
                    concert_pitch_period=concert_pitch_period,
                    startperiod=not_listed_startperiod,
                )
            new_container.append(p_new)
        return type(self)(new_container)

    def diff(self: "JIContainer", other: "JIContainer") -> float:
        """Calculate the difference between two Container - Objects.

        Return a float - value.
        """
        h0 = self.copy()
        h0._val_shift = 1
        h1 = other.copy()
        h1._val_shift = 1
        diff = JIPitch([])
        length = len(h0) + len(h1)
        if length != 0:
            for p0 in h0:
                for p1 in h1:
                    diff += p0 - p1
            return diff.summed() / length
        else:
            return 0.0


class JIMel(JIPitch.mk_iterable(mel.Mel), JIContainer):
    def __init__(self, iterable, multiply=260):
        JIContainer.__init__(self, iterable, multiply)

    def copy(self):
        copied = mel.Mel.copy(self)
        copied.val_border = self.val_border
        return copied

    @classmethod
    def from_str(cls, string) -> "JIMel":
        return JIContainer.from_str(cls, string)

    @classmethod
    def from_json(cls, js):
        return JIContainer.from_json(cls, js)

    def convert2json(self):
        return JIContainer.convert2json(self)

    @classmethod
    def load_json(cls, name: str):
        return JIContainer.load_json(cls, name)

    def export2json(self, name: str):
        return JIContainer.export2json(self, name)

    def calc(self):
        return mel.Mel.calc(self)

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

    def add(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 + m1 for m0, m1 in zip(self, other)))

    def sub(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 - m1 for m0, m1 in zip(self, other)))

    def mul(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 * m1 for m0, m1 in zip(self, other)))

    def div(self, other: "JIMel") -> "JIMel":
        return JIMel((m0 / m1 for m0, m1 in zip(self, other)))

    def remove(self, *pitch):
        return JIContainer.remove(self, *pitch)

    @property
    def val_border(self) -> int:
        return JIContainer.val_border.__get__(self)

    @val_border.setter
    def val_border(self, arg) -> None:
        JIContainer.val_border.__set__(self, arg)

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
        return tuple(abs(t0.lv - t1.lv) for t0, t1 in zip(self, self[1:]))

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
        return type(self)(
            functools.reduce(lambda x, y: x + y, tuple(t.subvert() for t in self)),
            self.multiply,
        )

    def accumulate(self):
        return type(self)(tuple(itertools.accumulate(self)), self.multiply)

    def separate(self):
        subverted = JIMel((self[0],)) + self.intervals.subvert()
        return type(self)(subverted, self.multiply).accumulate()

    def find_by_walk(self, pitch, compare_function) -> Type["JIMel"]:
        """Iterative usage of the find_by - method.

        The input pitch
        is the first argument, the resulting pitch is next input Argument etc.
        until the Container might be empty.
        """
        test_container = self.copy()
        result = [pitch]
        while len(test_container):
            pitch = test_container.find_by(pitch, compare_function)
            test_container = test_container.remove(pitch)
            result.append(pitch)
        return type(self)(result)

    def find_by_walk_best(self, pitch, compare_function) -> tuple:
        """Iterative usage of the find_by - method.

        The input pitch
        is the first argument, the resulting pitch is next input Argument etc.
        until the Container might be empty. Unlike the find_by_walk - method
        the find_by_walk_best - method will always return the best result
        (e. g. with the lowest summed fitness)
        """

        def calc_fitness(ind):
            fitness = tuple(compare_function(p0, p1) for p0, p1 in zip(ind, ind[1:]))
            return sum(fitness)

        permutations = itertools.permutations(self)
        result = []
        current_min = None
        for ind in permutations:
            if pitch is not None:
                ind = (pitch,) + ind
            fitness = calc_fitness(ind)
            if current_min is not None:
                if fitness == current_min:
                    current_min = fitness
                    result.append((ind, fitness))
                elif fitness < current_min:
                    current_min = fitness
                    result = [(ind, fitness)]
            else:
                current_min = fitness
                result.append((ind, fitness))

        minima = min(result, key=lambda x: x[1])
        all_minima = (
            type(self)(f[0]) for i, f in enumerate(result) if f[1] == minima[1]
        )
        return tuple(all_minima)


class JIHarmony(JIPitch.mk_iterable(mel.Harmony), JIContainer):
    def __init__(self, iterable, multiply=260):
        return JIContainer.__init__(self, iterable, multiply)

    @classmethod
    def from_str(cls, string) -> "JIHarmony":
        return JIContainer.from_str(cls, string)

    @classmethod
    def from_json(cls, js):
        return JIContainer.from_json(cls, js)

    def convert2json(self):
        return JIContainer.convert2json(self)

    @classmethod
    def load_json(cls, name: str):
        return JIContainer.load_json(cls, name)

    def export2json(self, name: str):
        return JIContainer.export2json(self, name)

    @property
    def intervals(self):
        """Return all present intervals between single notes."""
        data = tuple(self)
        intervals = JIHarmony([])
        for i, p0 in enumerate(data):
            for p1 in data[i + 1 :]:
                interval = p1 - p0
                intervals.add(interval)
                interval = p0 - p1
                intervals.add(interval)
        return intervals

    def dot(self: "JIHarmony", other: "JIHarmony") -> int:
        """Calculates dot product between every Pitch of itself with every Pitch of the

        other JIHarmony.
        It accumulates the results.
        """

        h0 = self.copy()
        h0._val_shift = 1
        h1 = other.copy()
        h1._val_shift = 1
        acc = 0
        for p0 in h0:
            for p1 in h1:
                acc += p0.dot(p1)
        return acc

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    def remove(self, *pitch):
        return JIContainer.remove(self, *pitch)

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
        if self:
            for p in self:
                return p.val_border
        return 1

    @val_border.setter
    def val_border(self, arg) -> None:
        JIContainer.val_border.__set__(self, arg)

    def converted2root(self):
        root = self.root
        if root:
            root = self.root[0]
            return JIHarmony(t - root for t in self)
        else:
            return JIHarmony([])

    def operator_harmony(self, func):
        # TODO(replace ugly implementation by a better one)
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
        return JIHarmony(
            functools.reduce(
                lambda x, y: x + y, tuple(tone.components + (tone,) for tone in self)
            )
        )

    def inverse_harmony(self):
        return self.inverse() | self

    def symmetric_harmony(self, *shift):
        return JIHarmony(
            functools.reduce(lambda x, y: x | y, tuple(self.shift(s) for s in shift))
        )

    def past_harmony(self):
        return JIHarmony(
            functools.reduce(
                lambda x, y: x + y, tuple(tone.past + (tone,) for tone in self)
            )
        )

    @property
    def differential(self):
        harmony = type(self)([])
        for p0 in self:
            for p1 in self:
                if p0 != p1:
                    diff = p0.differential(p1)
                    harmony.add(diff)
        return harmony

    def adjust_register_by_fitness(self, function, range_harmonies=None):
        # TODO(add documentation), make implementation better understandable
        identities = self.identity
        length_identities = len(identities)
        if range_harmonies is None:
            range_harmonies = tuple(range(8))
        possible_solutions = itertools.combinations(
            range_harmonies * length_identities, length_identities
        )
        data = []
        for solution in possible_solutions:
            solution = tuple(zip(identities, solution))
            fit = function(self, solution)
            data.append((solution, fit))
        minima = min(data, key=lambda i: i[1])
        minimas = (sol for sol in data if sol[1] == minima[1])
        return tuple(
            self.adjust_register_of_identities(*minima[0]) for minima in minimas
        )


class JICadence(JIPitch.mk_iterable(mel.Cadence), JIContainer):
    def __init__(self, iterable: List[JIHarmony], multiply: int = 1) -> None:
        super(type(self), self).__init__(iterable)
        self.multiply = multiply
        self._val_border = 1

    def copy(self):
        return type(self)(tuple(self), multiply=self.multiply)

    @classmethod
    def from_json(cls, data):
        return cls([JIHarmony.from_json(h) for h in data])

    def convert2json(self):
        return json.dumps(
            tuple(
                (
                    tuple((p._vec, p.val_border, p.multiply) for p in chord),
                    chord.multiply,
                )
                for chord in self
            )
        )

    @classmethod
    def load_json(cls, name: str):
        return JIContainer.load_json(cls, name)

    def export2json(self, name: str):
        return JIContainer.export2json(self, name)

    @property
    def val_border(self) -> int:
        return JIContainer.val_border.__get__(self)

    @val_border.setter
    def val_border(self, arg) -> None:
        JIContainer.val_border.__set__(self, arg)

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

    def adjust_register(self, *args, **kwargs):
        return type(self)([h.adjust_register(*args, **kwargs) for h in self])

    @property
    def differential(self):
        return type(self)([h.differential for h in self])

    @property
    def avg_lv(self):
        return tuple(h.avg_lv for h in self)

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

    def count_pitch(self, pitch) -> int:
        """Count how often the asked pitch occurs in the Cadence."""

        c = 0
        for harmony in self:
            for p in harmony:
                if p == pitch:
                    c += 1
                    break
        return c

    def count_different_pitches(self) -> int:
        """Count how many different pitches occur in the Cadence."""

        c = 0
        already = []
        for harmony in self:
            for p in harmony:
                if p not in already:
                    c += 1
                    already.append(p)
                    break
        return c

    @property
    def chord_rate(self):
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
    def chord_rate_sorted(self):
        return tuple(sorted(self.chord_rate, key=lambda obj: obj[1]))

    @property
    def different_chords(self):
        return tuple(zip(*self.chord_rate))[0]

    def adjust_register_of_identities(
        self,
        *identity,
        limitup: float = 2 ** 6,
        concert_pitch_period: int = 3,
        not_listed_startperiod: int = 3
    ):
        return type(self)(
            h.adjust_register_of_identities(
                *identity,
                limitup=limitup,
                concert_pitch_period=concert_pitch_period,
                not_listed_startperiod=not_listed_startperiod
            )
            for h in self
        )


class JIScale(JIPitch.mk_iterable(mel.Scale), JIContainer):
    _period_cls = JIMel

    def __init__(self, period, periodsize=JIPitch([1])):
        period = JIMel(sorted(period))
        for i, p in enumerate(period):
            while p > periodsize:
                p -= periodsize
            period[i] = p
        mel.Scale.__init__(self, period, periodsize)

    @property
    def intervals(self):
        return JIHarmony(self.period).intervals

    def map(self, function):
        return type(self)((function(x) for x in self.period), function(self.periodsize))

    def index(self, item):
        for c, x in enumerate(self):
            if x == item:
                return c
        raise ValueError("x not in tuple")

    @property
    def val_border(self) -> int:
        return JIContainer.val_border.__get__(self)

    @val_border.setter
    def val_border(self, arg) -> None:
        JIContainer.val_border.__set__(self, arg)


class GeneratorScale(JIScale):
    def __init__(self, *generator, n=5, period=JIPitch([])):
        generator = itertools.cycle(generator)
        s = (next(generator) for i in range(n - 1))
        s = JIMel(s)
        s.insert(0, r(1, 1))
        s = s.accumulate()
        JIScale.__init__(self, s, period)


class JIStencil(object):
    r"""This class implements a non-general way to handle complex JI harmony.

    To initialize a JIStencil - object tuples containing a
    Monzo or JIPitch - objects have to be passed:
    >>> mystencil = JIStencil(
            (JIPitch((1,), val_border=2), 0, 2),
            (JIPitch((0, 1), val_border=2), 1, 3))

    The second and the third number in the tuples are
    specifying the exponents of the passed pitch, e.g.
    (JIPitch((1,), 2), 0, 2) would result in a harmony
    containing (JIPitch((0,), val_border=2), JIPitch((1,), val_border=2)).

    If the first number equals 0 it could be skipped, meaning
    that (JIPitch((1,), 2), 0, 2) == (JIPitch((1,), 2), 2).
    """

    def __init__(self, *args):
        def add_zero(sub):
            if len(sub) == 2:
                return (sub[0], 0, sub[1])
            else:
                return sub

        self._vector = tuple(add_zero(arg) for arg in args)
        self._pitches = tuple(arg[0] for arg in args)

    def __len__(self):
        return len(self._vector)

    def __iter__(self):
        return iter(self._vector)

    @staticmethod
    def convertvec2harmony(vec):
        return tuple(vec[0].scalar(i) for i in range(vec[1], vec[2]))

    @property
    def primes(self):
        return tuple(
            set(functools.reduce(operator.add, (p.primes for p in self._pitches)))
        )

    @property
    def identity(self):
        return tuple(set(p.identity for p in self._pitches))

    def convert2harmony(self):
        """Return a converted version of itself (JIHarmony - object).

        >>> mystencil = JIStencil(
                (JIPitch((1,), 2), 0, 2), (JIPitch((0, 1), 2), 1, 3))
        >>> myharmony = mystencil.convert2harmony()
        JIHarmony({1, 25/16, 3/2, 5/4})
        """

        return JIHarmony(
            functools.reduce(
                operator.add, (self.convertvec2harmony(v) for v in self._vector)
            )
        )


class BlueprintPitch(object):
    r"""BlueprintPitch helps to abstract just intonation pitches.

    To initalise a new object two tuples or lists are necessary,
    one for the numerator and one for the denominator of the abstract
    pitch. Those tuples are filled with integer. Each integer expresses
    how many prime numbers of the nth power are occurring in the numerator
    or denominator, where n is the list index of the integer plus one.

    For instace a list [n, m, o] as an argument for the objects
    numerator represents an abstract pitch whose numerator is the product of
        1. n - prime numbers with the power of 1
        2. m - prime numbers with the power of 2
        3. o - prime numbers with the power of 3

    In a concrete case with n = 2, m = 1 and o = 1 and the prime numbers
    3, 5, 7, 11 this would result in:

        (3 ** 1) * (5 ** 1) * (7 ** 2) * (11 ** 3)


    >>> bp0 = BlueprintPitch((2,), (1,))
    >>> bp0(3, 5, 7)  # (3**1) * (5**1) / (7**1)
    15/7
    >>> bp0(7, 3, 5)  # (7**1) * (3**1) / (5**1)
    21/5
    >>> bp1 = BlueprintPitch((0, 1), (1,))
    >>> bp1(3, 5)  # 3**2 / 5**1
    9/5
    >>> bp1(5, 3)  # 5**2 / 3**1
    25/3
    """

    int2power = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }

    def __init__(
        self, numerator: tuple = tuple([]), denominator: tuple = tuple([])
    ) -> None:
        for n in (numerator, denominator):
            if n:
                assert sum(n) > 0
            for item in n:
                assert item >= 0

        self.__numerator, self.__denominator = tuple(numerator), tuple(denominator)
        self.__size = sum(sum(n) for n in (self.numerator, self.denominator))

    @classmethod
    def from_pitch(cls, pitch: JIPitch) -> "BlueprintPitch":
        return cls(*pitch.blueprint)

    def __repr__(self) -> str:
        item_counter = 0
        representations = []
        for num in (self.numerator, self.denominator):

            representation = ""

            for exponent, amount in enumerate(num):
                exponent = "".join(
                    self.int2power[integer] for integer in str(exponent + 1)
                )
                for counter in range(amount):
                    idx = counter + item_counter
                    if idx < 27:
                        idx = string.ascii_uppercase[idx]
                    representation += "{0}{1}".format(idx, exponent)
                item_counter += amount

            if not representation:
                representation = "1"

            representations.append(representation)

        return "({0} | {1})".format(*representations)

    def __hash__(self) -> int:
        return hash(self.blueprint)

    @property
    def blueprint(self) -> tuple:
        return (self.numerator, self.denominator)

    @property
    def numerator(self) -> tuple:
        return self.__numerator

    @property
    def denominator(self) -> tuple:
        return self.__denominator

    @property
    def size(self) -> int:
        return self.__size

    def inverse(self) -> "BlueprintPitch":
        return type(self)(self.denominator, self.numerator)

    def is_instance(self, pitch: JIPitch, ignore: tuple = (2,)) -> bool:

        try:
            return self.blueprint == pitch.blueprint
        except AttributeError:
            return False

    def __eq__(self, other) -> bool:
        try:
            return self.blueprint == other.blueprint
        except AttributeError:
            return False

    def __call__(self, *args: int) -> JIPitch:
        for arg in args:
            try:
                assert prime_factors.is_prime(arg)
            except AssertionError:
                msg = "Every argument has to be a prime number."
                msg += " {0} is not a prime number.".format(arg)
                raise ValueError(msg)

        try:
            assert len(args) == self.size
        except AssertionError:
            msg = "This BlueprintPitch needs {0} prime-numbers.".format(self.size)
            msg += " But only {0} numbers were given.".format(len(args))
            raise ValueError(msg)

        index = 0
        num_and_den = [[1], [1]]
        for item_idx, item in enumerate(self.blueprint):
            for exponent, amount in enumerate(item):
                exponent += 1
                for n in range(amount):
                    num_and_den[item_idx].append(args[index] ** exponent)
                    index += 1

        return r(*tuple(functools.reduce(operator.mul, n) for n in num_and_den))


class BlueprintHarmony(object):
    """BlueprintHarmony helps to abstract just intonation harmonies.

    The init-form is:
        (BlueprintPitch, indices),
        (BlueprintPitch, indices), ...

    Example:
        BlueprintHarmony((BlueprintPitch([1], [1]), (0, 1)),
                         (BlueprintPitch([2], []), (0,)))
    """

    def __init__(self, *blueprint: tuple) -> None:
        blueprint = tuple(
            (p, self.sort_blueprint_pitch_indices(p, indices))
            for p, indices in blueprint
        )

        ig1 = operator.itemgetter(1)
        numbers = tuple(ig1(bp) for bp in blueprint)
        available_numbers = tuple(set(functools.reduce(operator.add, numbers)))

        # initalise the attributes
        self.__size = len(available_numbers)
        self.__blueprint = blueprint
        self.__sorted_available_indices = tuple(sorted(available_numbers))

    @classmethod
    def from_harmony(cls, harmony: tuple) -> "BlueprintHarmony":
        def identify_blueprint_for_pitch(pitch: JIPitch, known_primes: tuple) -> tuple:
            blueprint = pitch.blueprint
            attributes = [[], []]
            for idx, num in enumerate(blueprint):
                for exponent, amount in enumerate(num):
                    exponent += 1

                    if idx == 1:  # for denominator
                        exponent = -exponent

                    for monzo_idx, monzo_value in enumerate(pitch.monzo):
                        if monzo_value is exponent:
                            attributes[idx].append(
                                known_primes.index(pitch.val[monzo_idx])
                            )

            return (
                BlueprintPitch(*blueprint),
                tuple(functools.reduce(operator.add, attributes)),
            )

        primes_per_pitch = tuple(p.primes for p in harmony)
        primes = tuple(sorted(set(functools.reduce(operator.add, primes_per_pitch))))

        blueprint = tuple(
            identify_blueprint_for_pitch(pitch, primes) for pitch in harmony
        )

        return cls(*blueprint)

    @staticmethod
    def get_id_vector_of_blueprint_pitch(blueprint_pitch: BlueprintPitch) -> tuple:
        bp = tuple(
            base
            for base in functools.reduce(operator.add, blueprint_pitch.blueprint)
            if base > 0
        )
        res = []
        for exponent_idx, n_bases in enumerate(bp):
            for to_add in range(n_bases):
                res.append(exponent_idx)
        return tuple(res)

    @staticmethod
    def sort_blueprint_pitch_indices(
        blueprint_pitch: BlueprintPitch, indices: tuple
    ) -> tuple:
        id_vector = BlueprintHarmony.get_id_vector_of_blueprint_pitch(blueprint_pitch)
        divisions = [[] for i in range(max(id_vector) + 1)]
        for idx, ID in zip(indices, id_vector):
            divisions[ID].append(idx)
        return tuple(
            functools.reduce(operator.add, tuple(sorted(div) for div in divisions))
        )

    @property
    def identity(self) -> set:
        items_per_idx = [[] for i in range(self.size)]
        for bp in self.blueprint:
            hash_blueprint_pitch = hash(bp[0].blueprint)
            id_vector = self.get_id_vector_of_blueprint_pitch(bp[0])
            for ID, item in zip(id_vector, bp[1]):
                real_index = self.__sorted_available_indices.index(item)
                items_per_idx[real_index].append((hash_blueprint_pitch, ID))
        return set(tuple(item) for item in items_per_idx)

    @property
    def blueprint(self) -> tuple:
        return self.__blueprint

    @property
    def size(self) -> int:
        return self.__size

    @property
    def n_pitch_repetitions(self) -> int:
        return len(self.blueprint) - len(set(self.blueprint))

    def inverse(self) -> "BlueprintHarmony":
        return type(self)(
            *tuple((bp.inverse(), indices) for bp, indices in self.blueprint)
        )

    def __hash__(self) -> int:
        return hash(self.blueprint)

    def __len__(self) -> int:
        return len(self.blueprint)

    def __repr__(self) -> str:
        return str(self.blueprint)

    def __eq__(self, other) -> bool:
        """Compare two BlueprintHarmony objects.

        Return True if both objects generate the same ouput when they get
        called with the same attributes in the same order.
        """
        try:
            return self.identity == other.identity
        except AttributeError:
            return False

    def n_common_pitches(self, other: "BlueprintHarmony") -> int:
        return len(set(self.blueprint).intersection(set(other.blueprint)))

    def is_instance(
        self, harmony: JIHarmony, gender: bool = True, ignore: tuple = (2,)
    ) -> bool:
        return self == type(self).from_harmony(harmony)

    def __call__(self, *args) -> tuple:
        return tuple(
            p(*tuple(args[idx] for idx in indices)) for p, indices in self.blueprint
        )


def find_all_available_pitches_in_a_specified_range(
    pitch: JIPitch, minima: JIPitch, maxima: JIPitch
) -> tuple:
    """Return pitches in all possible register between minima and maxima."""

    p_norm = pitch.normalize()
    min_norm = minima.normalize()
    max_norm = maxima.normalize()
    oct0 = minima.octave
    oct1 = maxima.octave + 1

    if p_norm.float < min_norm.float:
        oct0 += 1
    if p_norm.float > max_norm.float:
        oct1 -= 1

    assert oct0 != oct1

    return tuple(p_norm.register(o) for o in range(oct0, oct1))


def find_best_voice_leading(pitches: tuple, tonal_range: tuple) -> tuple:
    """Brute force searching for the best voice leading.

    The best voice leading is defined as the voice leading
    with the smallest sum of cent differences between two
    succeeding pitches.
    """

    def brute_force_algorithm(pitches: tuple) -> tuple:
        element = None
        fitness = None
        for solution in itertools.product(
            *tuple(
                find_all_available_pitches_in_a_specified_range(p, minima, maxima)
                for p in pitches
            )
        ):
            differences = sum(
                abs((p0 - p1).cents) for p0, p1 in zip(solution, solution[1:])
            )

            is_addable = True

            if element:
                is_addable = differences < fitness

            if is_addable:
                element = solution
                fitness = differences

        return element

    assert len(tonal_range) == 2
    minima, maxima = tonal_range

    splited_by_rests = tools.split_iterable_by_n(pitches, mel.TheEmptyPitch)

    result = tuple([])
    for part in splited_by_rests:
        if part[-1] == mel.TheEmptyPitch:
            is_last_rest = True
            part = part[:-1]
        else:
            is_last_rest = False

        result += brute_force_algorithm(part)

        if is_last_rest:
            result += (mel.TheEmptyPitch,)

    return tuple(result)


"""
    syntactic sugar for the creation of JIPitch - Objects:
"""


def r(num: int, den: int, val_border: int = 1, multiply: float = None) -> JIPitch:
    return JIPitch.from_ratio(num, den, val_border, multiply)


def m(*num: int, val_border: int = 1, multiply: float = None) -> JIPitch:
    return JIPitch.from_monzo(*num, val_border=val_border, multiply=multiply)
