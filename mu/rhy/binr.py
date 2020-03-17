import functools
import operator

try:
    import quicktions as fractions
except ImportError:
    import fractions

from mu.rhy import rhy
from mu.utils import tools

# TODO(Documentation) add module description and function doc
# TODO(methods) complete NotImplemented Methods


class Compound(rhy.AbstractRhythm):
    __valid_representations = ("relative", "absolute")

    def __init__(self, iterable: list) -> None:
        iterable = tuple(iterable)

        self.__representation = "relative"

        if iterable:
            if iterable[0] == 0:
                self.__representation = "absolute"

        if self.representation == "absolute":
            iterable = tuple(b - a for a, b in zip(iterable, iterable[1:]))

        essence, multiply = self.convert_rhythm2essence_and_multiply(iterable)
        self.__essence = essence
        self.__multiply = multiply

    def copy(self) -> "Compound":
        return type(self).from_int(int(self.essence), fractions.Fraction(self.multiply))

    # alternative init methods:
    @classmethod
    def from_binary(
        cls, binary_number: bin, multiply: fractions.Fraction = fractions.Fraction(1, 1)
    ) -> "Compound":
        new = cls(Compound.convert_binary2rhythm(binary_number))
        new.multiply = multiply
        return new

    @classmethod
    def from_int(
        cls, essence: int, multiply: fractions.Fraction = fractions.Fraction(1, 1)
    ):
        assert Compound.is_valid_essence(essence)
        assert Compound.is_valid_multiply(multiply)
        return cls(Compound.convert_essence_and_multiply2rhythm(essence, multiply))

    @classmethod
    def from_barlow(cls, primes: tuple, density: float) -> "Compound":
        raise NotImplementedError

    @classmethod
    def from_binary_rhythm(
        cls,
        binary_rhythm: tuple,
        multiply: fractions.Fraction = fractions.Fraction(1, 1),
    ) -> "Compound":
        new = cls(Compound.convert_binary_rhythm2rhythm(binary_rhythm))
        new.multiply = multiply
        return new

    @classmethod
    def from_euclid(
        cls, n: int, m: int, multiply: fractions.Fraction = fractions.Fraction(1, 1)
    ) -> "Compound":
        new = cls(tools.euclid(n, m))
        new.multiply = multiply
        return new

    @classmethod
    def from_generator(cls, period: int, size: int) -> "Compound":
        """Periodic repetitive rhythm maker."""
        assert period <= size
        generator = tuple(range(0, size, period))
        return cls([b - a for a, b in zip(generator, generator[1:])])

    @classmethod
    def from_synchronization(
        cls, *fac: int, multiply: fractions.Fraction = fractions.Fraction(1, 1)
    ) -> "Compound":
        """Schillinger-like synchronization algorithm."""
        size = functools.reduce(operator.mul, fac)
        return functools.reduce(
            lambda a, b: a.union(b),
            tuple(cls.from_generator(factor, size) for factor in fac),
        )
        raise NotImplementedError

    @classmethod
    def from_sieve(cls, *fac: int) -> "Compound":
        raise NotImplementedError

    @property
    def binr(self) -> tuple:
        return Compound.convert_int2binary_rhythm(self.essence)

    @property
    def intr(self) -> tuple:
        return Compound.convert_int2rhythm(self.essence)

    @property
    def beats(self) -> int:
        return len(self.binr)

    def __converted(self) -> list:
        rhythm = Compound.convert_essence_and_multiply2rhythm(
            self.essence, self.multiply
        )
        if self.representation is "relative":
            return rhythm
        else:
            return tools.accumulate_from_zero(rhythm)[:-1]

    def convert2absolute(self) -> "Compound":
        n = self.copy()
        n.representation = "absolute"
        return n

    def convert2relative(self) -> "Compound":
        n = self.copy()
        n.representation = "relative"
        return n

    def __repr__(self) -> str:
        return str(list(float(n) for n in self.__converted()))

    def __getitem__(self, idx: int) -> float:
        return self.__converted()[idx]

    def __setitem__(self, idx: int, item: float) -> None:
        essence, multiply = Compound.convert_rhythm2essence_and_multiply(
            tuple(
                original if i != idx else item
                for i, original in enumerate(self.__converted())
            )
        )
        self.essence = essence
        self.multiply = multiply

    def __iter__(self) -> iter:
        return iter(self.__converted())

    def __add__(self, other: "Compound") -> "Compound":
        return type(self)(list(self) + list(other))

    def __reversed__(self) -> "Compound":
        return type(self)(list(reversed(self)))

    def __len__(self) -> int:
        return len(self.__converted())

    def __sum__(self) -> float:
        return sum(self.__converted())

    def __eq__(self, other) -> bool:
        try:
            return self.essence == other.essence and self.multiply == other.multiply
        except AttributeError:
            return list(self) == other

    @property
    def delay(self) -> float:
        return sum(self)

    def flat(self) -> "Compound":
        return self.copy()

    def append(self, value: float) -> None:
        essence, multiply = Compound.convert_rhythm2essence_and_multiply(
            list(self) + [value]
        )
        self.essence = essence
        self.multiply = multiply

    def insert(self, position: int, value: float) -> None:
        new = list(self)
        new.insert(position, value)
        essence, multiply = Compound.convert_rhythm2essence_and_multiply(new)
        self.essence = essence
        self.multiply = multiply

    def stretch(self, factor: float) -> "Compound":
        new = self.copy()
        new.multiply *= factor
        return new

    def real_stretch(self, factor: int) -> "Compound":
        """Stretch the rhythm with factor.

        Unlike the 'stretch' method this method doesn't change
        the multiply attribute of the object, but its essence
        attribute.
        """
        int_rhythm = tuple(int(r * factor) for r in self.intr)
        essence = self.convert_int_rhythm2essence(int_rhythm)
        return type(self).from_int(essence, self.multiply)

    @staticmethod
    def convert_binary2rhythm(binary: bin) -> tuple:
        return Compound.convert_binary_rhythm2rhythm(
            Compound.convert_binary2binary_rhythm(binary)
        )

    @staticmethod
    def convert_binary2binary_rhythm(binary: bin) -> tuple:
        return tuple(int(item) for item in tuple(binary)[2:])

    @staticmethod
    def convert_essence_and_multiply2rhythm(
        essence: int, multiply: fractions.Fraction
    ) -> list:
        return list(n * multiply for n in Compound.convert_int2rhythm(essence))

    @staticmethod
    def convert_binary_rhythm2binary(binary_rhythm: tuple) -> bin:
        return bin(int("".join(str(n) for n in binary_rhythm), 2))

    @staticmethod
    def convert_int_rhythm2binary_rhythm(int_rhythm: tuple) -> tuple:
        size = sum(int_rhythm)
        indices = tools.accumulate_from_zero(int_rhythm)
        return tuple(1 if idx in indices else 0 for idx in range(size))

    @staticmethod
    def convert_int_rhythm2essence(int_rhythm: tuple) -> tuple:
        return int(Compound.convert_binary_rhythm2binary(
            Compound.convert_int_rhythm2binary_rhythm(int_rhythm)),
            2)

    @staticmethod
    def convert_int2binary_rhythm(integer: int) -> tuple:
        return Compound.convert_binary2binary_rhythm(bin(integer))

    @staticmethod
    def convert_binary_rhythm2rhythm(binary_rhythm: tuple) -> tuple:
        indices = tuple(idx for idx, i in enumerate(binary_rhythm) if i)
        return tuple(
            b - a for a, b in zip(indices, indices[1:] + (len(binary_rhythm),))
        )

    @staticmethod
    def convert_int2rhythm(integer: int) -> tuple:
        return Compound.convert_binary_rhythm2rhythm(
            Compound.convert_int2binary_rhythm(integer)
        )

    @staticmethod
    def convert_rhythm2essence_and_multiply(rhythm: tuple) -> tuple:
        if rhythm:
            rhythm_as_fraction = tuple(fractions.Fraction(r) for r in rhythm)
            lcd = tools.lcm(*tuple(r.denominator for r in rhythm_as_fraction))
            int_rhythm = tuple(int(r * lcd) for r in rhythm_as_fraction)
            essence = Compound.convert_int_rhythm2essence(int_rhythm)
            return essence, fractions.Fraction(1, lcd)
        else:
            return 0, 1

    @staticmethod
    def is_valid_essence(essence: int) -> bool:
        try:
            return essence >= 0 and (int(essence) - essence) is 0
        except TypeError:
            return False

    @staticmethod
    def is_valid_multiply(multiply: float) -> bool:
        try:
            return multiply >= 0 and type(multiply) in (float, int, fractions.Fraction)
        except TypeError:
            return False

    @classmethod
    def is_valid_representation(cls, representation: str) -> None:
        if representation not in cls.__valid_representations:
            msg = "Only allowed representations are {0}".format(
                cls.__valid_representations
            )
            raise ValueError(msg)

    @property
    def essence(self) -> int:
        return self.__essence

    @essence.setter
    def essence(self, value: int) -> int:
        assert Compound.is_valid_essence(value)
        self.__essence = value

    @property
    def multiply(self) -> float:
        return self.__multiply

    @multiply.setter
    def multiply(self, value: float) -> None:
        assert Compound.is_valid_multiply(value)
        self.__multiply = fractions.Fraction(value)

    @property
    def representation(self) -> str:
        return self.__representation

    @representation.setter
    def representation(self, value: str) -> None:
        type(self).is_valid_representation(value)
        self.__representation = value

    @property
    def harmonicity_barlow(self) -> float:
        # TODO(indispensabiliy) implement with default time signature, alternative
        # time signature can be choosen
        raise NotImplementedError

    def union(self, other: "Compound") -> "Compound":
        binr0, binr1 = self.binr, other.binr
        assert len(binr0) == len(binr1)
        return type(self).from_binary_rhythm(
            tuple(1 if 1 in (b0, b1) else 0 for b0, b1 in zip(binr0, binr1)),
            multiply=self.multiply,
        )

    def intersection(self, other: "Compound") -> "Compound":
        binr0, binr1 = self.binr, other.binr
        assert len(binr0) == len(binr1)
        return type(self).from_binary_rhythm(
            tuple(
                1 if 1 in (b0, b1) and b0 == b1 else 0 for b0, b1 in zip(binr0, binr1)
            ),
            multiply=self.multiply,
        )

    def modulate(self, other: "Compound") -> "Compound":
        intr_other = other.intr
        converted_self = list(self)
        assert sum(intr_other) == len(converted_self)
        indices = tools.accumulate_from_zero(intr_other)
        return type(self)(
            sum(converted_self[x:y]) for x, y in zip(indices, indices[1:])
        )

    def difference(self, other: "Compound") -> int:
        binr0, binr1 = self.binr, other.binr
        length_diff = abs(len(binr0) - len(binr1))
        content_diff = sum(1 if a != b else 0 for a, b in zip(binr0, binr1))
        return length_diff + content_diff

    def move_int(self, n: int) -> "Compound":
        if n < 0:
            try:
                assert n <= self.essence
            except AssertionError:
                msg = "Move value n={0} is too low for essence {1}.".format(
                    n, self.essence
                )
                raise ValueError(msg)

        new = self.copy()
        new.essence += n
        return new

    def move_gc(self, n: int) -> "Compound":
        binr = list(self.binr[1:])
        gc = tools.graycode(len(binr), 2)
        idx = (gc.index(binr) + n) % len(gc)
        return type(self).from_binary_rhythm([1] + gc[idx], self.multiply)
