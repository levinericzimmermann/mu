from mu.abstract import muobjects
from mu.mel import abstract

from typing import Any
import collections

try:
    import quicktions as fractions
except ImportError:
    import fractions


class SimplePitch(abstract.AbstractPitch):
    """A very simple pitch / interval implementation.

    SimplePitch - objects are specified via a concert_pitch frequency
    and a cent value, that describe the distance to the concert_pitch.
    """

    def __init__(self, concert_pitch_freq: float, cents: float = 0):
        self.__concert_pitch_freq = concert_pitch_freq
        self.__cents = cents

    def __repr__(self) -> str:
        return "<{0}ct|{1}Hz>".format(self.cents, self.concert_pitch_freq)

    def calc(self) -> float:
        return self.concert_pitch_freq * (2 ** (self.cents / 1200))

    @classmethod
    def from_scl(cls, scl_line: str, concert_pitch_freq: float) -> "SimplePitch":
        p = scl_line.split(" ")[0]
        if p[-1] == ".":
            cents = float(p[:-1])
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

            cents = abstract.AbstractPitch.ratio2ct(fractions.Fraction(num, den))

        return cls(concert_pitch_freq, cents)

    @property
    def cents(self) -> float:
        return self.__cents

    @property
    def concert_pitch_freq(self) -> float:
        return self.__concert_pitch_freq

    def copy(self) -> "SimplePitch":
        return type(self)(self.concert_pitch_freq, self.cents)

    def __add__(self, other) -> "SimplePitch":
        return type(self)(self.concert_pitch_freq, self.cents + other.cents)


class EmptyPitch(abstract.AbstractPitch):
    def calc(self, factor=0):
        return None

    def __repr__(self):
        return "NoPitch"

    def copy(self):
        return type(self)()

    @property
    def cents(self) -> None:
        return None


TheEmptyPitch = EmptyPitch()


class Mel(muobjects.MUList):
    def __init__(self, iterable: Any, multiply: int = 1) -> None:
        muobjects.MUList.__init__(self, iterable)
        self.multiply = 1

    @classmethod
    def from_scl(cls, name: str, concert_pitch: float) -> "JIContainer":
        """Generating JIContainer from the scl file format.

        See: http://huygens-fokker.org/scala/scl_format.html
        """

        with open(name, "r") as f:
            lines = f.read().splitlines()
            # deleting comments
            lines = tuple(l for l in lines if l and l[0] != "!")
            description = lines[0]
            pitches = lines[2:]
            estimated_amount_pitches = int(lines[1])
            real_amount_pitches = len(pitches)

            try:
                assert estimated_amount_pitches == real_amount_pitches
            except AssertionError:
                msg = "'{0}' contains {1} pitches ".format(
                    description, real_amount_pitches
                )
                msg += "while {2} pitches are expected.".format(
                    estimated_amount_pitches
                )
                raise ValueError(msg)

        pitches = tuple(SimplePitch.from_scl(p, concert_pitch) for p in pitches)
        return cls((SimplePitch(concert_pitch, 0),) + pitches)

    def copy(self):
        iterable = tuple(item.copy() for item in self)
        return type(self)(iterable, multiply=self.multiply)

    def __hash__(self) -> int:
        return hash(tuple(hash(t) for t in self))

    def calc(self, factor: int = 1) -> tuple:
        f = self.multiply * factor
        return tuple(p.calc() * f for p in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def cents(self) -> tuple:
        return tuple(item.cents for item in self)

    def uniqify(self):
        unique = collections.OrderedDict((x, True) for x in self).keys()
        return type(self)(unique)


class Harmony(muobjects.MUSet):
    def __hash__(self) -> int:
        return hash(tuple(hash(t) for t in self))

    def sorted(self):
        return sorted(self.calc())

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc() * factor for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()

    @property
    def cents(self) -> tuple:
        return tuple(item.cents for item in self)


class Cadence(muobjects.MUList):
    def __hash__(self):
        return hash(tuple(hash(h) for h in self))

    def calc(self, factor=1) -> tuple:
        return tuple(h.calc(factor) for h in self)

    @property
    def freq(self) -> tuple:
        return self.calc()


class Scale(muobjects.MUOrderedSet):
    _period_cls = Mel

    def __init__(self, period, periodsize):
        if not type(period) == self._period_cls:
            period = self._period_cls(period)
        period = period.sort().uniqify()
        muobjects.MUOrderedSet.__init__(self, period + self._period_cls((periodsize,)))
        self.period = period
        self.periodsize = periodsize

    def __add__(self, other):
        return type(self)(
            tuple(self.period) + tuple(other.period),
            max((self.periodsize, other.periodsize)),
        )
