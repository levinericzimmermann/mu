import math

try:
    import quicktions as fractions
except ImportError:
    import fractions


class Time(float):
    def __init__(self, value: float):
        try:
            assert value >= 0
        except AssertionError:
            msg = "There is no negative time! {0}".format(value)
            raise ValueError(msg)

        self.__value = self.__get_value(value)

    @staticmethod
    def __get_value(obj):
        try:
            return obj._Time__value
        except AttributeError:
            return obj

    def __repr__(self) -> str:
        return "Time({})".format(str(self))

    def __str__(self) -> str:
        return str(self.__value)

    def copy(self) -> "Time":
        return type(self)(self.__value)

    def __radd__(self, other) -> "Time":
        return type(self)(self.__value + self.__get_value(other))

    def __rsub__(self, other) -> "Time":
        return type(self)(self.__get_value(other) - self.__value)

    def __rmul__(self, other) -> "Time":
        return type(self)(self.__value * self.__get_value(other))

    def __rtruediv__(self, other) -> "Time":
        return type(self)(self.__get_value(other) / self.__value)

    def __rfloordiv__(self, other) -> "Time":
        return type(self)(self.__get_value(other) // self.__value)

    def __rmod__(self, other) -> "Time":
        return type(self)(self.__get_value(other) % self.__value)

    def __rpow__(self, other) -> "Time":
        return type(self)(self.__get_value(other) ** self.__value)

    def __add__(self, other) -> "Time":
        return type(self)(self.__value + self.__get_value(other))

    def __sub__(self, other) -> "Time":
        return type(self)(self.__value - self.__get_value(other))

    def __mul__(self, other) -> "Time":
        return type(self)(self.__value * self.__get_value(other))

    def __truediv__(self, other) -> "Time":
        return type(self)(self.__value / self.__get_value(other))

    def __floordiv__(self, other) -> "Time":
        return type(self)(self.__value // self.__get_value(other))

    def __mod__(self, other) -> "Time":
        return type(self)(self.__value % self.__get_value(other))

    def __pow__(self, other) -> "Time":
        return type(self)(self.__value ** self.__get_value(other))

    def __bool__(self) -> bool:
        return bool(self.__value)

    def __eq__(self, other) -> bool:
        return self.__value == self.__get_value(other)

    def __ne__(self, other) -> bool:
        return self.__value != self.__get_value(other)

    def __lt__(self, other) -> bool:
        return self.__value < self.__get_value(other)

    def __gt__(self, other) -> bool:
        return self.__value > self.__get_value(other)

    def __le__(self, other) -> bool:
        return self.__value <= self.__get_value(other)

    def __ge__(self, other) -> bool:
        return self.__value >= self.__get_value(other)

    def __pos__(self) -> "Time":
        return type(self)(+self.__value)

    def __neg__(self) -> "Time":
        return type(self)(-self.__value)

    def __abs__(self) -> "Time":
        return type(self)(abs(self.__value))

    def __round__(self) -> "Time":
        return type(self)(round(self.__value))

    def __floor__(self) -> "Time":
        return type(self)(math.floor(self.__value))

    def __ceil__(self) -> "Time":
        return type(self)(math.ceil(self.__value))

    def __trunc__(self) -> "Time":
        return type(self)(math.trunc(self.__value))

    def __int__(self) -> int:
        return int(self.__value)

    def __float__(self) -> float:
        return float(self.__value)

    def __hash__(self) -> int:
        return hash(self.__value)

    @property
    def fraction(self) -> fractions.Fraction:
        return fractions.Fraction(self.__value)

    @property
    def numerator(self) -> int:
        return self.fraction.numerator

    @property
    def denominator(self) -> int:
        return self.fraction.denominator

    @staticmethod
    def seconds2miliseconds(s: float) -> float:
        return s * 1000

    @staticmethod
    def minutes2miliseconds(m: float) -> float:
        return m * 60 * 1000

    @staticmethod
    def hours2miliseconds(h: float) -> float:
        return h * 60 * 60 * 1000

    @classmethod
    def from_seconds(cls, s: float) -> "Time":
        return cls(Time.seconds2miliseconds(s))

    @classmethod
    def from_minutes(cls, m: float) -> "Time":
        return cls(Time.minutes2miliseconds(m))

    @classmethod
    def from_hours(cls, h: float) -> "Time":
        return cls(Time.hours2miliseconds(h))
