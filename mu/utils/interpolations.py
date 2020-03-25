import abc
import bisect
import functools
import itertools
import numpy as np
import operator

from mu.abstract import muobjects
from mu.rhy import rhy


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

    def __hash__(self) -> int:
        return hash("LinearInterpolation")


class Logarithmic(Interpolation):
    def __call__(self, x0, x1, n, dtype=float) -> tuple:
        if x0 == 0:
            x0_pre = 0.00001
            return (0,) + tuple(np.geomspace(x0_pre, x1, n, dtype=dtype))[1:]
        else:
            return tuple(np.geomspace(x0, x1, n, dtype=dtype))

    def __hash__(self) -> int:
        return hash("LogarithmicInterpolation")


class Proportional(Interpolation):
    """The slope of this interpolation changes depending on the proportion.

    For a proportion > 1 the slope becomes sharper at the end of the interpolation,
    while a proportion < 1 means that the slope is bigger at the beginning.
    Proportion = 1 leads to linear interpolation.
    """

    def __init__(self, proportion: float):
        self.__interpolate = self.mk_interpolation_function(proportion)
        self.__proportion = proportion

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

    def __hash__(self) -> int:
        return hash((hash("ProportionalInterpolation"), hash(self.__proportion)))


class InterpolationEvent(object):
    def __init__(self, delay: rhy.Unit, interpolation_type=Linear()):
        if isinstance(delay, rhy.Unit) is False:
            delay = rhy.Unit(delay)
        self.__delay = delay
        self.__interpolation_type = interpolation_type

    def __repr__(self) -> str:
        return "InterpolationEvent({})".format(self.delay)

    @property
    def delay(self):
        return self.__delay

    @property
    def interpolation_type(self):
        return self.__interpolation_type

    @abc.abstractproperty
    def interpolate(self, other, steps: int) -> tuple:
        raise NotImplementedError


class FloatInterpolationEvent(InterpolationEvent):
    def __init__(
        self, delay: rhy.Unit, value, interpolation_type: Interpolation = Linear()
    ):
        super().__init__(delay, interpolation_type)
        self.__value = value

    def __repr__(self) -> str:
        return "FloatInterpolationEvent({}, {})".format(self.delay, self.value)

    @property
    def value(self) -> float:
        return self.__value

    def interpolate(self, other: "FloatInterpolationEvent", steps: int) -> tuple:
        return self.interpolation_type(self.value, other.value, steps)


class InterpolationLine(muobjects.MUList):
    """Container class to describe interpolations between states.

    They are expected to contain InterpolationEvent - objects.
    InterpolationLine - objects can be called to generate the interpolation.
    Input arguments are only Gridsize that describe how small
    one interpolation step is. InterpolationLine - objects are
    unchangeable objects.

    The last event is expected to have delay == 0.
    """

    def __init__(self, iterable):
        iterable = tuple(iterable)

        try:
            assert iterable[-1].delay == 0
        except AssertionError:
            raise ValueError("The last element has to have delay = 0")

        muobjects.MUList.__init__(self, iterable)

    def __call__(self, gridsize: float):
        def find_closest_point(points, time):
            pos = bisect.bisect_right(points, time)
            try:
                return min(
                    (
                        (abs(time - points[pos]), pos),
                        (abs(time - points[pos - 1]), pos - 1),
                    ),
                    key=operator.itemgetter(0),
                )[1]
            except IndexError:
                # if pos is len(points) + 1
                return pos

        points = Linear()(0, float(self.duration), int(self.duration / gridsize))
        absolute_delays = self.delay.convert2absolute()
        positions = tuple(
            find_closest_point(points, float(delay)) for delay in absolute_delays
        )
        interpolation_size = tuple(b - a for a, b in zip(positions, positions[1:]))
        interpolations = (
            item0.interpolate(item1, steps + 1)[:-1]
            for item0, item1, steps in zip(self, self[1:], interpolation_size)
        )
        return tuple(functools.reduce(operator.add, interpolations))

    def copy(self):
        return type(self)(item.copy() for item in self)

    @property
    def delay(self) -> rhy.Compound:
        return rhy.Compound(obj.delay.copy() for obj in self)

    @property
    def duration(self) -> float:
        return sum(self.delay)


class ShadowInterpolationLine(InterpolationLine):
    """InterpolationLine that follows a particular list of events.

    The following of the events is similar to a 'shadow' of those events.

    The input argument events has the form
        ((absolute position, value), (absolute position, value), ...)

    If no event is occuring the InterpolationLine has an average value.
    If an event is occuring the InterpolationLine interpolates to the value
    of the event. The interpolation time until this new value is reached is
    declared with the shadow_size argument.

    The complete duration of the ShadowInterpolationLine is declared with
    the duration argument. If the duration is None, it will be as long as the
    latest event plus the shadow_size.
    """

    def __init__(
        self,
        average_value: float,
        shadow_size: float,
        events: tuple,
        duration: float = None,
        interpolation_type: Interpolation = Linear(),
        precision: int = 10000,
    ) -> None:
        self.__interpolation_type = interpolation_type
        self.__precision = precision

        if duration is None:
            duration = events[-1][0] + shadow_size

        try:
            assert duration > events[-1][0]
        except AssertionError:
            msg = "Complete duration has to be bigger than the farest event."
            raise ValueError(msg)

        line = []

        if events[0][0] != 0:
            if events[0][0] == shadow_size:
                line.append((0, average_value))

            elif events[0][0] > shadow_size:
                line.append((0, average_value))
                line.append((events[0][0] - shadow_size, average_value))

            else:
                line.append(
                    (
                        0,
                        self.find_value_after_shorter_duration(
                            events[0][1], average_value, shadow_size, events[0][0]
                        ),
                    )
                )

        line.append(events[0])

        for ev0, ev1 in zip(events, events[1:]):
            shadow_position0 = ev0[0] + shadow_size
            shadow_position1 = ev1[0] - shadow_size

            if shadow_position0 < shadow_position1:
                line.append((shadow_position0, average_value))
                line.append((shadow_position1, average_value))

            elif shadow_position0 == shadow_position1:
                line.append((shadow_position0, average_value))

            else:
                duration_until_next_shadow = (ev1[0] - ev0[0]) / 2

                target_value = max((ev0[1], ev1[1]))

                line.append(
                    (
                        ev0[0] + duration_until_next_shadow,
                        self.find_value_after_shorter_duration(
                            target_value,
                            average_value,
                            shadow_size,
                            duration_until_next_shadow,
                        ),
                    )
                )

            line.append(ev1)

        last_shadow_position = events[-1][0] + shadow_size

        if last_shadow_position < duration:
            line.append((last_shadow_position, average_value))
            line.append((duration, average_value))

        elif last_shadow_position == duration:
            line.append((duration, average_value))

        else:
            line.append(
                (
                    duration,
                    self.find_value_after_shorter_duration(
                        events[-1][1],
                        average_value,
                        shadow_size,
                        shadow_size - (last_shadow_position - duration),
                    ),
                )
            )

        ig0 = operator.itemgetter(0)
        ig1 = operator.itemgetter(1)

        line_delays = tuple(ig0(ev) for ev in line)
        line_values = tuple(ig1(ev) for ev in line)

        line_delays = tuple(
            b - a for a, b in zip(line_delays, line_delays[1:] + (duration,))
        )

        line = tuple(
            FloatInterpolationEvent(delay, value, interpolation_type)
            for delay, value in zip(line_delays, line_values)
        )

        super().__init__(line)

    def find_value_after_shorter_duration(
        self,
        value0: float,
        value1: float,
        usual_duration: float,
        shorter_duration: float,
    ):
        interpolation_type = self.__interpolation_type
        precision = self.__precision
        positions = Linear()(0, usual_duration, precision)
        interpolated = FloatInterpolationEvent(
            usual_duration, value0, interpolation_type
        ).interpolate(FloatInterpolationEvent(0, value1, interpolation_type), precision)
        index = bisect.bisect_left(positions, shorter_duration)
        return interpolated[index]
