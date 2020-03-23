import bisect
import functools
import math
import operator
from typing import Callable
from typing import Optional
from typing import Tuple

from mu.abstract import muobjects
from mu.mel import ji
from mu.mel import mel

from mu.mel.abstract import AbstractPitch
from mu.rhy import rhy
from mu.sco import abstract
from mu.time import time
from mu.utils import interpolation


"""This module represents musical structures that are based on discreet tones."""


class InterpolationEvent(abstract.UniformEvent):
    def __init__(self, delay: rhy.Unit, interpolation_type=interpolation.Linear()):
        if isinstance(delay, rhy.Unit) is False:
            delay = rhy.Unit(delay)
        self.__delay = delay
        self.__interpolation_type = interpolation_type

    @property
    def delay(self):
        return self.__delay

    @property
    def interpolation_type(self):
        return self.__interpolation_type

    @abstract.abc.abstractproperty
    def interpolate(self, other, steps):
        raise NotImplementedError


class PitchInterpolation(InterpolationEvent):
    def __init__(
        self,
        delay: rhy.Unit,
        pitch: AbstractPitch,
        interpolation_type: interpolation.Interpolation = interpolation.Linear(),
    ):
        InterpolationEvent.__init__(self, delay, interpolation_type)
        self.__pitch = pitch

    @property
    def pitch(self):
        return self.__pitch

    def __hash__(self) -> int:
        return hash((hash(self.delay), hash(self.pitch), hash(self.interpolation_type)))

    def copy(self):
        return type(self)(
            rhy.Unit(self.delay), self.pitch.copy(), self.interpolation_type
        )

    def interpolate(self, other, steps) -> tuple:
        cents0 = self.pitch.cents
        cents1 = other.pitch.cents
        return self.interpolation_type(cents0, cents1, steps)


class RhythmicInterpolation(InterpolationEvent):
    def __init__(
        self,
        delay: rhy.Unit,
        rhythm: rhy.Unit,
        interpolation_type: interpolation.Interpolation = interpolation.Linear(),
    ):
        InterpolationEvent.__init__(self, delay, interpolation_type)
        self.__rhythm = rhythm

    @property
    def rhythm(self):
        return self.__rhythm

    def __hash__(self) -> int:
        return hash(
            (hash(self.delay), hash(self.rhythm), hash(self.interpolation_type))
        )

    def copy(self):
        return type(self)(
            rhy.Unit(self.delay), self.rhythm.copy(), self.interpolation_type
        )

    def interpolate(self, other, steps) -> tuple:
        return self.interpolation_type(self.rhythm, other.rhythm, steps)


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

        points = interpolation.Linear()(
            0, float(self.duration), int(self.duration / gridsize)
        )
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
    def duration(self):
        return sum(self.delay)


class GlissandoLine(object):
    """Class to simulate the Glissando of a Tone.

    Only necessary input is an InterpolationLine,
    containing PitchInterpolation - objects.
    """

    def __init__(self, pitch_line: InterpolationLine):
        self.__pitch_line = pitch_line

    @property
    def pitch_line(self):
        return self.__pitch_line

    def interpolate(self, gridsize: float) -> tuple:
        """Return tuple filled with cent values."""
        return self.pitch_line(gridsize)


class VibratoLine(object):
    """Class to simulate the Vibrato of a Tone.

    up_pitch_line is in the InterpolationLine for
    the maximum pitch to go up. The pitch object is
    expected to have a positve cent value.
    down_pitch_line is the InterpolationLine for
    the maximum pitch to go down. The pitch object
    is expected to have negative cent value.
    The period_size_line describes the size of one
    vibrato-period where the pitch goes once down
    and once up.
    The direction argument specifies whether
    the vibrato goes first to the upper pitch
    or first to the lower pitch.
    """

    def __init__(
        self,
        up_pitch_line: InterpolationLine,
        down_pitch_line: InterpolationLine,
        period_size_line: InterpolationLine,
        direction="up",
    ):
        self.direction = direction
        self.__up_pitch_line = up_pitch_line
        self.__down_pitch_line = down_pitch_line
        self.__period_size_line = period_size_line

    @property
    def direction(self):
        return self.__direction

    @direction.setter
    def direction(self, direction: str):
        try:
            assert direction == "up" or direction == "down"
        except AssertionError:
            raise ValueError("Direction has to be 'up' or 'down'!")
        self.__direction = direction

    @property
    def up_pitch_line(self):
        return self.__up_pitch_line.copy()

    @property
    def down_pitch_line(self):
        return self.__down_pitch_line.copy()

    @property
    def period_size_line(self):
        return self.__period_size_line.copy()

    def calculate_pitch_size(self, period_position, max_up, max_down) -> float:
        if self.direction is "up":
            max_cent = max_up
            min_cent = max_down
        else:
            min_cent = max_up
            max_cent = max_down
        percent = round(math.sin(period_position * 2 * math.pi), 5)
        if percent > 0:
            return max_cent * percent
        else:
            return min_cent * abs(percent)

    def interpolate(self, gridsize: float) -> tuple:
        """Return a tuple filled with cent values."""
        up_pitch_line = self.__up_pitch_line(gridsize)
        down_pitch_line = self.__down_pitch_line(gridsize)
        period_size_line = self.__period_size_line(gridsize)
        generator = zip(up_pitch_line, down_pitch_line, period_size_line)
        acc = 0
        cents = []
        for up_pitch, down_pitch, period_size in generator:
            current = acc * gridsize
            if current >= period_size:
                acc = 0
                current = 0
            cent = self.calculate_pitch_size(
                current / period_size, up_pitch, down_pitch
            )
            cents.append(cent)
            acc += 1
        return tuple(cents)


class Tone(abstract.UniformEvent):
    def __init__(
        self,
        pitch: Optional[AbstractPitch],
        delay: rhy.Unit,
        duration: Optional[rhy.Unit] = None,
        volume: Optional = None,
        glissando: GlissandoLine = None,
        vibrato: VibratoLine = None,
    ) -> None:
        if pitch is None:
            pitch = mel.EmptyPitch()
        self.pitch = pitch
        if isinstance(delay, rhy.Unit) is False:
            delay = rhy.Unit(delay)
        if not duration:
            duration = delay
        elif isinstance(duration, rhy.Unit) is False:
            duration = rhy.Unit(duration)
        self._dur = duration
        self.delay = delay
        self.volume = volume
        self.glissando = glissando
        self.vibrato = vibrato

    def __hash__(self) -> int:
        return hash((self.pitch, self.delay, self.duration, self.volume))

    def __repr__(self) -> str:
        return str(
            (repr(self.pitch), repr(self.delay), repr(self.duration), repr(self.volume))
        )

    def __eq__(self, other: "Tone") -> bool:
        return all(
            (
                self.pitch == other.pitch,
                self.duration == other.duration,
                self.delay == other.delay,
            )
        )

    def copy(self) -> "Tone":
        return type(self)(
            self.pitch.copy(),
            self.delay.copy(),
            self.duration.copy(),
            self.volume,
            self.glissando,
            self.vibrato,
        )


class Rest(Tone):
    def __init__(self, delay: rhy.Unit, duration: rhy.Unit = None) -> None:
        if duration is None:
            duration = delay

        self._dur = duration
        self.delay = delay
        self.volume = 0

    def __repr__(self):
        return repr(self.delay)

    @property
    def pitch(self):
        return mel.EmptyPitch()

    @property
    def harmony(self):
        return mel.Harmony((self.pitch,))

    @property
    def glissando(self):
        return None

    @property
    def vibrato(self):
        return None

    def copy(self):
        return type(self)(self.delay.copy())


class Chord(abstract.SimultanEvent):
    """A Chord contains simultanly played Tones."""

    def __init__(
        self, harmony, delay: rhy.Unit, duration: Optional[rhy.Unit] = None, volume=None
    ) -> None:
        if isinstance(delay, rhy.Unit) is False:
            delay = rhy.Unit(delay)
        if not duration:
            duration = delay
        elif isinstance(duration, rhy.Unit) is False:
            duration = rhy.Unit(duration)

        self.harmony = harmony
        self._dur = duration
        self.delay = delay
        self.volume = volume

    @property
    def pitch(self):
        return self.harmony

    @pitch.setter
    def pitch(self, arg):
        self.harmony = arg

    @property
    def duration(self):
        return self._dur

    @duration.setter
    def duration(self, dur):
        self._dur = dur

    def __repr__(self):
        return str((repr(self.harmony), repr(self.delay), repr(self.duration)))

    def copy(self) -> "Chord":
        return type(self)(
            self.pitch.copy(), self.delay.copy(), self.duration.copy(), self.volume
        )


class AbstractLine(abstract.MultiSequentialEvent):
    """Abstract superclass for subclasses that describe specific events on a time line.

    The specific attributes of those events are:
        * pitch
        * delay
        * duration
        * volume

    Examples of those events would be: Tone or Chord.
    """

    _sub_sequences_class_names = ("pitch", "delay", "dur", "volume")

    def __init__(self, iterable, time_measure="relative"):
        abstract.MultiSequentialEvent.__init__(self, iterable)
        try:
            assert any((time_measure == "relative", time_measure == "absolute"))
        except AssertionError:
            raise ValueError("Time can only be 'relative' or 'absolute'.")
        self._time_measure = time_measure

    def copy(self):
        copied = abstract.MultiSequentialEvent.copy(self)
        copied._time_measure = str(self._time_measure)
        return copied

    @property
    def time_measure(self):
        return self._time_measure

    def __hash__(self):
        return hash(tuple(hash(item) for item in self))

    def convert2absolute(self) -> "AbstractLine":
        """Change time dimension of the object.

        Delay becomes the starting time of the specific event,
        duration becomes the stoptime of the specific event.
        """
        copy = self.copy()
        if self.time_measure == "relative":
            copy.delay = copy.delay.convert2absolute()
            stop = ((d + s) for d, s in zip(copy.delay, copy.dur))
            copy.dur = type(copy.dur)(stop)
            copy._time_measure = "absolute"
        return copy

    def convert2relative(self):
        """Change time dimension of the object.

        Starting time of specific event becomes its Delay ,
        stoptime of specific event becomes its duration.
        """
        copy = self.copy()
        if self.time_measure == "absolute":
            copy.delay = copy.delay.convert2relative()
            copy.dur = type(copy.dur)(
                dur - delay for dur, delay in zip(self.dur, self.delay)
            )
            copy.delay.append(copy.dur[-1])
            copy._time_measure = "relative"
        return copy

    @property
    def freq(self) -> Tuple[float]:
        return self.pitch.freq

    @property
    def duration(self):
        if self.time_measure == "relative":
            return time.Time(sum(self.delay))
        else:
            return time.Time(self.dur[-1])

    def tie_by(self, function):
        tied = function(list(self))
        copied = self.copy()
        for i, item in enumerate(tied):
            copied[i] = item
        copied = copied[: len(tied)]
        return copied

    def tie(self):
        def sub(line):
            new = []
            for i, it0 in enumerate(line):
                try:
                    it1 = line[i + 1]
                except IndexError:
                    new.append(it0)
                    break
                if it0.duration >= it0.delay and it0.pitch == it1.pitch:
                    t_new = type(it0)(
                        it0.pitch, it0.duration + it1.delay, it0.duration + it1.duration
                    )
                    return new + sub([t_new] + line[i + 2 :])
                else:
                    new.append(it0)
            return new

        return self.tie_by(sub)

    def discard_rests(self):
        def is_rest(pitch) -> bool:
            if isinstance(pitch, AbstractPitch):
                return pitch.is_empty
            elif pitch is None:
                return True
            else:
                return len(pitch) == 0

        def sub(line):
            new = []
            for i, it0 in enumerate(line):
                if i != 0:
                    if is_rest(it0.pitch):
                        new[-1].delay += it0.delay
                        new[-1].duration += it0.duration
                    else:
                        new.append(it0)
                else:
                    new.append(it0)
            return new

        return self.tie_by(sub)

    def cut_up_by_time(
        self, start: rhy.Unit, stop: rhy.Unit, add_earlier=False, hard_cut=True
    ) -> "AbstractLine":
        line = self.convert2absolute()
        new = []

        for event in line:
            ev_start = event.delay
            ev_stop = event.duration

            appendable_conditions = (
                ev_start >= start and ev_start < stop,
                ev_stop <= stop and ev_stop > start,
                ev_start <= start and ev_stop >= stop,
            )

            appendable = any(appendable_conditions)

            if ev_start < start and appendable:
                appendable = bool(add_earlier)

            if appendable:
                if hard_cut:
                    if ev_stop > stop:
                        event.duration = stop
                    if ev_start < start:
                        event.delay = start

                new.append(event)

        if new:
            if new[0].delay > start:
                new.insert(0, Rest(start, new[0].delay))
        else:
            new.append(Rest(start, stop))

        new = type(self)(new)

        if self.time_measure is "relative":
            new.dur = type(new.dur)(du - de for du, de in zip(new.dur, new.delay))
            new.delay = type(new.delay)(new.dur)

        return new

    def cut_up_by_idx(
        self, itemidx, add_earlier=False, hard_cut=True
    ) -> "AbstractLine":
        item = self.convert2absolute()[itemidx]
        item_start = item.delay
        item_stop = item.duration
        return self.cut_up_by_time(item_start, item_stop, hard_cut, add_earlier)

    def split(self):
        """Split items to Item-Rest pairs if their delay is longer than their duration.

        """
        new = []
        for item in self:
            diff = item.delay - item.duration
            if diff > 0:
                new.append(type(item)(item.pitch, item.duration, volume=item.volume))
                new.append(Rest(diff))
            else:
                new.append(item)
        return type(self)(new)


class Melody(AbstractLine):
    """A Melody contains sequentially played Pitches."""

    _obj_class = Tone
    _sub_sequences_class = (mel.Mel, rhy.Compound, rhy.Compound, list, list, list)
    _sub_sequences_class_names = (
        "pitch",
        "delay",
        "dur",
        "volume",
        "glissando",
        "vibrato",
    )

    @classmethod
    def subvert_object(cls, tone: Tone):
        return (
            tone.pitch,
            tone.delay,
            tone.duration,
            tone.volume,
            tone.glissando,
            tone.vibrato,
        )

    @property
    def freq(self) -> Tuple[float]:
        return self.mel.freq

    # @property
    # def duration(self):
    #     return float(sum(element.delay for element in self))

    @property
    def mel(self):
        """Alias for backwards compatibility,"""
        return self.pitch

    @mel.setter
    def mel(self, arg):
        self.pitch = arg

    @property
    def rhy(self):
        """Alias for backwards compatibility,"""
        return self.delay

    @rhy.setter
    def rhy(self, arg):
        self.delay = arg

    def __hash__(self):
        return hash(tuple(hash(t) for t in self))

    def tie_pauses(self):
        def sub(melody):
            new = []
            for i, it0 in enumerate(melody):
                try:
                    it1 = melody[i + 1]
                except IndexError:
                    new.append(it0)
                    break
                pitch_test = (
                    it0.pitch == mel.EmptyPitch(),
                    it1.pitch == mel.EmptyPitch(),
                )
                # if it0.duration == it0.delay and all(pitch_test):
                if all(pitch_test):
                    t_new = type(it0)(
                        it0.pitch, it0.duration + it1.delay, it0.duration + it1.duration
                    )
                    return new + sub([t_new] + melody[i + 2 :])
                else:
                    new.append(it0)
            return new

        return self.tie_by(sub)


class JIMelody(Melody):
    _sub_sequences_class = (ji.JIMel, rhy.Compound, rhy.Compound, list, list, list)


class Cadence(AbstractLine):
    """A Cadence contains sequentially played Harmonies."""

    _obj_class = Chord
    _sub_sequences_class = (mel.Cadence, rhy.Compound, rhy.Compound, list)

    @property
    def harmony(self):
        """Alias for backwards compatibility."""
        return self.pitch

    @harmony.setter
    def harmony(self, arg):
        self.pitch = arg

    @property
    def rhy(self):
        """Alias for backwards compatibility."""
        return self.delay

    @rhy.setter
    def rhy(self, arg):
        self.delay = arg

    @classmethod
    def subvert_object(cls, chord):
        return chord.harmony, chord.delay, chord.duration, chord.volume

    @property
    def freq(self):
        return self.harmony.freq

    def tie_pauses(self):
        def sub(melody):
            new = []
            for i, it0 in enumerate(melody):
                try:
                    it1 = melody[i + 1]
                except IndexError:
                    new.append(it0)
                    break
                pitch_test = (
                    all(p == mel.EmptyPitch() for p in it0.pitch),
                    all(p == mel.EmptyPitch() for p in it1.pitch),
                )
                if all(pitch_test):
                    t_new = type(it0)(
                        it0.pitch, it0.duration + it1.delay, it0.duration + it1.duration
                    )
                    return new + sub([t_new] + melody[i + 2 :])
                else:
                    new.append(it0)
            return new

        return self.tie_by(sub)


class JICadence(Cadence):
    _sub_sequences_class = (ji.JICadence, rhy.Compound, rhy.Compound, list)


class PolyLine(abstract.SimultanEvent):
    """A Container for Melody and Cadence - Objects."""

    @staticmethod
    def find_simultan_events_in_absolute_polyline(polyline, polyidx, itemidx):
        item = polyline[polyidx][itemidx]
        istart = item.delay
        istop = item.duration
        simultan = []
        for pnum, poly in enumerate(polyline):
            for enum, event in enumerate(poly):
                if (pnum, enum) != (polyidx, itemidx):
                    estart = event.delay
                    estop = event.duration
                    if estart >= istart and estart < istop:
                        simultan.append(event)
                    elif estop <= istop and estop > istart:
                        simultan.append(event)
                    elif estart <= istart and estop >= istop:
                        simultan.append(event)
        return tuple(simultan)

    def __init__(self, iterable, time_measure="relative"):
        abstract.SimultanEvent.__init__(self, iterable)
        self._time_measure = time_measure

    @property
    def time_measure(self):
        return self._time_measure

    def copy(self):
        copied = abstract.SimultanEvent.copy(self)
        copied._time_measure = str(self._time_measure)
        return copied

    def fill(self) -> "PolyLine":
        """Add Rests until each Voice has the same length.

        so that: sum(Polyphon[0].rhy) == sum(Polyphon[1].rhy) == ...
        """
        poly = self.copy()
        total = self.duration
        for v in poly:
            summed = sum(v.delay)
            if summed < total:
                v.append(Rest(rhy.Unit(total - summed)))
        return poly

    @property
    def duration(self) -> time.Time:
        dur_sub = tuple(element.duration for element in self)
        try:
            return time.Time(max(dur_sub))
        except ValueError:
            return None

    def convert2absolute(self):
        """Change time dimension.

        Delay becomes the starting time of the specific event,
        duration becomes the stoptime of the specific event.
        """
        copy = self.copy()
        if self.time_measure == "relative":
            for i, item in enumerate(copy):
                copy[i] = item.convert2absolute()
            copy._time_measure = "absolute"
        return copy

    def convert2relative(self):
        """Change time dimension.

        Starting time of specific event becomes its Delay,
        stoptime of specific event becomes its duration.
        """
        copy = self.copy()
        if self.time_measure == "absolute":
            for i, item in enumerate(copy):
                copy[i] = item.convert2relative()
            copy._time_measure = "relative"
        return copy

    def horizontal_add(self, other: "PolyLine", fill=True) -> "PolyLine":
        voices = []
        poly0 = self.copy()
        if fill is True:
            poly0 = poly0.fill()
        for m0, m1 in zip(poly0, other):
            voices.append(m0 + m1)
        length0 = len(self)
        length1 = len(other)
        for m_rest in poly0[length1:]:
            voices.append(m_rest.copy())
        for m_rest in other[length0:]:
            m_rest = type(m_rest)([Rest(poly0.duration)]) + m_rest
            voices.append(m_rest.copy())
        res = type(self)(voices)
        res = res.fill()
        return res

    def find_simultan_events(self, polyidx, itemidx) -> tuple:
        converted_poly = self.convert2absolute()
        return self.find_simultan_events_in_absolute_polyline(
            converted_poly, polyidx, itemidx
        )

    def find_exact_simultan_events(
        self, polyidx, itemidx, convert2relative=True
    ) -> tuple:
        converted_poly = self.convert2absolute()
        simultan = self.find_simultan_events_in_absolute_polyline(
            converted_poly, polyidx, itemidx
        )
        item = converted_poly[polyidx][itemidx]
        item_start = item.delay
        item_stop = item.duration
        for event in simultan:
            if event.delay < item_start:
                event.delay = item_start
            if event.duration > item_stop:
                event.duration = item_stop
        if convert2relative is True:
            for event in simultan:
                event.duration = event.duration - event.delay
                event.delay = event.duration
        return simultan

    def cut_up_by_time(
        self, start: rhy.Unit, stop: rhy.Unit, hard_cut=True, add_earlier=True
    ) -> "PolyLine":
        polyline = self.convert2absolute()
        for i, sub in enumerate(polyline):
            new = []
            for event in sub:
                appendable = False
                ev_start = event.delay
                ev_stop = event.duration
                if ev_start >= start and ev_start < stop:
                    appendable = True
                elif ev_stop <= stop and ev_stop > start:
                    appendable = True
                elif ev_start <= start and ev_stop >= stop:
                    appendable = True
                if ev_start < start and appendable is True:
                    if add_earlier is True:
                        appendable = True
                    else:
                        appendable = False
                if appendable is True:
                    if hard_cut is True:
                        if ev_stop > stop:
                            event.duration = stop
                        if ev_start < start:
                            event.delay = start
                    new.append(event)
            if new:
                if new[0].delay > start:
                    new.insert(0, Rest(start, start))
            else:
                new.append(Rest(start, stop))
            polyline[i] = type(polyline[i])(new, "absolute")

        if hard_cut is False:
            earliest = min(sub.delay[0] for sub in polyline)
            if earliest < start:
                for i, sub in enumerate(polyline):
                    if sub.delay[0] > earliest:
                        sub.insert(0, Rest(earliest, earliest))
                        polyline[i] = sub

        if self.time_measure is "relative":
            for sub in polyline:
                sub.dur = type(sub.dur)(d - sub.delay[0] for d in sub.dur)
                sub.delay = type(sub.delay)(d - sub.delay[0] for d in sub.delay)
            polyline = polyline.convert2relative()

        return polyline

    def cut_up_by_idx(
        self, polyidx, itemidx, hard_cut=True, add_earlier=True
    ) -> "PolyLine":
        item = self[polyidx].convert2absolute()[itemidx]
        item_start = item.delay
        item_stop = item.duration
        return self.cut_up_by_time(item_start, item_stop, hard_cut, add_earlier)


class Polyphon(PolyLine):
    """Container for Melody - Objects."""

    def chordify(
        self, cadence_class=Cadence, harmony_class=mel.Harmony, add_longer=False
    ) -> Cadence:
        """Return chordal reduction of polyphonic music.

        Each change of a pitch results in a new chord.
        """
        events = functools.reduce(
            operator.add, tuple(line.convert2absolute() for line in self)
        )
        starts = tuple(ev.delay for ev in events)
        stops = tuple(ev.duration for ev in events)
        positions = sorted(set(starts + stops))
        available_chords = len(positions) - 1
        harmonies = [[] for i in range(available_chords)]
        volumes = [[] for i in range(available_chords)]

        for event in events:
            indices = [positions.index(event.delay)]

            if add_longer:
                for position in positions[indices[0] + 1 :]:
                    if position < event.duration:
                        indices.append(indices[-1] + 1)

            pitch = event.pitch
            volume = event.volume
            for idx in indices:
                harmonies[idx].append(pitch)
                if volume is not None:
                    volumes[idx].append(volume)

        rhythms = tuple(b - a for a, b in zip(positions, positions[1:]))
        volumes = [sum(v) / len(v) if v else None for v in volumes]

        return cadence_class(
            Chord(harmony_class(h), r, r, volume=v)
            for h, r, v in zip(harmonies, rhythms, volumes)
        )


class Instrument(object):
    def __init__(self, name, pitches):
        self.name = name
        self.pitches = pitches

    def __repr__(self):
        return repr(self.name)

    def copy(self):
        return type(self)(self.name, self.pitches)


class Ensemble(muobjects.MUDict):
    melody_class = Melody

    def get_instrument_by_pitch(self, pitch):
        """return all Instruments, which could play the asked pitch"""
        possible = []
        for instr in self:
            if pitch in instr.pitches:
                possible.append(instr.name)
        return possible


class ToneSet(muobjects.MUSet):
    @classmethod
    def from_melody(cls, melody: Melody) -> "ToneSet":
        return cls.from_polyphon(Polyphon([melody]))

    @classmethod
    def from_polyphon(cls, polyphon: Polyphon) -> "ToneSet":
        new_set = cls()
        for melody in polyphon:
            d = 0
            for t in melody.copy():
                delay = float(t.delay)
                t.delay = rhy.Unit(d)
                d += delay
                new_set.add(t)
        return new_set

    @classmethod
    def from_cadence(cls, cadence: Cadence) -> "ToneSet":
        new_set = cls()
        d = 0
        for chord in cadence:
            delay = float(chord.delay)
            for p in chord.pitch:
                t = Tone(p, rhy.Unit(d), chord.duration, chord.volume)
                new_set.add(t)
            d += delay
        return new_set

    def pop_by(self, test: Callable, args) -> "ToneSet":
        new_set = ToneSet()
        for arg in args:
            for t in self:
                if test(t, arg):
                    new_set.add(t)
        return new_set

    def pop_by_pitch(self, *pitch) -> "ToneSet":
        return self.pop_by(lambda t, p: t.pitch == p, pitch)

    def pop_by_duration(self, *duration) -> "ToneSet":
        return self.pop_by(lambda t, d: t.duration == d, duration)

    def pop_by_start(self, *start) -> "ToneSet":
        return self.pop_by(lambda t, s: t.delay == s, start)

    def pop_by_time(self, *time) -> "ToneSet":
        def test(tone, time):
            start = tone.delay
            duration = tone.duration
            return time >= start and time < start + duration

        return self.pop_by(test, time)

    def convert2melody(self) -> Melody:
        sorted_by_delay = sorted(list(self.copy()), key=lambda t: t.delay)
        first = sorted_by_delay[0].delay
        if first != 0:
            sorted_by_delay.insert(0, Rest(0))
        for t, t_after in zip(sorted_by_delay, sorted_by_delay[1:]):
            diff = t_after.delay - t.delay
            t.delay = rhy.Unit(diff)
        sorted_by_delay[-1].delay = rhy.Unit(sorted_by_delay[-1].duration)
        return Melody(sorted_by_delay)

    def convert2cadence(self) -> Cadence:
        sorted_by_delay = sorted(list(self.copy()), key=lambda t: t.delay)
        first = sorted_by_delay[0].delay
        if first != 0:
            sorted_by_delay.insert(0, Rest(0))
        cadence = Cadence([])
        harmony = mel.Harmony([])
        for t, t_after in zip(sorted_by_delay, sorted_by_delay[1:] + [0]):
            try:
                diff = t_after.delay - t.delay
            except AttributeError:
                diff = t.duration
            harmony.add(t.pitch)
            if diff != 0:
                cadence.append(Chord(harmony, rhy.Unit(diff), rhy.Unit(t.duration)))
                harmony = mel.Harmony([])
        cadence[-1].delay = rhy.Unit(sorted_by_delay[-1].duration)
        return Cadence(cadence)
