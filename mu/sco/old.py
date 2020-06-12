"""This module contains musical structures that are based on discreet tones and chords."""

import functools
import math
import operator

from typing import Callable
from typing import Tuple

from mu.abstract import muobjects
from mu.mel import mel

from mu.mel.abstract import AbstractPitch
from mu.rhy import rhy
from mu.sco import abstract
from mu.time import time

from mu.utils import interpolations
from mu.utils import tools


class PitchInterpolation(interpolations.InterpolationEvent):
    def __init__(
        self,
        delay: rhy.Unit,
        pitch: AbstractPitch,
        interpolation_type: interpolations.Interpolation = interpolations.Linear(),
    ):
        super().__init__(delay, interpolation_type)
        self.__pitch = pitch

    @property
    def pitch(self):
        return self.__pitch

    def __repr__(self) -> str:

        return "PitchInter({}, {})".format(self.delay, self.pitch)

    def __hash__(self) -> int:
        return hash((hash(self.delay), hash(self.pitch), hash(self.interpolation_type)))

    def copy(self):
        return type(self)(
            rhy.Unit(self.delay), self.pitch.copy(), self.interpolation_type
        )

    def interpolate(self, other, steps: int) -> tuple:
        cents0 = self.pitch.cents
        cents1 = other.pitch.cents

        try:
            assert cents0 is not None and cents1 is not None
        except AssertionError:
            msg = "{} or {} aren't valid pitches.".format(self.pitch, other.pitch)
            raise TypeError(msg)

        return self.interpolation_type(cents0, cents1, steps)


class RhythmicInterpolation(interpolations.InterpolationEvent):
    def __init__(
        self,
        delay: rhy.Unit,
        rhythm: rhy.Unit,
        interpolation_type: interpolations.Interpolation = interpolations.Linear(),
    ):
        super().__init__(delay, interpolation_type)
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


class GlissandoLine(object):
    """Class to simulate the Glissando of a Tone.

    Only necessary input is an InterpolationLine,
    containing PitchInterpolation - objects.
    """

    def __init__(self, pitch_line: interpolations.InterpolationLine):
        self.__pitch_line = pitch_line

    def __repr__(self) -> str:
        return "GlissandoLine({})".format(self.pitch_line)

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
        up_pitch_line: interpolations.InterpolationLine,
        down_pitch_line: interpolations.InterpolationLine,
        period_size_line: interpolations.InterpolationLine,
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
        if self.direction == "up":
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


class Ovent(abstract.UniformEvent):
    """An old Event - e.g. either a Chord or a Tone."""

    _essential_attributes = ("pitch", "duration", "delay")

    def __init__(
        self,
        pitch: tuple = mel.TheEmptyPitch,
        delay: rhy.Unit = 1,
        duration: rhy.Unit = None,
        volume: float = None,
        glissando: GlissandoLine = None,
        vibrato: VibratoLine = None,
    ) -> None:

        if pitch is None:
            pitch = mel.TheEmptyPitch

        self.pitch = pitch

        if not duration:
            duration = delay

        self.duration = duration
        self.delay = delay
        self.volume = volume
        self.glissando = glissando
        self.vibrato = vibrato

    @staticmethod
    def _return_correct_time_type(time_item) -> rhy.Unit:
        if isinstance(time_item, rhy.Unit):
            return time_item
        else:
            return rhy.Unit(time_item)

    @classmethod
    def _get_standard_attributes(cls) -> tuple:
        return tools.find_attributes_of_object(cls())

    @property
    def pitch(self) -> list:
        return self._pitch

    @pitch.setter
    def pitch(self, arg: list) -> None:
        try:
            arg = list(arg)
        except TypeError:
            arg = [arg]

        self._pitch = arg

    @property
    def delay(self) -> rhy.Unit:
        return self._delay

    @delay.setter
    def delay(self, arg: rhy.Unit) -> None:
        self._delay = self._return_correct_time_type(arg)

    @property
    def duration(self) -> rhy.Unit:
        return self._duration

    @duration.setter
    def duration(self, arg: rhy.Unit) -> None:
        self._duration = self._return_correct_time_type(arg)

    def __hash__(self) -> int:
        return hash((self.pitch, self.delay, self.duration, self.volume))

    def __repr__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            str(tuple(repr(getattr(self, arg)) for arg in self._essential_attributes)),
        )

    def __eq__(self, other: "Tone") -> bool:
        attributes0 = type(self)._get_standard_attributes()
        try:
            attributes1 = type(other)._get_standard_attributes()
        except AttributeError:
            return False

        intersection = set(attributes0).intersection(attributes1)

        for essential_attribute in self._essential_attributes:
            try:
                assert essential_attribute in intersection
            except AssertionError:
                return False

        return all(
            tuple(getattr(self, attr) == getattr(other, attr) for attr in intersection)
        )

    def copy(self) -> "Ovent":
        return type(self)(
            **{arg: getattr(self, arg) for arg in type(self)._get_standard_attributes()}
        )


class Tone(Ovent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def pitch(self) -> AbstractPitch:
        return self._pitch

    @pitch.setter
    def pitch(self, arg: AbstractPitch) -> None:
        if arg is None:
            arg = mel.TheEmptyPitch
        try:
            assert isinstance(arg, AbstractPitch)
        except AssertionError:
            msg = "Pitch argument has to be either 'None' or a subclass of"
            msg += " 'AbstractPitch' and not type '{}'!".format(type(arg))
            raise TypeError(msg)

        self._pitch = arg


class Rest(Tone):
    def __init__(
        self, delay: rhy.Unit = 1, duration: rhy.Unit = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if duration is None:
            duration = delay

        self.duration = duration
        self.delay = delay
        self.volume = 0

    def __repr__(self):
        return "Rest({})".format(repr(self.delay))

    @property
    def pitch(self):
        return mel.TheEmptyPitch

    @pitch.setter
    def pitch(self, arg):
        pass


class Chord(Ovent):
    """A Chord contains simultanly played Tones."""

    def __init__(self, harmony: mel.Harmony = mel.Harmony([]), *args, **kwargs) -> None:
        super().__init__(harmony, *args, **kwargs)


class AbstractLine(abstract.MultiSequentialEvent):
    """Abstract superclass for subclasses that describe specific events on a time line.

    The specific attributes of those events are:
        * pitch
        * delay
        * duration
        * volume

    Examples of those events would be: Tone or Chord.
    """

    def __init__(self, iterable, time_measure="relative"):
        abstract.MultiSequentialEvent.__init__(self, iterable)
        try:
            assert any((time_measure == "relative", time_measure == "absolute"))
        except AssertionError:
            raise ValueError("Time can only be 'relative' or 'absolute'.")
        self._time_measure = time_measure

    def copy(self):
        copied = super().copy()
        copied._time_measure = str(self._time_measure)
        return copied

    @property
    def time_measure(self):
        return self._time_measure

    def convert2absolute(self) -> "AbstractLine":
        """Change time dimension of the object.

        Delay becomes the starting time of the specific event,
        duration becomes the stoptime of the specific event.
        """
        copy = self.copy()

        if self.time_measure == "relative":
            copy.delay = rhy.Compound(copy.delay).convert2absolute()
            stop = tuple((d + s) for d, s in zip(copy.delay, copy.dur))
            copy.dur = stop
            copy._time_measure = "absolute"

        return copy

    def convert2relative(self):
        """Change time dimension of the object.

        Starting time of specific event becomes its Delay ,
        stoptime of specific event becomes its duration.
        """
        copy = self.copy()
        if self.time_measure == "absolute":
            delay = rhy.Compound(copy.delay).convert2relative()
            copy.dur = tuple(dur - delay for dur, delay in zip(self.dur, self.delay))
            delay.append(copy.dur[-1])
            copy.delay = delay
            copy._time_measure = "relative"
        return copy

    @property
    def freq(self) -> Tuple[float]:
        return tuple(p.freq for p in self.pitch)

    @property
    def dur(self) -> abstract._LinkedList:
        return self.__get_duration__()

    @dur.setter
    def dur(self, arg) -> None:
        self.__set_duration__(arg)

    @property
    def duration(self) -> time.Time:
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

        if self.time_measure == "relative":
            new.dur = tuple(du - de for du, de in zip(new.dur, new.delay))
            new.delay = tuple(new.dur)

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


class OventLine(AbstractLine):
    """A Melody contains sequentially played Ovents."""

    _object = Ovent()


class Melody(AbstractLine):
    """A Melody contains sequentially played Pitches."""

    _object = Tone()

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


class Cadence(AbstractLine):
    """A Cadence contains sequentially played Harmonies."""

    _object = Chord()

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
                    new.insert(0, Rest(start, new[0].delay))
            else:
                new.append(Rest(start, stop))

            polyline[i] = type(polyline[i])(new, "absolute")

        if hard_cut is False:
            earliest = min(sub.delay[0] for sub in polyline)
            if earliest < start:
                for i, sub in enumerate(polyline):
                    if sub.delay[0] > earliest:
                        sub.insert(0, Rest(earliest, sub.delay[0]))
                        polyline[i] = sub

        if self.time_measure == "relative":
            for sub in polyline:
                sub.dur = tuple(d - sub.delay[0] for d in sub.dur)
                sub.delay = tuple(d - sub.delay[0] for d in sub.delay)

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
            time = float(time)
            start = float(tone.delay)
            duration = float(tone.duration)
            return time >= start and time < (start + duration)

        return self.pop_by(test, time)

    def convert2melody(self) -> Melody:
        sorted_by_delay = sorted(list(self.copy()), key=lambda t: t.delay)
        first = sorted_by_delay[0].delay
        if first != 0:
            sorted_by_delay.insert(0, Rest(0, first))
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
