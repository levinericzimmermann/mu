# @Author: Levin Eric Zimmermann
# @Date:   2018-02-07T18:18:59+01:00
# @Email:  levin-eric.zimmermann@folkwang-uni.de
# @Project: mu
# @Last modified by:   uummoo
# @Last modified time: 2018-04-06T14:53:22+02:00


from typing import Callable, Optional, Tuple, Union
from mu.abstract import muobjects
from mu.sco import abstract
from mu.rhy import rhy
from mu.mel import mel
from mu.mel import ji
from mu.mel.abstract import AbstractPitch
from mu.utils import music21
from mu.time import time


class Tone(abstract.UniformEvent):
    def __init__(self, pitch: Optional[AbstractPitch], delay: rhy.RhyUnit,
                 duration: Optional[rhy.RhyUnit] = None,
                 volume: Optional = None) -> None:
        if not duration:
            duration = delay
        if pitch is None:
            pitch = mel.EmptyPitch()
        self.pitch = pitch
        self._dur = rhy.RhyUnit(duration)
        self.delay = rhy.RhyUnit(delay)
        self.volume = volume

    def __hash__(self) -> int:
        return hash((self.pitch, self.delay, self.duration))

    def __repr__(self):
        return str((repr(self.pitch), repr(self.delay), repr(self.duration)))

    def __eq__(self, other: "Tone") -> bool:
        return all((self.pitch == other.pitch, self.duration == other.duration,
                    self.delay == other.delay))

    def copy(self) -> "Tone":
        return type(self)(self.pitch.copy(),
                          self.delay.copy(),
                          self.duration.copy())

    @music21.decorator
    def convert2music21(self):
        stream = music21.m21.stream.Stream()
        duration_mu = self.duration
        duration = duration_mu.convert2music21()
        if self.pitch is not None:
            pitch = self.pitch.convert2music21()
            if duration_mu > 4:
                am_4 = int(duration_mu // 4)
                rest = int(duration_mu % 4)
                for i in range(am_4):
                    stream.append(music21.m21.note.Note(
                        pitch,
                        duration=music21.m21.duration.Duration(4),
                        tie=music21.m21.tie.Tie()))
                if rest > 0:
                    stream.append(music21.m21.note.Note(
                        pitch,
                        duration=music21.m21.duration.Duration(rest),
                        tie=music21.m21.tie.Tie("stop")))
            else:
                stream.append(music21.m21.note.Note(pitch, duration=duration))
            difference = self.delay - self.duration
            if difference > 0:
                rhythm = rhy.RhyUnit(difference).convert2music21()
                stream.append(music21.m21.note.Rest(duration=rhythm))
        else:
            if duration > 4:
                am_4 = duration // 4
                rest = duration % 4
                for i in range(am_4):
                    stream.append(music21.m21.note.Rest(duration=4))
                stream.append(music21.m21.note.Rest(duration=rest))
            else:
                stream.append(music21.m21.note.Rest(duration=duration))
        return stream


class Rest(Tone):
    def __init__(self, delay: rhy.RhyUnit,
                 duration: rhy.RhyUnit=rhy.RhyUnit(0)) -> None:
        self._dur = duration
        self.delay = delay

    def __repr__(self):
        return repr(self.duration)

    @property
    def pitch(self):
        return mel.EmptyPitch()

    def copy(self):
        return type(self)(self.delay.copy())


class Chord(abstract.SimultanEvent):
    """
    A Chord contains simultanly played Tones.
    """

    def __init__(self, harmony, delay: rhy.RhyUnit,
                 duration: Optional[rhy.RhyUnit] = None) -> None:
        if not duration:
            duration = delay
        self.harmony = harmony
        self._dur = duration
        self.delay = delay

    @property
    def pitch(self):
        return self.harmony

    @pitch.setter
    def pitch(self, arg):
        self.harmony = arg

    @property
    def duration(self):
        return self._dur

    def __repr__(self):
        return str((repr(self.harmony), repr(self.delay), repr(self.duration)))

    @music21.decorator
    def convert2music21(self):
        # TODO: make a proper implementation of this
        stream = music21.m21.stream.Stream()
        pitches = tuple(p.convert2music21() for p in self.harmony)
        duration_mu = float(self.duration)
        duration = self.duration.convert2music21()
        difference = self.delay - self.duration
        if pitches:
            if duration_mu > 4:
                am_4 = int(duration_mu // 4)
                rest = duration_mu % 4
                for i in range(am_4):
                    if i == 0:
                        tie = "start"
                    elif i == am_4 - 1 and rest < 1:
                        tie = "stop"
                    else:
                        tie = "continue"
                    chord = music21.m21.chord.Chord(
                        tuple(p.convert2music21() for p in self.harmony),
                        duration=music21.m21.duration.Duration(4))
                    chord.tie = music21.m21.tie.Tie(tie)
                    stream.append(chord)
                if rest > 0:
                    chord = music21.m21.chord.Chord(
                        tuple(p.convert2music21() for p in self.harmony),
                        duration=music21.m21.duration.Duration(rest))
                    chord.tie = music21.m21.tie.Tie("stop")
                    stream.append(chord)

            else:
                chord = music21.m21.chord.Chord(
                    pitches, duration=duration)
                stream.append(chord)

        else:
            chord = music21.m21.note.Rest(duration=duration)
            stream.append(chord)
        if difference > 0:
            rhythm = rhy.RhyUnit(difference).convert2music21()
            stream.append(music21.m21.note.Rest(duration=rhythm))
        return stream


class AbstractLine(abstract.MultiSequentialEvent):
    _sub_sequences_class_names = ("pitch", "delay", "dur")

    def __init__(self, iterable, time_measure="relative"):
        abstract.MultiSequentialEvent.__init__(self, iterable)
        try:
            assert(any((time_measure == "relative",
                        time_measure == "absolute")))
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

    @property
    def freq(self) -> Tuple[float]:
        return self.pitch.freq

    @property
    def duration(self):
        return time.Time(sum(self.delay))

    def __hash__(self):
        return hash(tuple(hash(item) for item in self))

    def tie(self):
        def sub(melody):
            new = []
            for i, it0 in enumerate(melody):
                try:
                    it1 = melody[i + 1]
                except IndexError:
                    new.append(it0)
                    break
                if it0.duration == it0.delay and it0.pitch == it1.pitch:
                    t_new = type(it0)(it0.pitch,
                                      it0.duration + it1.delay,
                                      it0.duration + it1.duration)
                    return new + sub([t_new] + melody[i + 2:])
                else:
                    new.append(it0)
            return new
        return type(self)(sub(list(self)))

    def split(self):
        """
        Split items, whose delay may be longer than their
        duration into Item-Rest - Pairs.
        """
        new = []
        for item in self:
            diff = item.delay - item.duration
            if diff > 0:
                new.append(type(item)(item.pitch, item.duration))
                new.append(Rest(diff))
            else:
                new.append(item)
        return type(self)(new)

    def convert2absolute_time(self):
        """
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

    def convert2relative_time(self):
        """
        Starting time of specific event becomes its Delay ,
        stoptime of specific event becomes its duration.
        """
        copy = self.copy()
        if self.time_measure == "absolute":
            copy.delay = copy.delay.convert2relative()
            copy.dur = type(copy.dur)(
                dur - delay for dur, delay in zip(self.dur, self.delay))
            copy.delay.append(copy.dur[-1])
            copy._time_measure = "relative"
        return copy


class Melody(AbstractLine):
    """
    A Melody contains sequentially played Pitches.
    """
    _obj_class = Tone
    _sub_sequences_class = (mel.Mel, rhy.RhyCompound, rhy.RhyCompound)

    @classmethod
    def subvert_object(cls, tone: Tone) -> Union[
            Tuple[AbstractPitch, rhy.RhyUnit, rhy.RhyUnit],
            Tuple[None, rhy.RhyUnit, rhy.RhyUnit]]:
        return tone.pitch, tone.delay, tone.duration

    @property
    def freq(self) -> Tuple[float]:
        return self.mel.freq

    @property
    def duration(self):
        return time.Time(sum(element.delay for element in self))

    @property
    def mel(self):
        """
        Alias for backwards compatbility,
        """
        return self.pitch

    @mel.setter
    def mel(self, arg):
        self.pitch = arg

    @property
    def rhy(self):
        """
        Alias for backwards compatbility,
        """
        return self.delay

    @rhy.setter
    def rhy(self, arg):
        self.delay = arg

    @music21.decorator
    def convert2music21(self):
        stream = music21.m21.stream.Stream()
        for t in self:
            m21_tone = t.convert2music21()
            for sub in m21_tone:
                stream.append(sub)
        return stream

    def __hash__(self):
        return hash(tuple(hash(t) for t in self))


class JIMelody(Melody):
    _sub_sequences_class = (ji.JIMel, rhy.RhyCompound, rhy.RhyCompound)


class Cadence(AbstractLine):
    """
    A Cadence contains sequentially played Harmonies.
    """
    _obj_class = Chord
    _sub_sequences_class = (mel.Cadence, rhy.RhyCompound, rhy.RhyCompound)

    @property
    def harmony(self):
        """
        Alias for backwards compatbility,
        """
        return self.pitch

    @harmony.setter
    def harmony(self, arg):
        self.pitch = arg

    @property
    def rhy(self):
        """
        Alias for backwards compatbility,
        """
        return self.delay

    @rhy.setter
    def rhy(self, arg):
        self.delay = arg

    @classmethod
    def subvert_object(cls, chord):
        return chord.harmony, chord.delay, chord.duration

    @property
    def freq(self):
        return self.harmony.freq

    @music21.decorator
    def convert2music21(self):
        stream = music21.m21.stream.Stream()
        for c in self:
            m21_chord = c.convert2music21()
            for sub in m21_chord:
                stream.append(sub)
        return stream


class JICadence(Cadence):
    _sub_sequences_class = (ji.JICadence, rhy.RhyCompound, rhy.RhyCompound)


class PolyLine(abstract.SimultanEvent):
    """
    A Container for Melody and Cadence - Objects.
    """

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
        """
        Add Rests to all Voices, which stop earlier than the
        longest voice, so that
        sum(Polyphon[0].rhy) == sum(Polyphon[1].rhy) == ...
        """
        poly = self.copy()
        total = self.duration
        for v in poly:
            summed = sum(v.delay)
            if summed < total:
                v.append(Rest(rhy.RhyUnit(total - summed)))
        return poly

    @property
    def duration(self) -> time.Time:
        dur_sub = tuple(element.duration for element in self)
        try:
            return time.Time(max(dur_sub))
        except ValueError:
            return None

    def convert2absolute_time(self):
        """
        Delay becomes the starting time of the specific event,
        duration becomes the stoptime of the specific event.
        """
        copy = self.copy()
        if self.time_measure == "relative":
            for i, item in enumerate(copy):
                copy[i] = item.convert2absolute_time()
            copy._time_measure = "absolute"
        return copy

    def convert2relative_time(self):
        """
        Starting time of specific event becomes its Delay,
        stoptime of specific event becomes its duration.
        """
        copy = self.copy()
        if self.time_measure == "absolute":
            for i, item in enumerate(copy):
                copy[i] = item.convert2relative_time()
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
        """
        """
        converted_poly = self.convert2absolute_time()
        return self.find_simultan_events_in_absolute_polyline(
            converted_poly, polyidx, itemidx)

    def find_exact_simultan_events(self, polyidx, itemidx,
                                   convert2relative=True) -> tuple:
        """
        """
        converted_poly = self.convert2absolute_time()
        simultan = self.find_simultan_events_in_absolute_polyline(
            converted_poly, polyidx, itemidx)
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

    def cut_up_by_time(self, start: rhy.RhyUnit, stop: rhy.RhyUnit,
                       hard_cut=True, add_earlier=True) -> "PolyLine":
        """
        """
        polyline = self.convert2absolute_time()
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
                sub.delay = type(sub.delay)(
                    d - sub.delay[0] for d in sub.delay)
            polyline = polyline.convert2relative_time()
        return polyline

    def cut_up_by_idx(self, polyidx, itemidx,
                      hard_cut=True, add_earlier=True) -> "PolyLine":
        """
        """
        item = self[polyidx].convert2absolute_time()[itemidx]
        item_start = item.delay
        item_stop = item.duration
        return self.cut_up_by_time(
                item_start, item_stop, hard_cut, add_earlier)


class Polyphon(PolyLine):
    """
    Container for Melody - Objects.
    """
    @music21.decorator
    def convert2music21(self):
        score = music21.m21.stream.Score(id='mainScore')
        measures = (music21.m21.stream.Measure(
            m.convert2music21(), number=1)
            for m in self)
        parts = (music21.m21.stream.Part([measure], id='part{0}'.format(i))
                 for i, measure in enumerate(measures))
        for part in parts:
            score.append(part)
        return score

    def chordify(self, cadence_class=Cadence, harmony_class=mel.Harmony):
        """
        Similar to music21.stream.Stream.chordify() - method:
        Create a chordal reduction of polyphonic music, where each
        change to a new pitch results in a new chord.
        """
        t_set = ToneSet.from_polyphon(self)
        melody = t_set.convert2melody()
        cadence = []
        acc = 0
        for t in melody:
            if t.delay != 0:
                harmony = t_set.pop_by_time(acc).convert2melody().mel
                harmony = harmony_class(
                    (p for p in harmony if p is not None))
                new_chord = Chord(harmony, t.delay, t.duration)
                cadence.append(new_chord)
                acc += t.delay
        return cadence_class(cadence)


class Instrument:
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
        """
        return all Instruments, which could play the
        asked pitch
        """
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
                t.delay = rhy.RhyUnit(d)
                d += delay
                new_set.add(t)
        return new_set

    @classmethod
    def from_cadence(cls, cadence: Cadence) -> "ToneSet":
        new_set = cls()
        d = 0
        for chord in cadence:
            delay = float(chord.delay)
            for p in chord.harmony:
                t = Tone(p, rhy.RhyUnit(d), chord.duration)
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
            t.delay = rhy.RhyUnit(diff)
        sorted_by_delay[-1].delay = rhy.RhyUnit(sorted_by_delay[-1].duration)
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
                cadence.append(Chord(
                    harmony, rhy.RhyUnit(diff), rhy.RhyUnit(t.duration)))
                harmony = mel.Harmony([])
        cadence[-1].delay = rhy.RhyUnit(sorted_by_delay[-1].duration)
        return Cadence(cadence)
