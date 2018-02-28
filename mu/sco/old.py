# @Author: Levin Eric Zimmermann
# @Date:   2018-02-07T18:18:59+01:00
# @Email:  levin-eric.zimmermann@folkwang-uni.de
# @Project: mu
# @Last modified by:   uummoo
# @Last modified time: 2018-02-07T18:20:40+01:00


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
                 duration: Optional[rhy.RhyUnit] = None) -> None:
        if not duration:
            duration = delay
        self.pitch = pitch
        self._dur = rhy.RhyUnit(duration)
        self.delay = rhy.RhyUnit(delay)

    def __hash__(self) -> int:
        return hash((self.pitch, self.delay, self.duration))

    def __repr__(self):
        return str((repr(self.pitch), repr(self.delay), repr(self.duration)))

    def __eq__(self, other: "Tone") -> bool:
        return all((self.pitch == other.pitch, self.duration == other.duration,
                    self.delay == other.delay))

    def copy(self) -> "Tone":
        return type(self)(self.pitch, self.delay, self.duration)

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
    def __init__(self, delay: rhy.RhyUnit) -> None:
        self._dur = rhy.RhyUnit(0)
        self.delay = delay

    def __repr__(self):
        return repr(self.duration)

    @property
    def pitch(self) -> None:
        return None


class Chord(abstract.SimultanEvent):
    """A Chord contains simultanly played Tones."""

    def __init__(self, harmony, delay: rhy.RhyUnit,
                 duration: Optional[rhy.RhyUnit] = None) -> None:
        if not duration:
            duration = delay
        self.harmony = harmony
        self._dur = duration
        self.delay = delay

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


class Melody(abstract.MultiSequentialEvent):
    """A Melody contains sequentially played Pitches."""
    _obj_class = Tone
    _sub_sequences_class = (mel.Mel, rhy.RhyCompound, rhy.RhyCompound)
    _sub_sequences_class_names = ("mel", "rhy", "dur")

    @classmethod
    def subvert_object(cls, tone: Tone) -> Union[
            Tuple[AbstractPitch, rhy.RhyUnit, rhy.RhyUnit],
            Tuple[None, rhy.RhyUnit, rhy.RhyUnit]]:
        return tone.pitch, tone.delay, tone.duration

    @property
    def freq(self) -> Tuple[float, float, float]:
        return self.mel.freq

    @property
    def duration(self):
        return time.Time(sum(element.delay for element in self))

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

    def tie(self):
        def sub(melody):
            new = []
            for i, t0 in enumerate(melody):
                try:
                    t1 = melody[i + 1]
                except IndexError:
                    new.append(t0)
                    break
                if t0.duration == t0.delay and t0.pitch == t1.pitch:
                    t_new = type(t0)(t0.pitch,
                                     t0.duration + t1.delay,
                                     t0.duration + t1.duration)
                    return new + sub([t_new] + melody[i + 2:])
                else:
                    new.append(t0)
            return new
        return type(self)(sub(list(self)))


class JIMelody(Melody):
    _sub_sequences_class = (ji.JIMel, rhy.RhyCompound, rhy.RhyCompound)


class Cadence(abstract.MultiSequentialEvent):
    """A Cadence contains sequentially played Harmonies."""
    _obj_class = Chord
    _sub_sequences_class = (mel.Cadence, rhy.RhyCompound, rhy.RhyCompound)
    _sub_sequences_class_names = ("harmony", "rhy", "dur")

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


class Polyphon(abstract.SimultanEvent):
    """
    A Container for Melody - Objects.
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

    def fill(self):
        """
        Add Rests to all Voices, which stop earlier than the
        longest voice, so that
        sum(Polyphon[0].rhy) == sum(Polyphon[1].rhy) == ...
        """
        poly = self.copy()
        total = self.duration
        for v in poly:
            summed = sum(v.rhy)
            if summed < total:
                v.append(Rest(rhy.RhyUnit(total - summed)))
        return poly

    @property
    def duration(self):
        dur_sub = tuple(element.duration for element in self)
        try:
            return time.Time(max(dur_sub))
        except ValueError:
            return None

    def horizontal_add(self, other: "Polyphon", fill=True):
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
        """return all Instruments, which could play the
        asked pitch"""
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
