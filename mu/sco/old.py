try:
    import music21
except ImportError:
    music21 = False
from mu.abstract import muobjects
from mu.sco import abstract
from mu.rhy import rhy
from mu.mel import mel
from mu.mel import ji


class Tone(abstract.UniformEvent):
    def __init__(self, pitch, delay, duration=None):
        if not duration:
            duration = delay
        self.pitch = pitch
        self._dur = duration
        self.delay = delay

    def __hash__(self):
        return hash((self.pitch, self.delay, self.duration))

    def __repr__(self):
        return str((repr(self.pitch), repr(self.delay), repr(self.duration)))

    def __eq__(self, other):
        return all((self.pitch == other.pitch, self.duration == other.duration,
                   self.delay == other.delay))

    def copy(self):
        return type(self)(self.pitch, self.delay, self.duration)


class Rest(Tone):
    def __init__(self, delay):
        self._dur = 0
        self.delay = delay

    def __repr__(self):
        return repr(self.duration)

    @property
    def pitch(self):
        return None


class Chord(abstract.SimultanEvent):
    """A Chord contains simultanly played Tones."""
    def __init__(self, harmony, duration):
        self.harmony = harmony
        self._dur = duration

    @property
    def duration(self):
        return self._dur


class Melody(abstract.MultiSequentialEvent):
    """A Melody contains sequentially played Pitches."""
    _obj_class = Tone
    _sub_sequences_class = (mel.Mel, rhy.RhyCompound, rhy.RhyCompound)
    _sub_sequences_class_names = ("mel", "rhy", "dur")

    @classmethod
    def subvert_object(cls, tone):
        return tone.pitch, tone.delay, tone.duration

    @property
    def freq(self):
        return self.mel.freq

    def convert2music21(self):
        pass


class JIMelody(Melody):
    _sub_sequences_class = (ji.JIMel, rhy.RhyCompound, rhy.RhyCompound)


class Cadence(abstract.MultiSequentialEvent):
    """A Cadence contains sequentially played Harmonies."""
    _obj_class = Chord
    _sub_sequences_class = (mel.Harmony, rhy.RhyCompound)
    _sub_sequences_class_names = ("harmony", "rhy")

    @classmethod
    def subvert_object(cls, chord):
        return chord.harmony, chord.duration

    @property
    def freq(self):
        return self.mel.freq


class Polyphon(abstract.SimultanEvent):
    pass


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
    def from_melody(cls, melody):
        new_set = cls()
        d = 0
        for t in melody.copy():
            delay = float(t.delay)
            t.delay = rhy.RhyUnit(d)
            d += delay
            new_set.add(t)
        return new_set

    def pop_by(self, test, args):
        new_set = ToneSet()
        for arg in args:
            for t in self:
                if test(t, arg):
                    new_set.add(t)
        return new_set

    def pop_by_pitch(self, *pitch):
        return self.pop_by(lambda t, p: t.pitch == p, pitch)

    def pop_by_duration(self, *duration):
        return self.pop_by(lambda t, d: t.duration == d, duration)

    def pop_by_start(self, *start):
        return self.pop_by(lambda t, s: t.delay == s, start)

    def convert2melody(self):
        sorted_by_delay = sorted(list(self.copy()), key=lambda t: t.delay)
        first = sorted_by_delay[0].delay
        if first != 0:
            sorted_by_delay.insert(0, Rest(0))
        for t, t_after in zip(sorted_by_delay, sorted_by_delay[1:]):
            diff = t_after.delay - t.delay
            t.delay = rhy.RhyUnit(diff)
        sorted_by_delay[-1].delay = rhy.RhyUnit(sorted_by_delay[-1].duration)
        return Melody(sorted_by_delay)
