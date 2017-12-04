from mu.sco import abstract
from mu.rhy import rhy
from mu.mel import mel


class Tone(abstract.UniformEvent):
    def __init__(self, pitch, duration):
        self.pitch = pitch
        self._dur = duration

    def __hash__(self):
        return hash((self.pitch, self.duration))

    def __repr__(self):
        return str((repr(self.pitch), repr(self.duration)))

    def __eq__(self, other):
        return self.pitch == other.pitch and self.duration == other.duration


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
    _sub_sequences_class = (mel.Mel, rhy.RhyCompound)
    _sub_sequences_class_names = ("mel", "rhy")

    @classmethod
    def subvert_object(cls, tone):
        return tone.pitch, tone.duration

    @property
    def freq(self):
        return self.mel.freq


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
