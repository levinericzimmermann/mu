from mu.sco import abstract
from mu.rhy import rhy
from mu.mel import abstract as mel_abstract


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


class Melody(abstract.MultiSequentialEvent):
    """A Melody contains sequentially played Tones."""
    _obj_class = Tone
    _sub_sequences_class = (mel_abstract.Mel, rhy.RhyCompound)
    _sub_sequences_class_names = ("mel", "rhy")

    @classmethod
    def subvert_object(cls, tone):
        return tone.pitch, tone.duration

    @property
    def freq(self):
        return self.mel.freq

    @property
    def dur(self):
        return self.rhy


class Cadence(abstract.MultiSequentialEvent):
    """A Cadence contains sequentially played Chords."""
    _obj_class = Chord
    _sub_sequences_class = (mel_abstract.Harmony, rhy.RhyCompound)
    _sub_sequences_class_names = ("harmony", "rhy")

    @classmethod
    def subvert_object(cls, chord):
        return chord.harmony, chord.duration

    @property
    def freq(self):
        return self.mel.freq

    @property
    def dur(self):
        return self.rhy
