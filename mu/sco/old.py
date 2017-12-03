from mu.sco import abstract


class Tone(abstract.UniformEvent):
    def __init__(self, pitch, duration):
        self.pitch = pitch
        self._dur = duration

    def __hash__(self):
        return hash((self.pitch, self.duration))

    def __repr__(self):
        return str((repr(self.pitch), repr(self.duration)))


class Chord(abstract.SimultanEvent):
    """A Chord contains simultanly played Tones."""
    pass


class Melody(abstract.SequentialEvent):
    """A Melody contains sequentially played Tones."""
    @property
    def freq(self):
        return tuple((t.pitch.freq for t in self))

    @property
    def dur(self):
        return tuple((t.duration for t in self))


class Cadence(abstract.SequentialEvent):
    """A Cadence contains sequentially played Chords."""
    pass
