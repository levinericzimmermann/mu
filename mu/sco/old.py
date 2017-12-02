from mu.sco import abstract


class Tone(abstract.UniformEvent):
    def __init__(self, pitch, duration):
        self.pitch = pitch
        self.duration = duration


class Chord(abstract.ComplexEvent, list):
    pass


class Melody(abstract.ComplexEvent, list):
    pass


class Cadence(abstract.ComplexEvent, list):
    pass
