import abjad
from abc import ABC, abstractmethod


class AbstractAPI(ABC):
    @staticmethod
    def copy_notes():
        pass

    @abstractmethod
    def to_abjad(self):
        raise NotImplementedError


class Note(AbstractAPI):
    def __init__(self, m=None, dur=None):
        self.m = m
        self.dur = dur

    @staticmethod
    def dur2assignable(dur):
        def seperate(num):
            a = num // 2
            return a, num % a + a
        if dur.is_assignable:
            return tuple((dur,))
        else:
            a, b = seperate(dur.numerator)
            a = Note.dur2assignable(abjad.Duration(a, dur.denominator))
            b = Note.dur2assignable(abjad.Duration(b, dur.denominator))
            return tuple(sorted(list(a + b), reverse=True))

    def to_abjad(self):
        pitch = abjad.NamedPitch.from_hertz(self.m.calc()[0])
        return tuple(abjad.Note(pitch, d) for d in Note.dur2assignable(self.dur))


class Melody(list):
    """implemented per abjad.Voice"""

    @staticmethod
    def mk_voice():
        voice = abjad.Voice()
        voice.remove_commands.append("Note_heads_engraver")
        voice.consists_commands.append("Completion_heads_engraver")
        voice.remove_commands.append("Rest_engraver")
        voice.consists_commands.append("Completion_rest_engraver")
        return voice

    def to_abjad(self):
        voice = Melody.mk_voice()
        for note in (n.to_abjad() for n in self):
            for counter, subnote in enumerate(note):
                voice.append(subnote)
            if counter:
                abjad.attach(abjad.Tie(), voice[len(voice) - counter - 1:])
        return voice


class Score(list):
    def to_abjad(self):
        pass
