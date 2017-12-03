import abc
from mu.abstract import muobjects


class Event(abc.ABC):
    """An Event might be any Object, which contains discrete Information
    about Time and at least one other mu - Object, e.g.
    a Melody might be an Event, since a Melody contains Information
    about Pitch and Duration, while a Harmony isn't an Event, since it could
    only contain Pitch-Objects. On the other hand, a Chord could be
    an Event Object, since it wouldn't contain only Information about
    Pitch, but also about its Duration, Volume, e.g. a Chord might
    contain Tone-Objects.

    There are two different Types of Event-Objects:
        a) Objects of type >Uniform< don't contain other Event-Objects.
        b) Objects of type >Complex< contain other Event-Objects.

    It is possible to ask an Event-Object, whether it is type >Uniform<
    or type >Complex< through its 'is_uniform'-Method."""

    @abc.abstractclassmethod
    def is_uniform(self):
        raise NotImplementedError

    @abc.abstractproperty
    def duration(self):
        raise NotImplementedError


class UniformEvent(Event):
    """Event-Object, which doesn't contain other Event-Objects."""
    @classmethod
    def is_uniform(cls):
        return True

    @property
    def duration(self):
        return self._dur

    @duration.setter
    def duration(self, dur):
        self._dur = dur


class ComplexEvent(Event):
    """Event-Object, which might contain other Event-Objects."""
    @classmethod
    def is_uniform(cls):
        return False

    def __hash__(self):
        return hash(tuple(hash(t) for t in self))


class SequentialEvent(ComplexEvent, muobjects.MUList):
    @property
    def duration(self):
        return sum(element.duration for element in self)


class SimultanEvent(ComplexEvent, muobjects.MUSet):
    @property
    def duration(self):
        return max(tuple(element.duration for element in self))
