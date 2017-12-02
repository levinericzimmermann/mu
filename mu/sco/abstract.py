import abc


class Event(abc.ABC):
    """An Event might be any Object, which contains Information
    about two different mu - Objects, e.g. a Melody might be an Event,
    since a Melody contains Information about Pitch and Duration,
    while a Harmony isn't an Event, since it could
    only contain Pitch-Objects. On the other hand, a Chord could be
    an Event Object, since it wouldn't contain only Information about
    Pitch, but also about its Duration, Volume, e.g. a Chord might
    contain Tone-Objects.

    There are two different Types of Event-Objects:
        a) Objects of type >Complex< contains other Event-Objects.
        b) Objects of type >Uniform< don't contains non-Event-Objects.

    It is possible to ask an Event-Object, whether it is Type >Complex< or
    >Uniform< through its 'is_uniform'-Method."""

    @abc.abstractmethod
    def is_uniform(self):
        raise NotImplementedError


class UniformEvent(Event):
    def is_uniform(self):
        return True


class ComplexEvent(Event):
    def is_uniform(self):
        return False
