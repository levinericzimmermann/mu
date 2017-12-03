import abc
from mu.abstract import muobjects
from mu.time import time


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


class MultiSequentialEvent(ComplexEvent):
    _obj_class = None
    _sub_sequences_class = None
    _sub_sequences_class_names = None

    def __init__(self, iterable):
        def mk_property(num):
            def func(self):
                return self.sequences[num]
            return property(func)
        self.sequences = type(self).subvert_iterable(iterable)
        for i, name in enumerate(type(self)._sub_sequences_class_names):
            p = mk_property(i)
            setattr(self, name, p.fget(self))
            # setattr(self, name, p.fset)

    @abc.abstractclassmethod
    def subvert_object(cls, obj):
        raise NotImplementedError

    @classmethod
    def subvert_iterable(cls, iterable):
        sequences = tuple(c([]) for c in cls._sub_sequences_class)
        for obj in iterable:
            for i, unit in enumerate(cls.subvert_object(obj)):
                sequences[i].append(unit)
        return sequences

    @classmethod
    def build_objects(cls, *parameter):
        return tuple(cls._obj_class(*data) for data in zip(*parameter))

    @classmethod
    def from_parameter(cls, *parameter):
        return cls(cls.build_objects(*parameter))

    def mk_sequence(self):
        return [type(self)._obj_class(
                *data) for data in zip(*self.sequences)]

    def __getitem__(self, idx):
        return self.mk_sequence()[idx]

    def __repr__(self):
        return repr(self.mk_sequence())

    def __iter__(self):
        return iter(self.mk_sequence())

    def append(self, arg):
        data = type(self).subvert_object(arg)
        for s, el in zip(self.sequences, data):
            s.append(el)

    def insert(self, pos, arg):
        data = type(self).subvert_object(arg)
        for s, el in zip(self.sequences, data):
            s.insert(pos, el)

    def extend(self, arg):
        for el in arg:
            self.append(el)

    def reverse(self):
        return type(self)(reversed(self.mk_sequence()))

    def copy(self):
        return type(self)(self[:])

    @property
    def duration(self):
        return time.Time(sum(element.duration for element in self))

    def __eq__(self, other):
        return self.mk_sequence() == other.mk_sequence()


class SimultanEvent(ComplexEvent, muobjects.MUSet):
    @property
    def duration(self):
        return time.Time(max(tuple(element.duration for element in self)))