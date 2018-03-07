import abc
from mu.abstract import muobjects
from mu.time import time


class Event(abc.ABC):
    """
    An Event might be any Object, which contains discrete Information
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
    or type >Complex< through its 'is_uniform'-Method.
    """

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

    def __new__(cls, *args, **kwargs):
        def mk_property(num):
            def set_value(self, arg):
                self.sequences[num] = arg
            return lambda self: self.sequences[num], set_value
        for i, name in enumerate(cls._sub_sequences_class_names):
            getter, setter = mk_property(i)
            getter_name = "get_{0}".format(name)
            setter_name = "set_{0}".format(name)
            setattr(cls, getter_name, getter)
            setattr(cls, setter_name, setter)
            setattr(cls, name, property(
                getattr(cls, getter_name), getattr(cls, setter_name)))
        return ComplexEvent.__new__(cls)

    def __init__(self, iterable):
        self.sequences = type(self).subvert_iterable(iterable)

    def __len__(self):
        return len(self.mk_sequence())

    @abc.abstractclassmethod
    def subvert_object(cls, obj):
        raise NotImplementedError

    @classmethod
    def subvert_iterable(cls, iterable):
        sequences = list(c([]) for c in cls._sub_sequences_class)
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
        seq = self.mk_sequence()[idx]
        if type(idx) == slice:
            return type(self)(seq)
        else:
            return seq

    def __repr__(self):
        return repr(self.mk_sequence())

    def __iter__(self):
        return iter(self.mk_sequence())

    def __add__(self, other):
        return type(self)(self.mk_sequence() + other.mk_sequence())

    def __sub__(self, other):
        return type(self)(self.mk_sequence() - other.mk_sequence())

    def __mul__(self, fac):
        return type(self)(self.mk_sequence() * fac)

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


class SimultanEvent(ComplexEvent, muobjects.MUList):
    @property
    def duration(self):
        return time.Time(max(tuple(element.duration for element in self)))
