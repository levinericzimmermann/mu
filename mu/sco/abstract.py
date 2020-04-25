import abc
import types

from mu.abstract import muobjects
from mu.time import time


class Event(abc.ABC):
    """Event-Class.

    An Event may be any object that contains discrete Information
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
    _object = None

    # TODO(remove bug when melody[0].delay = 100, melody.delay[0] != 100 -> you have to
    # update linked list before calling them)

    def __new__(cls, *args, **kwargs):
        def mk_property(attribute: str):
            def get_value(self) -> list:

                for n in range(len(self)):
                    self.__update_nth_index_of_linked_lists(n)

                return getattr(self, cls.__format_hidden_attribute(attribute))

            def set_value(self, arg) -> None:
                arg = _LinkedList(self, attribute, arg[:])
                setattr(self, cls.__format_hidden_attribute(attribute), arg)

                for item, value in zip(self, arg):
                    setattr(item, attribute, type(getattr(item, attribute))(value))

            return get_value, set_value

        for name in cls.__find_attributes():
            getter, setter = mk_property(name)
            getter_name = "__get_{}__".format(name)
            setter_name = "__set_{}__".format(name)
            setattr(cls, getter_name, getter)
            setattr(cls, setter_name, setter)
            if name != "duration":
                setattr(
                    cls,
                    name,
                    property(getattr(cls, getter_name), getattr(cls, setter_name)),
                )

        return ComplexEvent.__new__(cls)

    def __init__(self, iterable: list):
        self.__iterable = list(iterable)
        self.__attributes = self.__find_attributes()
        for attribute in self.__attributes:
            data = [getattr(item, attribute) for item in self]
            linked_list = _LinkedList(self, attribute, data)
            setattr(self, self.__format_hidden_attribute(attribute), linked_list)

    @classmethod
    def from_parameter(cls, *parameter):
        obj_class = type(cls._object)
        return cls([obj_class(*par) for par in zip(*parameter)])

    def __repr__(self) -> str:
        return repr(self.__iterable)

    def __str__(self) -> str:
        return str(self.__iterable)

    def __eq__(self, other) -> bool:
        return self[:] == other[:]

    def __len__(self) -> int:
        return len(self.__iterable)

    def reverse(self) -> "MultiSequentialEvent":
        return type(self)(list(reversed(self.__iterable)))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx == slice(None):
                return list(self.__iterable)

            else:
                return type(self)(self.__iterable[idx])

        else:
            return self.__iterable[idx]

    def __setitem__(self, idx, item) -> None:
        # set compound list
        self.__iterable[idx] = item

        # set seperated lists
        if isinstance(idx, slice):
            indices = tuple(range(*idx.indices(len(self))))
        else:
            indices = (idx,)

        for idx in indices:
            self.__update_nth_index_of_linked_lists(idx)

    def __update_nth_index_of_linked_lists(self, idx: int) -> None:
        for attribute in self.__attributes:
            formated_attribute = self.__format_hidden_attribute(attribute)
            getattr(self, formated_attribute)[idx] = getattr(self[idx], attribute)

    def __iter__(self) -> iter:
        return iter(self.__iterable)

    def append(self, item) -> None:
        self.__iterable.append(item)
        for attribute in self.__attributes:
            formated_attribute = self.__format_hidden_attribute(attribute)
            getattr(self, formated_attribute)._append(getattr(item, attribute))

    def extend(self, iterable: tuple) -> None:
        for item in iterable:
            self.append(item)

    def insert(self, idx: int, item) -> None:
        self.__iterable.insert(idx, item)
        for attribute in self.__attributes:
            formated_attribute = self.__format_hidden_attribute(attribute)
            getattr(self, formated_attribute)._insert(idx, getattr(item, attribute))

    def copy(self) -> "MultiSequentialEvent":
        return type(self)(tuple(item.copy() for item in self))

    def __add__(self, other) -> "MultiSequentialEvent":
        return type(self)(self[:] + other[:])

    def __mul__(self, factor: int) -> "MultiSequentialEvent":
        return type(self)(tuple(item.copy() for item in (self.__iterable * factor)))

    @classmethod
    def __find_attributes(cls) -> tuple:
        return tuple(
            attribute
            for attribute in dir(cls._object)
            # no private attributes
            if attribute[0] != "_"
            # no methods
            and not isinstance(getattr(cls._object, attribute), types.MethodType)
        )

    @staticmethod
    def __format_hidden_attribute(attribute: str) -> str:
        return "__{}".format(attribute)


class _LinkedList(object):
    """Private helper class for making MultiSequentialEvent."""

    def __init__(
        self, linked_object: MultiSequentialEvent, attribute: str, iterable: tuple
    ):
        self.__attribute = attribute
        self.__iterable = list(iterable)
        self.__linked_object = linked_object

    def __str__(self) -> str:
        return str(self.__iterable)

    def __eq__(self, other) -> bool:
        return self[:] == other[:]

    def __repr__(self) -> str:
        return "LinkedList({})".format(repr(self.__iterable))

    def __len__(self) -> int:
        return len(self.__iterable)

    def __sum__(self):
        return sum(self[:])

    def __getitem__(self, idx):
        return self.__iterable[idx]

    def _insert(self, idx: int, arg):
        self.__iterable.insert(idx, arg)

    def _append(self, arg):
        self.__iterable.append(arg)

    def __setitem__(self, idx, arg):
        self.__iterable[idx] = arg

        if isinstance(idx, slice):
            indices = tuple(range(*idx.indices(len(self))))
        else:
            indices = (idx,)
            arg = (arg,)

        for idx, argument in zip(indices, arg):
            # special case attribute 'duration' which always has to be overwritten
            if self.__attribute == "duration":
                attribute = "dur"
            else:
                attribute = self.__attribute

            try:
                setattr(self.__linked_object[idx], attribute, argument)

            except AttributeError:
                # if it can't be set, just ignore it
                pass

    def __iter__(self) -> iter:
        return iter(self.__iterable)


class SimultanEvent(ComplexEvent, muobjects.MUList):
    @property
    def duration(self):
        return time.Time(max(tuple(element.duration for element in self)))
