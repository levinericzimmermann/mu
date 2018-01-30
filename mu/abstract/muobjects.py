from mu.abstract import mutate
import orderedset


class MUList(mutate.mutate_class(list)):
    def reverse(self) -> "MUList":
        return type(self)(reversed(self))

    def sort(self) -> "MUList":
        return type(self)(sorted(self))


class MUTuple(mutate.mutate_class(tuple)):
    def reverse(self) -> "MUTuple":
        return type(self)(reversed(self))

    def sort(self) -> "MUList":
        return type(self)(sorted(self))


class MUSet(mutate.mutate_class(set)):
    def copy(self) -> "MUSet":
        return type(self)([t.copy() for t in self])


MUInt = mutate.mutate_class(int)


MUFloat = mutate.mutate_class(float)


MUDict = mutate.mutate_class(dict)


_mutated_OrderedSet = mutate.mutate_class(orderedset.OrderedSet)


class MUOrderedSet(_mutated_OrderedSet):
    def __getitem__(self, idx):
        if type(idx) == slice:
            return orderedset.OrderedSet.__getitem__(self, idx)
        else:
            return super(_mutated_OrderedSet, self).__getitem__(idx)
