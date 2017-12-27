from mu.abstract import mutate


class MUList(mutate.mutate_class(list)):
    def reverse(self) -> "MUList":
        return type(self)(reversed(self))


class MUSet(mutate.mutate_class(set)):
    def copy(self) -> "MUSet":
        return type(self)([t.copy() for t in self])


MUFloat = mutate.mutate_class(float)


MUDict = mutate.mutate_class(dict)
