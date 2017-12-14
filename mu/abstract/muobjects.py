from mu.abstract import mutate


class MUList(mutate.mutate_class(list)):
    def reverse(self):
        return type(self)(reversed(self))


class MUSet(mutate.mutate_class(set)):
    def copy(self):
        return type(self)([t.copy() for t in self])


MUFloat = mutate.mutate_class(float)
