from mu.abstract import mutate


class MUList(mutate.mutate_class(list)):
    def reverse(self):
        return type(self)(reversed(self))


MUSet = mutate.mutate_class(set)


MUFloat = mutate.mutate_class(float)
