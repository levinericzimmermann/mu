class MUList(list):
    def __add__(self, other):
        return type(self)(list.__add__(self, other))

    def __sub__(self, other):
        return type(self)(list.__sub__(self, other))

    def __mul__(self, other):
        return type(self)(list.__mul__(self, other))

    def __getitem__(self, idx):
        if type(idx) == slice:
            return type(self)(list.__getitem__(self, idx))
        else:
            return list.__getitem__(self, idx)

    def reverse(self):
        return type(self)(reversed(self))

    def copy(self):
        return type(self)(list.copy(self))


class MUSet(set):
    def __or__(self, other):
        return type(self)(set.__or__(self, other))

    def __and__(self, other):
        return type(self)(set.__and__(self, other))

    def __sub__(self, other):
        return type(self)(set.__sub__(self, other))

    def __xor__(self, other):
        return type(self)(set.__xor__(self, other))
