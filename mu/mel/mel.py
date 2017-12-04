from mu.abstract import muobjects


class Mel(muobjects.MUList):
    def __init__(self, iterable, multiply=260):
        muobjects.MUList.__init__(self, iterable)
        self.multiply = multiply

    def __hash__(self):
        return hash(tuple(hash(t) for t in self))

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()


class Harmony(muobjects.MUSet):
    def __hash__(self):
        return hash(tuple(hash(t) for t in self))

    def sorted(self):
        return sorted(self.calc())

    def calc(self, factor=1) -> tuple:
        return tuple(t.calc(self.multiply * factor) for t in self)

    @property
    def freq(self) -> tuple:
        return self.calc()


class Cadence(muobjects.MUList):
    def __hash__(self):
        return hash(tuple(hash(h) for h in self))

    def calc(self, factor=1) -> tuple:
        return tuple(h.calc(self.multiply * factor) for h in self)

    @property
    def freq(self) -> tuple:
        return self.calc()
