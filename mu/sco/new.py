from mu.rhy import rhy
from mu.sco import abstract


"""This module represents musical structures that are based on continuous textures."""


class ControlChange(abstract.UniformEvent):
    def __init__(self, value, delay: rhy.RhyUnit) -> None:
        if isinstance(delay, rhy.RhyUnit) is False:
            delay = rhy.RhyUnit(delay)

        self.value = value
        self.delay = delay

    def __repr__(self) -> str:
        return str(("CC: ", repr(self.value), repr(self.delay)))

    def duration(self) -> rhy.RhyUnit:
        return self.delay
