from mu.abstract import muobjects


class Time(muobjects.MUFloat):
    def __init__(self, value: float):
        try:
            assert value >= 0
        except AssertionError:
            msg = "There is no negative time! {0}".format(value)
            raise ValueError(msg)
        muobjects.MUFloat.__init__(value)

    @staticmethod
    def seconds2miliseconds(s):
        return s * 1000

    @staticmethod
    def minutes2miliseconds(m):
        return m * 60 * 1000

    @staticmethod
    def hours2miliseconds(h):
        return h * 60 * 60 * 1000

    @classmethod
    def from_seconds(cls, s):
        return cls(Time.seconds2miliseconds(s))

    @classmethod
    def from_minutes(cls, m):
        return cls(Time.minutes2miliseconds(m))

    @classmethod
    def from_hours(cls, h):
        return cls(Time.hours2miliseconds(h))
