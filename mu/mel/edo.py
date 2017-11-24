from mu.mel import abstract
import math


class EdoTone(abstract.AbstractTone):
    _frame = None
    _steps = None
    _concert_pitch = None
    _concert_pitch_shift = 0
    __multiply = 1

    def __init__(self, pitch: float):
        self.pitch = pitch

    def __repr__(self):
        return str(self.pitch)

    @staticmethod
    def isPower(num, base):
        """https://stackoverflow.com/questions/15352593/how-\
        to-check-if-a-number-is-a-power-of-base-b"""
        if base == 1 and num != 1:
            return False
        if base == 1 and num == 1:
            return True
        if base == 0 and num != 1:
            return False
        power = int(math.log(num, base) + 0.5)
        if num < base:
            power -= 1
        return base ** power == num

    @property
    def multiply(self):
        return self.__multiply

    @multiply.setter
    def multiply(self, arg):
        if EdoTone.isPower(arg, self._frame):
            self.__multiply = arg
        else:
            w = "Multiply-Argument has to be a power of the frame {0}.".format(
                self._frame)
            raise ValueError(w)

    @property
    def factor(self):
        return pow(self._frame, 1 / self._steps)

    @property
    def p0(self):
        return self._concert_pitch / pow(self.factor,
                                         self._concert_pitch_shift)

    def calc(self, factor=1):
        return (self.factor ** self.pitch) * self.multiply * factor * self.p0

    @classmethod
    def mk_new_edo_class(cls, frame, steps, concert_pitch=None,
                         concert_pitch_shift=0):
        new = type("Edo_{0}/{1}_Tone".format(frame, steps), (cls,), {})
        new._frame = frame
        new._steps = steps
        new._concert_pitch = concert_pitch
        new._concert_pitch_shift = concert_pitch_shift
        return new


class EDO2_12Tone(EdoTone.mk_new_edo_class(2, 12, 440, 9)):
    pass


class EDO2_12Harmony(EDO2_12Tone.mk_iterable(abstract.AbstractHarmony)):
    def set_multiply(self, arg):
        for t in self:
            t.multiply = arg

    @property
    def freq(self):
        return sorted(self.calc())

    @classmethod
    def make_scale(cls, step, octaves=1, start=0):
        return cls(EDO2_12Tone(num) for num in range(0, 12*octaves, step))
