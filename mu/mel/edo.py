# @Author: Levin Eric Zimmermann <Levin_Eric_Zimmermann>
# @Date:   2018-01-05T23:56:21+01:00
# @Email:  levin-eric.zimmermann@folkwang-uni.de
# @Project: mu
# @Last modified by:   Levin_Eric_Zimmermann
# @Last modified time: 2018-02-04T00:52:10+01:00


from mu.mel import abstract
from mu.mel import mel
import math


class EdoPitch(abstract.AbstractPitch):
    _frame = None
    _steps = None
    _concert_pitch = None
    _concert_pitch_shift = 0
    __multiply = 1
    __pitchclass = 0

    def __init__(self, pitchclass: float, multiply=1):
        self.pitchclass = pitchclass
        self.multiply = multiply

    def __repr__(self):
        return str((self.pitchclass, self.multiply))

    @staticmethod
    def is_power(num, base):
        """
        https://stackoverflow.com/questions/15352593/how-\
        to-check-if-a-number-is-a-power-of-base-b
        """
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
    def pitchclass(self):
        return self.__pitchclass

    @pitchclass.setter
    def pitchclass(self, arg):
        if arg < self._steps and arg >= 0:
            self.__pitchclass = arg
        else:
            msg = "Pitchclass can only be between 0 - {0}".format(
                self._steps - 1)
            raise ValueError(msg)

    @property
    def multiply(self):
        return self.__multiply

    @multiply.setter
    def multiply(self, arg):
        if arg == 1 or EdoPitch.is_power(arg, self._frame):
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
        return self._concert_pitch / pow(
            self.factor, self._concert_pitch_shift)

    def calc(self, factor: float=1):
        return (self.factor ** self.pitchclass
                ) * self.multiply * self.p0 * factor

    @classmethod
    def mk_new_edo_class(cls, frame, steps, concert_pitch=None,
                         concert_pitch_shift=0):
        new = type("Edo_{0}/{1}_Pitch".format(frame, steps), (cls,), {})
        new._frame = frame
        new._steps = steps
        new._concert_pitch = concert_pitch
        new._concert_pitch_shift = concert_pitch_shift
        return new


EDO2_12Pitch = EdoPitch.mk_new_edo_class(2, 12, 440, 9)


class EDO2_12Harmony(EDO2_12Pitch.mk_iterable(mel.Harmony)):
    def set_multiply(self, arg):
        for t in self:
            t.multiply = arg

    @property
    def freq(self):
        return sorted(self.calc())

    @classmethod
    def make_scale(cls, step, octaves=1, start=1):
        return cls(EDO2_12Pitch(num, oc + 1) for num, oc in zip(
            tuple(range(0, 12, step)) * octaves, range(0, octaves)))
