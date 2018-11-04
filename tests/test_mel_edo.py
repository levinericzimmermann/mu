import unittest
from mu.mel import edo


class EdoTest(unittest.TestCase):
    def test_is_power(self):
        self.assertEqual(edo.EdoPitch.is_power(1, 1), True)
        self.assertEqual(edo.EdoPitch.is_power(2, 1), False)
        self.assertEqual(edo.EdoPitch.is_power(2, 0), False)
        self.assertEqual(edo.EdoPitch.is_power(4, 2), True)

    def test_mk_new_edo_class(self):
        f = 2
        steps = 12
        cp = 440
        cp_shift = 9

        class EDO2_12PitchClass(edo.EdoPitch):
            _frame = f
            _steps = steps
            _concert_pitch = cp
            _concert_pitch_shift = cp_shift

        EDO2_12PitchMethod = edo.EdoPitch.mk_new_edo_class(f, steps, cp, cp_shift)
        self.assertEqual(EDO2_12PitchMethod._frame, EDO2_12PitchClass._frame)
        self.assertEqual(EDO2_12PitchMethod._steps, EDO2_12PitchClass._steps)
        self.assertEqual(
            EDO2_12PitchMethod._concert_pitch, EDO2_12PitchClass._concert_pitch
        )
        self.assertEqual(
            EDO2_12PitchMethod._concert_pitch_shift,
            EDO2_12PitchClass._concert_pitch_shift,
        )

    def test_pitchclass(self):
        p0 = edo.EDO2_12Pitch(0, 1)
        self.assertEqual(p0.pitchclass, 0)
        # use too high number for pitchclass
        self.assertRaises(ValueError, lambda: edo.EDO2_12Pitch(13))
        # use false number for multiply
        self.assertRaises(ValueError, lambda: edo.EDO2_12Pitch(0, 3))

    def test_factor(self):
        p0 = edo.EDO2_12Pitch(0, 1)
        expected_factor = pow(2, 1 / 12)
        self.assertEqual(p0.factor, expected_factor)

    def test_cents(self):
        p0 = edo.EDO2_12Pitch(0, 1)
        p1 = edo.EDO2_12Pitch(1, 1)
        p2 = edo.EDO2_12Pitch(2, 1)
        p3 = edo.EDO2_12Pitch(2.5, 1)
        p4 = edo.EDO2_12Pitch(2.5, 2)
        self.assertEqual(p0.cents, 0)
        self.assertEqual(round(p1.cents, 3), 100)
        self.assertEqual(round(p2.cents, 3), 200)
        self.assertEqual(round(p3.cents, 3), 250)
        self.assertEqual(round(p4.cents, 3), 1450)

    def test_repr(self):
        p0 = edo.EDO2_12Pitch(0, 1)
        self.assertEqual(repr(p0), str((p0.pitchclass, p0.multiply)))


if __name__ == "__main__":
    unittest.main()
