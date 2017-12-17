import unittest
from mu.mel import ji
from fractions import Fraction


class MonzoTest(unittest.TestCase):
    def test_gcd(self):
        self.assertEqual(ji.Monzo.gcd(2, 4), 2)
        self.assertEqual(ji.Monzo.gcd(3, 9), 3)
        self.assertEqual(ji.Monzo.gcd(1, 7), 1)
        self.assertEqual(ji.Monzo.gcd(4, 16), 4)
        self.assertEqual(ji.Monzo.gcd(4, 16, 2), 2)
        self.assertEqual(ji.Monzo.gcd(4, 16, 1), 1)
        self.assertEqual(ji.Monzo.gcd(8, 96, 400), 8)
        self.assertEqual(ji.Monzo.gcd(8, 96, 400, 6), 2)

    def test_ratio(self):
        m0 = ji.Monzo([0, 1], 2)
        m1 = ji.Monzo([0, 0, -1], 2)
        m2 = ji.Monzo([2, 0, -1], 2)
        m3 = ji.Monzo([2], 2)
        self.assertEqual(m0.ratio, Fraction(5, 4))
        self.assertEqual(m1.ratio, Fraction(8, 7))
        self.assertEqual(m2.ratio, Fraction(9, 7))
        self.assertEqual(m3.ratio, Fraction(9, 8))

    def test__vector(self):
        m0 = ji.Monzo([0, 1], 1)
        m1 = ji.Monzo([0, 0, -1], 2)
        m2 = ji.Monzo([2, 0, -1], 3)
        m3 = ji.Monzo([2], 5)
        self.assertEqual(m0._vector, (0, 1))
        self.assertEqual(m1._vector, (0, 0, 0, -1))
        self.assertEqual(m2._vector, (0, 0, 2, 0, -1))
        self.assertEqual(m3._vector, (0, 0, 0, 2))

    def test__vec(self):
        m0 = ji.Monzo([0, 1], 1)
        m1 = ji.Monzo([0, 0, -1], 2)
        m2 = ji.Monzo([2, 0, -1], 3)
        m3 = ji.Monzo([2], 5)
        self.assertEqual(m0._vec, m0._vector[m0._val_shift:])
        self.assertEqual(m1._vec, m1._vector[m1._val_shift:])
        self.assertEqual(m2._vec, m2._vector[m2._val_shift:])
        self.assertEqual(m3._vec, m3._vector[m3._val_shift:])

    def test_val_border(self):
        m0 = ji.Monzo([-1, 1], 1)
        m1 = ji.Monzo([0, 1], 2)
        m2 = ji.Monzo([1], 7)
        self.assertEqual(m0.ratio, Fraction(3, 2))
        self.assertEqual(m1.ratio, Fraction(5, 4))
        self.assertEqual(m2.ratio, Fraction(11, 7))
        m0.val_border = 2
        self.assertEqual(m0.val_border, 2)
        self.assertEqual(m0._val_shift, 1)
        m0.val_border = 3
        self.assertEqual(m0.val_border, 3)
        self.assertEqual(m0._val_shift, 2)
        m0.val_border = 5
        self.assertEqual(m0.val_border, 5)
        self.assertEqual(m0._val_shift, 3)

    def test_val(self):
        m0 = ji.Monzo([-1, 1], 1)
        m1 = ji.Monzo([-1, 1], 2)
        m2 = ji.Monzo([-1, 1, 0, 0, 1], 3)
        self.assertEqual(m0.val, (2, 3))
        self.assertEqual(m1.val, (3, 5))
        self.assertEqual(m2.val, (5, 7, 11, 13, 17))

    def test_lv(self):
        m0 = ji.Monzo([1], 2)
        m1 = ji.Monzo([2], 2)
        m2 = ji.Monzo([1, -1], 2)
        m3 = ji.Monzo([3, -3], 2)
        m4 = ji.Monzo([3, -3, 1], 2)
        m5 = ji.Monzo([0], 2)
        self.assertEqual(m0.lv, 1)
        self.assertEqual(m1.lv, 2)
        self.assertEqual(m2.lv, 1)
        self.assertEqual(m3.lv, 3)
        self.assertEqual(m4.lv, 1)
        self.assertEqual(m5.lv, 1)

    def test_identity(self):
        m0 = ji.Monzo([1], 2)
        m1 = ji.Monzo([2], 2)
        m2 = ji.Monzo([1, -1], 2)
        m3 = ji.Monzo([3, -3], 2)
        m4 = ji.Monzo([3, -3, 1], 2)
        self.assertEqual(m0.identity.ratio, m0.ratio)
        self.assertEqual(m1.identity, m0)
        self.assertEqual(m2.identity, m2)
        self.assertEqual(m3.identity, m2)
        self.assertEqual(m4.identity, m4)

    def test_past(self):
        m0 = ji.Monzo([0], 2)
        m1 = ji.Monzo([1], 2)
        m2 = ji.Monzo([2], 2)
        m3 = ji.Monzo([3], 2)
        m4 = ji.Monzo([1, -1], 2)
        m5 = ji.Monzo([2, -2], 2)
        m6 = ji.Monzo([3, -3], 2)
        self.assertEqual(m3.past, (m0, m1, m2))
        self.assertEqual(m6.past, (m0, m4, m5))

    def test_adjust_ratio(self):
        self.assertEqual(ji.Monzo.adjust_ratio(
            Fraction(9, 1), 1), Fraction(9, 1))
        self.assertEqual(ji.Monzo.adjust_ratio(
            Fraction(9, 4), 2), Fraction(9, 8))
        self.assertEqual(ji.Monzo.adjust_ratio(
            Fraction(9, 16), 2), Fraction(9, 8))
        self.assertEqual(ji.Monzo.adjust_ratio(
            Fraction(15, 7), 2), Fraction(15, 14))
        self.assertEqual(ji.Monzo.adjust_ratio(
            Fraction(15, 7), 3), Fraction(15, 7))

    def test_ratio2monzo(self):
        self.assertEqual(ji.Monzo.ratio2monzo(
            Fraction(4, 3)), ji.Monzo((2, -1,)))
        self.assertEqual(ji.Monzo.ratio2monzo(
            Fraction(9, 8)), ji.Monzo((-3, 2)))
        self.assertEqual(ji.Monzo.ratio2monzo(
            Fraction(9, 5)), ji.Monzo((0, 2, -1)))

    def test_math(self):
        m0 = ji.Monzo([0, -1])
        m1 = ji.Monzo([0, 0, 1])
        m2 = ji.Monzo([0, 2, 1])
        m3 = ji.Monzo([0, 1, 2])
        m4 = ji.Monzo([0, 0, 2])
        m5 = ji.Monzo([0, 0, 3])
        m6 = ji.Monzo([2, 2, 2])
        m7 = ji.Monzo([3, 3, 3])
        self.assertEqual(m0 + m1 + m2, m3)
        self.assertEqual(m3 - m0 - m1, m2)
        self.assertEqual(m1 * m6, m4)
        self.assertEqual(m1 * m7, m5)

    def test_sum(self):
        m0 = ji.Monzo([0, -1, 1, 3, 2, -3])
        self.assertEqual(m0.summed(), 10)

    def test_scalar(self):
        m0 = ji.Monzo([0, -1, 1])
        m1 = ji.Monzo([0, -2, 2])
        self.assertEqual(m0.scalar(2), m1)

    def test_dot(self):
        m0 = ji.Monzo([0, -1, 1])
        m1 = ji.Monzo([0, -2, 2])
        self.assertEqual(m0.dot(m0), 2)
        self.assertEqual(m0.dot(m1), 4)
        self.assertEqual(m1.dot(m1), 8)

    def test_matrix(self):
        m0 = ji.Monzo([0, -1, 1])
        m1 = ji.Monzo([0, -2, 2])
        m2 = ji.Monzo([0, 0, 0])
        m3 = ji.Monzo([0, 1, -1])
        m4 = ji.Monzo([0, -1, 1])
        m5 = ji.Monzo([0, 2, -2])
        m6 = ji.Monzo([0, -2, 2])
        m7 = ji.Monzo([0, 4, -4])
        m8 = ji.Monzo([0, -4, 4])
        self.assertEqual(m0.matrix(m0), (m2, m3, m4) * 2)
        self.assertEqual(m0.matrix(m1), (m2, m5, m6) * 2)
        self.assertEqual(m1.matrix(m1), (m2, m7, m8) * 2)

    def test_float(self):
        m0 = ji.Monzo((-1, 1))
        m1 = ji.Monzo((0, 1), 2)
        self.assertEqual(m0.float, 1.5)
        self.assertEqual(m1.float, 1.25)

    def test_inverse(self):
        m0 = ji.Monzo([0, -1, 1, 3, 2, -3])
        m1 = ji.Monzo([0, 1, -1, -3, -2, 3])
        self.assertEqual(m0.inverse(), m1)

    def test_shift_staticmethod(self):
        m0 = ji.Monzo([0, 1, 2])
        m1 = ji.Monzo([0, 0, 0, 1, 2])
        self.assertEqual(ji.Monzo(ji.Monzo._shift_vector(m0, 2)), m1)

    def test_shifted(self):
        m0 = ji.Monzo([0, 1, 2])
        m1 = ji.Monzo([0, 0, 0, 1, 2])
        m2 = ji.Monzo([2])
        self.assertEqual(m0.shift(2), m1)
        self.assertEqual(m0.shift(-2), m2)

    def test_subvert(self):
        m0 = ji.Monzo([0, 0, 1, 2, -2])
        ls = [ji.Monzo([0, 0, 1]), ji.Monzo([0, 0, 0, 1]),
              ji.Monzo([0, 0, 0, 1]), ji.Monzo([0, 0, 0, 0, -1]),
              ji.Monzo([0, 0, 0, 0, -1])]
        self.assertEqual(m0.subvert(), ls)

    def test_copy(self):
        m0 = ji.Monzo([0, 0, 1, 2, -2])
        m1 = ji.Monzo([0, 0, 1, 2, -2], 2)
        m2 = ji.Monzo([0, 0, 1, 2, -2], 5)
        self.assertEqual(m0.copy(), m0)
        self.assertEqual(m1.copy(), m1)
        self.assertEqual(m2.copy(), m2)

    def test_gender(self):
        m0 = ji.Monzo([0, 1])
        m1 = ji.Monzo([0, -1])
        m2 = ji.Monzo([0, 1, -1])
        m3 = ji.Monzo([0, -2, 0, 0, 1])
        m4 = ji.Monzo([0, 0])
        self.assertEqual(m0.gender, True)
        self.assertEqual(m1.gender, False)
        self.assertEqual(m2.gender, False)
        self.assertEqual(m3.gender, True)
        self.assertEqual(m4.gender, True)

    def test_harmonic(self):
        m0 = ji.Monzo([1], 2)
        m1 = ji.Monzo([0, 1], 2)
        m2 = ji.Monzo([0, 0, -1, 1], 2)
        m3 = ji.Monzo([-1], 2)
        m4 = ji.Monzo([0, 0, -1], 2)
        m5 = ji.Monzo([])
        self.assertEqual(m0.harmonic, 3)
        self.assertEqual(m1.harmonic, 5)
        self.assertEqual(m2.harmonic, 0)
        self.assertEqual(m3.harmonic, -3)
        self.assertEqual(m4.harmonic, -7)
        self.assertEqual(m5.harmonic, 1)

    def test_primes(self):
        m0 = ji.Monzo([1], 2)
        m1 = ji.Monzo([-2, 1], 2)
        self.assertEqual(m0.primes, (3,))
        self.assertEqual(m1.primes, (3, 5))

    def test_quantity(self):
        m0 = ji.Monzo([1], 2)
        m1 = ji.Monzo([-2, 1], 2)
        m2 = ji.Monzo([-2, 1, 1, 1], 2)
        m3 = ji.Monzo([-2, 1, 1, 1, -4, -2, 1], 2)
        self.assertEqual(m0.quantity, 1)
        self.assertEqual(m1.quantity, 2)
        self.assertEqual(m2.quantity, 4)
        self.assertEqual(m3.quantity, 7)

    def test_components(self):
        m0 = ji.Monzo([1, -1, 2, 0, 1], 1)
        m0_0 = ji.Monzo([1], 1)
        m0_1 = ji.Monzo([0, -1], 1)
        m0_2 = ji.Monzo([0, 0, 2], 1)
        m0_3 = ji.Monzo([0, 0, 0, 0, 1], 1)
        m1 = ji.Monzo([-2, 1], 2)
        m1_0 = ji.Monzo([-2], 2)
        m1_1 = ji.Monzo([0, 1], 2)
        self.assertEqual(m0.components, (m0_0, m0_1, m0_2, m0_3))
        self.assertEqual(m1.components, (m1_0, m1_1))

    def test_relation(self):
        m0 = ji.Monzo([1], 1)
        m1 = ji.Monzo([-2, 1], 1)
        m2 = ji.Monzo([0, 1], 1)
        self.assertEqual(m0.is_related(m0), True)
        self.assertEqual(m0.is_related(m1), True)
        self.assertEqual(m0.is_related(m2), False)

    def test_congeneric(self):
        m0 = ji.Monzo([1], 1)
        m1 = ji.Monzo([-2, 1], 1)
        m2 = ji.Monzo([0, 1], 1)
        m3 = ji.Monzo([-1], 1)
        self.assertEqual(m0.is_congeneric(m0), True)
        self.assertEqual(m0.is_congeneric(m1), False)
        self.assertEqual(m0.is_congeneric(m2), False)
        self.assertEqual(m0.is_congeneric(m3), True)

    def test_inheritance(self):
        for method in dir(ji.Monzo):
            self.assertIn(method, dir(ji.Monzo))
        self.assertIsInstance(ji.Monzo([]), ji.Monzo)

    def test_root(self):
        m0 = ji.Monzo([0, 0, 1])
        m1 = ji.Monzo([0, -1, 1])
        m2 = ji.Monzo([1])
        m3 = ji.Monzo([1], val_border=2)
        m4 = ji.Monzo([0], val_border=2)
        m5 = ji.Monzo([1, 0, 0], val_border=1)
        self.assertFalse(m0.is_root)
        self.assertFalse(m1.is_root)
        self.assertTrue(m2.is_root)
        self.assertFalse(m3.is_root)
        self.assertTrue(m4.is_root)
        self.assertTrue(m5.is_root)

    def test_virtual_root(self):
        p0 = ji.r(7, 5)
        p0_vr = ji.r(1, 5)
        p1 = ji.r(13, 11)
        p1_vr = ji.r(1, 11)
        p2 = ji.r(11, 13)
        p2_vr = ji.r(1, 13)
        self.assertEqual(p0.virtual_root, p0_vr)
        self.assertEqual(p1.virtual_root, p1_vr)
        self.assertEqual(p2.virtual_root, p2_vr)

    def test_abs(self):
        p0 = ji.JIPitch((1, -1), 2)
        p1 = ji.JIPitch((2, -2), 2)
        p2 = ji.JIPitch((1, 1), 2)
        p3 = ji.JIPitch((2, 2), 2)
        self.assertEqual(abs(p0), p2)
        self.assertEqual(abs(p1), p3)


class JIPitchTest(unittest.TestCase):
    def test_calc(self):
        n0 = ji.JIPitch([-1, 1])
        n0.multiply = 200
        self.assertEqual(n0.calc(), n0.multiply * Fraction(3, 2))
        self.assertEqual(n0.freq, n0.multiply * Fraction(3, 2))

    def test_constructor(self):
        self.assertEqual(ji.JIPitch.from_ratio(3, 2), ji.JIPitch([-1, 1]))
        self.assertEqual(ji.JIPitch.from_ratio(3, 2), ji.JIPitch([-1, 1]))
        self.assertEqual(ji.JIPitch.from_ratio(1, 1), ji.JIPitch((0,)))
        self.assertEqual(ji.JIPitch.from_ratio(2, 1), ji.JIPitch((1,)))
        self.assertEqual(ji.JIPitch.from_ratio(1, 2), ji.JIPitch((-1,)))
        self.assertEqual(ji.JIPitch.from_monzo(-1), ji.JIPitch((-1,)))

    def test_comparsion(self):
        t0 = ji.r(1, 1)
        t1 = ji.r(1, 1)
        t2 = ji.r(1, 1)
        t0.multiply = 200
        t1.multiply = 200
        t2.multiply = 220
        self.assertEqual(t0, t1)
        self.assertNotEqual(t0, t2)
        self.assertLess(t0, t2)
        self.assertGreater(t2, t1)


class JIMelTest(unittest.TestCase):
    def test_math(self):
        n0 = ji.JIPitch([0, 1])
        n1 = ji.JIPitch([0, 0, 1])
        n2 = ji.JIPitch([0, 1, 1])
        n3 = ji.JIPitch([0, 1, -1])
        n4 = ji.JIPitch([0, -1, 1])
        mel0 = ji.JIMel([n0, n1])
        mel1 = ji.JIMel([n1, n0])
        mel2 = ji.JIMel([n2, n2])
        mel3 = ji.JIMel([n3, n4])
        self.assertEqual(mel0.add(mel1), mel2)
        self.assertEqual(mel0.sub(mel1), mel3)

    def test_calc(self):
        n0 = ji.JIPitch([1], 2)
        n1 = ji.JIPitch([0, 1], 2)
        n0.multiply = 2
        m_fac = 200
        mel0 = ji.JIMel([n0, n1], m_fac)
        self.assertEqual(mel0.multiply, m_fac)
        correct = (float(m_fac * 2 * Fraction(3, 2)),
                   float(m_fac * Fraction(5, 4)))
        self.assertEqual(mel0.calc(), correct)

    def test_inheritance(self):
        t0 = ji.JIPitch([1])
        t1 = ji.JIPitch([0, 1])
        m0 = ji.JIMel([t0, t1])
        m1 = ji.JIMel([t0.inverse(), t1.inverse()])
        self.assertEqual(m0.inverse(), m1)

    def test_mk_line(self):
        test_mel0 = ji.JIMel.mk_line(ji.JIPitch((0, 1, -1)), 3)
        test_mel1 = ji.JIMel(
            (ji.JIPitch((0, 1, -1)), ji.JIPitch((0, 2, -2)),
             ji.JIPitch((0, 3, -3))))
        self.assertEqual(test_mel0, test_mel1)

    def test_mk_line_and_inverse(self):
        test_mel0 = ji.JIMel.mk_line_and_inverse(ji.JIPitch((0, 1, -1)), 3)
        test_mel1 = ji.JIMel.mk_line(ji.JIPitch((0, 1, -1)), 3)
        test_mel1 = test_mel1 + test_mel1.inverse()
        self.assertEqual(test_mel0, test_mel1)

    def test_intervals(self):
        test_mel0 = ji.JIMel(
            (ji.JIPitch((0, 1, -1)), ji.JIPitch((0, 2, -2)),
             ji.JIPitch((0, 3, -3))))
        test_mel1 = ji.JIMel(
            (ji.JIPitch((0, 1, -1)), ji.JIPitch((0, 1, -1))))
        self.assertEqual(test_mel0.intervals, test_mel1)

    def test_pitch_rate(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        t4 = ji.JIPitch((0,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t3, t4, t3, t3])
        self.assertEqual(test_mel0.pitch_rate, ((t0, 1), (t1, 1),
                                                (t2, 1), (t3, 1)))
        self.assertEqual(test_mel1.pitch_rate, ((t0, 1), (t3, 3)))
        self.assertEqual(test_mel2.pitch_rate, ((t3, 4),))
        self.assertEqual(test_mel3.pitch_rate, ((t3, 3), (t4, 1)))

    def test_pitch_rate_sorted(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        t4 = ji.JIPitch((0,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t3, t4, t3, t3])
        self.assertEqual(test_mel0.pitch_rate_sorted, ((t0, 1), (t1, 1),
                                                       (t2, 1), (t3, 1)))
        self.assertEqual(test_mel1.pitch_rate_sorted, ((t0, 1), (t3, 3)))
        self.assertEqual(test_mel2.pitch_rate_sorted, ((t3, 4),))
        self.assertEqual(test_mel3.pitch_rate_sorted, ((t4, 1), (t3, 3)))

    def test_different_pitches(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        t4 = ji.JIPitch((0,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t3, t4, t3, t3])
        self.assertEqual(test_mel0.different_pitches, (t0, t1, t2, t3))
        self.assertEqual(test_mel1.different_pitches, (t0, t3))
        self.assertEqual(test_mel2.different_pitches, (t3,))
        self.assertEqual(test_mel3.different_pitches, (t3, t4))

    def test_most_common_pitch(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        t4 = ji.JIPitch((0,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t3, t4, t3, t3])
        self.assertEqual(test_mel0.most_common_pitch, t3)
        self.assertEqual(test_mel1.most_common_pitch, t3)
        self.assertEqual(test_mel2.most_common_pitch, t3)
        self.assertEqual(test_mel3.most_common_pitch, t3)

    def test_least_common_pitch(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        t4 = ji.JIPitch((0,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t3, t4, t3, t3])
        self.assertEqual(test_mel0.least_common_pitch, t0)
        self.assertEqual(test_mel1.least_common_pitch, t0)
        self.assertEqual(test_mel2.least_common_pitch, t3)
        self.assertEqual(test_mel3.least_common_pitch, t4)

    def test_avg_gender(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        t4 = ji.JIPitch((0, -1))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t4, t4, t4, t3])
        self.assertEqual(test_mel0.avg_gender, 0)
        self.assertEqual(test_mel1.avg_gender, 0.5)
        self.assertEqual(test_mel2.avg_gender, 1)
        self.assertEqual(test_mel3.avg_gender, -0.5)

    def test_order(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((0, 2, -2))
        t2 = ji.JIPitch((0, 3, -3))
        test_mel0 = ji.JIMel([t0, t1, t2, t1])
        test_mel1 = ji.JIMel([t0, t1, t0, t2])
        test_mel2 = ji.JIMel([t0, t0, t0])
        test_mel3 = ji.JIMel([t0, t2, t0, t2])
        self.assertEqual(test_mel0.order(), 1)
        self.assertEqual(test_mel1.order(), 4 / 3)
        self.assertEqual(test_mel2.order(), 1)
        self.assertEqual(test_mel3.order(), 2)

    def test_is_ordered(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((0, 2, -2))
        t2 = ji.JIPitch((0, 3, -3))
        test_mel0 = ji.JIMel([t0, t1, t2, t1])
        test_mel1 = ji.JIMel([t0, t1, t0, t2])
        test_mel2 = ji.JIMel([t0, t0, t0])
        test_mel3 = ji.JIMel([t0, t2, t0, t2])
        self.assertEqual(test_mel0.is_ordered, True)
        self.assertEqual(test_mel1.is_ordered, False)
        self.assertEqual(test_mel2.is_ordered, True)
        self.assertEqual(test_mel3.is_ordered, False)

    def test_subvert(self):
        test_mel0 = ji.JIMel.mk_line(ji.JIPitch((0, 1, -1)), 2)
        f = ji.JIPitch((0, 0, -1))
        t = ji.JIPitch((0, 1))
        test_mel1 = ji.JIMel((t, f, t, t, f, f))
        self.assertEqual(test_mel0.subvert(), test_mel1)

    def test_accumulate(self):
        f = ji.JIPitch((0, 0, -1))
        t = ji.JIPitch((0, 1))
        test_mel0 = ji.JIMel((f, f, t, t))
        test_mel1 = ji.JIMel((f, f + f, f + f + t, f + f + t + t))
        self.assertEqual(test_mel0.accumulate(), test_mel1)

    def test_separate(self):
        test_mel0 = ji.JIMel.mk_line(ji.JIPitch((0, 1, -1)), 2)
        test_mel1 = ji.JIMel((ji.JIPitch.from_ratio(3, 5),
                              ji.JIPitch.from_ratio(9, 5),
                              ji.JIPitch.from_ratio(9, 25)))
        self.assertEqual(test_mel0.separate(), test_mel1)

    def test_dot_sum(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        test_mel0 = ji.JIMel.mk_line(t0, 2)
        test_mel1 = ji.JIMel.mk_line(t0.inverse(), 2)
        test_mel2 = ji.JIMel.mk_line_and_inverse(t0, 2)
        test_mel3 = ji.JIMel.mk_line(t0, 2) + ji.JIMel.mk_line(t1, 2)
        test_mel4 = ji.JIMel.mk_line(t0, 2) + ji.JIMel.mk_line(t2, 2)
        self.assertEqual(test_mel0.dot_sum(), 8)
        self.assertEqual(test_mel1.dot_sum(), 8)
        self.assertEqual(test_mel2.dot_sum(), -20)
        self.assertEqual(test_mel3.dot_sum(), 34)
        self.assertEqual(test_mel4.dot_sum(), 10)

    def test_count_roots(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        self.assertEqual(test_mel0.count_root(), 1)
        self.assertEqual(test_mel1.count_root(), 3)
        self.assertEqual(test_mel2.count_root(), 4)

    def test_count_repeats(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t1, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t0, t0, t1, t0])
        self.assertEqual(test_mel0.count_repeats(), 0)
        self.assertEqual(test_mel1.count_repeats(), 1)
        self.assertEqual(test_mel2.count_repeats(), 3)

    def test_count_related(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((0, -1))
        t3 = ji.JIPitch((1,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t1, t3, t1])
        test_mel2 = ji.JIMel([t3, t0, t3, t2])
        self.assertEqual(test_mel0.count_related(), 2)
        self.assertEqual(test_mel1.count_related(), 3)
        self.assertEqual(test_mel2.count_related(), 0)

    def test_count_congeneric(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((0, -1))
        t3 = ji.JIPitch((0, -1, 1))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t1])
        test_mel2 = ji.JIMel([t3, t0, t3, t3])
        self.assertEqual(test_mel0.count_congeneric(), 0)
        self.assertEqual(test_mel1.count_congeneric(), 2)
        self.assertEqual(test_mel2.count_congeneric(), 3)

    def test_count_different_pitches(self):
        t0 = ji.JIPitch((0, 1, -1))
        t1 = ji.JIPitch((-1, 1))
        t2 = ji.JIPitch((2, -1))
        t3 = ji.JIPitch((1,))
        t4 = ji.JIPitch((0,))
        test_mel0 = ji.JIMel([t0, t1, t2, t3])
        test_mel1 = ji.JIMel([t0, t3, t3, t3])
        test_mel2 = ji.JIMel([t3, t3, t3, t3])
        test_mel3 = ji.JIMel([t3, t4, t3, t3])
        self.assertEqual(test_mel0.count_different_pitches(), 4)
        self.assertEqual(test_mel1.count_different_pitches(), 2)
        self.assertEqual(test_mel2.count_different_pitches(), 1)
        self.assertEqual(test_mel3.count_different_pitches(), 2)
        test_mel3.val_border = 2
        self.assertEqual(test_mel3.count_different_pitches(), 1)

    def test_lv_difference(self):
        p0 = ji.JIPitch((1, -1), 2)
        p1 = ji.JIPitch((2, -2), 2)
        p2 = ji.JIPitch((2,), 2)
        p3 = ji.JIPitch((1,), 2)
        test_mel0 = ji.JIMel([p0, p1, p2, p3, p0])
        self.assertEqual(test_mel0.lv_difference, (1, 0, 1, 0))

    def test_dominant_prime(self):
        p0 = ji.JIPitch((1, -1), 2)
        p1 = ji.JIPitch((1,), 2)
        p2 = ji.JIPitch((0, 1), 2)
        p3 = ji.JIPitch((2,), 2)
        p4 = ji.JIPitch((-2,), 2)
        test_mel0 = ji.JIMel([p0, p1, p2, p3, p4])
        self.assertEqual(test_mel0.dominant_prime, 3)


class JIHarmonyTest(unittest.TestCase):
    def test_root(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        n4 = ji.JIPitch([-1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = h0.inverse() | h0
        h3 = h1.inverse()
        h4 = h1.inverse() | h1
        h5 = ji.JIHarmony([n1, n3])
        h6 = h5 | h5.inverse()
        self.assertEqual(h0.root, (n0,))
        self.assertEqual(h1.root, (n1,))
        self.assertEqual(h2.root, (n0,))
        self.assertEqual(h3.root, (n4,))
        self.assertEqual(h4.root, (n0,))
        self.assertEqual(h5.root, (n1, n3))
        self.assertEqual(h6.root, (n1, n1.inverse(), n3.inverse(), n3))

    def test_converted2root(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        n4 = ji.JIPitch([-1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = ji.JIHarmony([n0, n4, n3])
        self.assertEqual(h0.converted2root(), h0)
        self.assertEqual(h1.converted2root(), h2)

    def test_mk_harmonic_series(self):
        test_pitch = ji.r(11, 7)
        harmonic_series = ji.JIHarmony([ji.r(1, 7), ji.r(2, 7), ji.r(3, 7),
                                        ji.r(4, 7), ji.r(5, 7)])
        self.assertEqual(ji.JIHarmony.mk_harmonic_series(test_pitch, 6),
                         harmonic_series)


class JICadenceTest(unittest.TestCase):
    def test_identity(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = h0.inverse() | h0
        h3 = h1.inverse()
        h4 = h1.inverse() | h1
        h5 = ji.JIHarmony([n1, n3])
        h6 = h5 | h5.inverse()
        cadence0 = ji.JICadence([h0, h1, h2])
        cadence1 = ji.JICadence([h3, h4, h5, h6])
        self.assertEqual(cadence0.identity,
                         (h0.identity, h1.identity, h2.identity))
        self.assertEqual(cadence1.identity, (
                h3.identity, h4.identity, h5.identity, h6.identity))

    def test_empty_chords(self):
        n0 = ji.JIPitch([], val_border=2)
        h0 = ji.JIHarmony([n0, n0])
        h1 = ji.JIHarmony([n0])
        h2 = ji.JIHarmony([])
        cadence0 = ji.JICadence([h0, h1, h2])
        cadence1 = ji.JICadence([h2, h2, h2])
        cadence2 = ji.JICadence([h1, h1, h1])
        self.assertEqual(cadence0.empty_chords, (2,))
        self.assertEqual(cadence1.empty_chords, (0, 1, 2))
        self.assertEqual(cadence2.empty_chords, tuple([]))

    def test_has_empty_chords(self):
        n0 = ji.JIPitch([], val_border=2)
        h0 = ji.JIHarmony([n0, n0])
        h1 = ji.JIHarmony([n0])
        h2 = ji.JIHarmony([])
        cadence0 = ji.JICadence([h0, h1, h2])
        cadence1 = ji.JICadence([h2, h2, h2])
        cadence2 = ji.JICadence([h1, h1, h1])
        self.assertEqual(cadence0.has_empty_chords, True)
        self.assertEqual(cadence1.has_empty_chords, True)
        self.assertEqual(cadence2.has_empty_chords, False)


class JIModule(unittest.TestCase):
    def test_m(self):
        n0 = ji.JIPitch([-1, 1], 2)
        n0.multiply = 200
        n1 = ji.m(-1, 1, val_border=2, multiply=200)
        self.assertEqual(n0, n1)
        self.assertEqual(n1.multiply, 200)
        self.assertEqual(n1.val_border, 2)

    def test_r(self):
        n0 = ji.JIPitch([-1, 1], 2)
        n0.multiply = 200
        n1 = ji.r(5, 3, multiply=200, val_border=2)
        self.assertEqual(n0, n1)
        self.assertEqual(n1.multiply, 200)
        self.assertEqual(n1.val_border, 2)


if __name__ == "__main__":
    unittest.main()
