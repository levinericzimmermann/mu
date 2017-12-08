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
        self.assertEqual(m0.lv, 1)
        self.assertEqual(m1.lv, 2)
        self.assertEqual(m2.lv, 1)
        self.assertEqual(m3.lv, 3)
        self.assertEqual(m4.lv, 1)

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
