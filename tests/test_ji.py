import unittest
from mu.mel import ji
from fractions import Fraction


class MonzoTest(unittest.TestCase):
    def test_prime_decomposition(self):
        self.assertEqual(ji.Monzo.decompose(45), (3, 3, 5))

    def test_ratio(self):
        m0 = ji.Monzo([0, 1], 2)
        m1 = ji.Monzo([0, 0, -1], 2)
        m2 = ji.Monzo([2, 0, -1], 2)
        m3 = ji.Monzo([2], 2)
        self.assertEqual(m0.ratio, Fraction(5, 4))
        self.assertEqual(m1.ratio, Fraction(8, 7))
        self.assertEqual(m2.ratio, Fraction(9, 7))
        self.assertEqual(m3.ratio, Fraction(9, 8))

    def test_variable_val(self):
        m0 = ji.Monzo([-1, 1], 0)
        self.assertEqual(m0.ratio, Fraction(3, 2))

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
        self.assertEqual(m0 + m1 + m2, m3)
        self.assertEqual(m3 - m0 - m1, m2)
        self.assertEqual(m1 * 2, m4)
        self.assertEqual(m1 * 3, m5)

    def test_sum(self):
        m0 = ji.Monzo([0, -1, 1, 3, 2, -3])
        self.assertEqual(m0.summed(), 10)

    def test_inverse(self):
        m0 = ji.Monzo([0, -1, 1, 3, 2, -3])
        m1 = ji.Monzo([0, 1, -1, -3, -2, 3])
        self.assertEqual(m0.inverse(), m1)

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

    def test_gender(self):
        m0 = ji.Monzo([0, 1])
        m1 = ji.Monzo([0, -1])
        m2 = ji.Monzo([0, 1, -1])
        m3 = ji.Monzo([0, -2, 0, 0, 1])
        m4 = ji.Monzo([0, 0])
        self.assertEqual(m0.gender, 1)
        self.assertEqual(m1.gender, -1)
        self.assertEqual(m2.gender, -1)
        self.assertEqual(m3.gender, 1)
        self.assertEqual(m4.gender, 0)

    def test_wilson(self):
        m0 = ji.Monzo([0, 1])
        m1 = ji.Monzo([0, 1, 1])
        m2 = ji.Monzo([0, 1, -2])
        self.assertEqual(m0.wilson, 3)
        self.assertEqual(m1.wilson, 15)
        self.assertEqual(m2.wilson, 28)

    def test_vogel(self):
        m0 = ji.Monzo([1], 2)
        m1 = ji.Monzo([1, 1], 2)
        m2 = ji.Monzo([1, -2], 2)
        self.assertEqual(m0.vogel, 4)
        self.assertEqual(m1.vogel, 18)
        self.assertEqual(m2.vogel, 32)

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


class JIToneTest(unittest.TestCase):
    def test_calc(self):
        n0 = ji.JITone([-1, 1])
        n0.multiply = 200
        self.assertEqual(n0.calc(), n0.multiply * Fraction(3, 2))
        self.assertEqual(n0.freq, n0.multiply * Fraction(3, 2))

    def test_constructor(self):
        self.assertEqual(ji.JITone.from_ratio(3, 2), ji.JITone([-1, 1]))
        self.assertEqual(ji.JITone.from_ratio(3, 2), ji.JITone([-1, 1]))
        self.assertEqual(ji.JITone.from_ratio(1, 1), ji.JITone((0,)))
        self.assertEqual(ji.JITone.from_ratio(2, 1), ji.JITone((1,)))
        self.assertEqual(ji.JITone.from_ratio(1, 2), ji.JITone((-1,)))
        self.assertEqual(ji.JITone.from_monzo(-1), ji.JITone((-1,)))


class JIMelTest(unittest.TestCase):
    def test_math(self):
        n0 = ji.JITone([0, 1])
        n1 = ji.JITone([0, 0, 1])
        n2 = ji.JITone([0, 1, 1])
        n3 = ji.JITone([0, 1, -1])
        n4 = ji.JITone([0, -1, 1])
        mel0 = ji.JIMel([n0, n1])
        mel1 = ji.JIMel([n1, n0])
        mel2 = ji.JIMel([n2, n2])
        mel3 = ji.JIMel([n3, n4])
        self.assertEqual(mel0 + mel1, mel2)
        self.assertEqual(mel0 - mel1, mel3)

    def test_calc(self):
        n0 = ji.JITone([1], 2)
        n1 = ji.JITone([0, 1], 2)
        n0.multiply = 2
        m_fac = 200
        mel0 = ji.JIMel([n0, n1], m_fac)
        self.assertEqual(mel0.multiply, m_fac)
        correct = (float(m_fac * 2 * Fraction(3, 2)),
                   float(m_fac * Fraction(5, 4)))
        self.assertEqual(mel0.calc(), correct)

    def test_inheritance(self):
        t0 = ji.JITone([1])
        t1 = ji.JITone([0, 1])
        m0 = ji.JIMel([t0, t1])
        m1 = ji.JIMel([t0.inverse(), t1.inverse()])
        self.assertEqual(m0.inverse(), m1)

    def test_mk_line(self):
        test_mel0 = ji.JIMel.mk_line(ji.JITone((0, 1, -1)), 3)
        test_mel1 = ji.JIMel(
            (ji.JITone((0, 1, -1)), ji.JITone((0, 2, -2)),
             ji.JITone((0, 3, -3))))
        self.assertEqual(test_mel0, test_mel1)

    def test_subvert(self):
        test_mel0 = ji.JIMel.mk_line(ji.JITone((0, 1, -1)), 2)
        f = ji.JITone((0, 0, -1))
        t = ji.JITone((0, 1))
        test_mel1 = ji.JIMel((t, f, t, t, f, f))
        self.assertEqual(test_mel0.subvert(), test_mel1)

    def test_accumulate(self):
        f = ji.JITone((0, 0, -1))
        t = ji.JITone((0, 1))
        test_mel0 = ji.JIMel((f, f, t, t))
        test_mel1 = ji.JIMel((f, f + f, f + f + t, f + f + t + t))
        self.assertEqual(test_mel0.accumulate(), test_mel1)

    def test_separate(self):
        test_mel0 = ji.JIMel.mk_line(ji.JITone((0, 1, -1)), 2)
        test_mel1 = ji.JIMel((ji.JITone.from_ratio(3, 5),
                              ji.JITone.from_ratio(9, 5),
                              ji.JITone.from_ratio(9, 25)))
        self.assertEqual(test_mel0.separate(), test_mel1)


if __name__ == "__main__":
    unittest.main()
