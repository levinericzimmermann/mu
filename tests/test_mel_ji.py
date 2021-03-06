import unittest

from mu.mel import ji

from fractions import Fraction
import json
import os


class MonzoTest(unittest.TestCase):
    def test_from_ratio(self):
        m0 = ji.Monzo([0, 1], 2)
        m0B = ji.Monzo.from_ratio(5, 4, val_border=2)
        m1 = ji.Monzo([2], 2)
        m1B = ji.Monzo.from_ratio(9, 8, val_border=2)
        m2 = ji.Monzo([-1, 1, 1], 1)
        m2B = ji.Monzo.from_ratio(15, 2, val_border=1)
        self.assertEqual(m0B, m0)
        self.assertEqual(m1B, m1)
        self.assertEqual(m2B, m2)

    def test_hash(self):
        m0 = ji.Monzo([0, 1, -1], 2)
        m0B = ji.Monzo([0, 1, -1], 2)
        m1 = ji.Monzo([0, 2], 2)
        m1B = ji.Monzo([0, 2], 2)
        m2 = ji.Monzo([2, -1], 1)
        m2B = ji.Monzo([2, -1], 1)
        self.assertEqual(hash(m0), hash(m0B))
        self.assertEqual(hash(m0), hash((m0._val_shift, m0._vec)))
        self.assertEqual(hash(m1), hash(m1B))
        self.assertEqual(hash(m1), hash((m1._val_shift, m1._vec)))
        self.assertEqual(hash(m2), hash(m2B))
        self.assertEqual(hash(m2), hash((m2._val_shift, m2._vec)))
        self.assertNotEqual(hash(m0), hash(m1))
        self.assertNotEqual(hash(m0), hash(m2))
        self.assertNotEqual(hash(m1), hash(m2))

    def test_from_json(self):
        arg = [0, 1, -1]
        vborder = 2
        m0 = ji.Monzo(arg, vborder)
        js = [arg, vborder]
        self.assertEqual(ji.Monzo.from_json(js), m0)

    def test_convert2json(self):
        arg = [0, 1, -1]
        vborder = 2
        m0 = ji.Monzo(arg, vborder)
        js = [arg, vborder]
        js = json.dumps(js)
        self.assertEqual(m0.convert2json(), js)

    def test_load_json(self):
        arg = [0, 1, -1]
        vborder = 2
        m0 = ji.Monzo(arg, vborder)
        js = [arg, vborder]
        js = json.dumps(js)
        name = "test_load_json.json"
        with open(name, "w") as f:
            f.write(js)
        self.assertEqual(ji.Monzo.load_json(name), m0)
        os.remove(name)

    def test_export_json(self):
        arg = [0, 1, -1]
        vborder = 2
        m0 = ji.Monzo(arg, vborder)
        name = "test_load_json.json"
        m0.export2json(name)
        self.assertEqual(ji.Monzo.load_json(name), m0)
        os.remove(name)

    def test_repr(self):
        m0 = ji.Monzo([0, 1, -1], 2)
        self.assertEqual(repr(m0), repr(m0._vec))

    def test_from_str(self):
        m0 = ji.Monzo([0, 1], 2)
        m0B = ji.Monzo.from_str("5/4")
        m0B._val_shift = 1
        m1 = ji.Monzo([2], 2)
        m1B = ji.Monzo.from_str("9/8")
        m1B._val_shift = 1
        m2 = ji.Monzo([-1, 1, 1], 1)
        m2B = ji.Monzo.from_str("15/2")
        self.assertEqual(m0B, m0)
        self.assertEqual(m1B, m1)
        self.assertEqual(m2B, m2)

    def test_from_monzo(self):
        m0 = ji.Monzo([0, 1], 2)
        m0B = ji.Monzo.from_monzo(0, 1, val_border=2)
        m1 = ji.Monzo([2], 2)
        m1B = ji.Monzo.from_monzo(2, val_border=2)
        m2 = ji.Monzo([-1, 1, 1], 1)
        m2B = ji.Monzo.from_monzo(-1, 1, 1, val_border=1)
        self.assertEqual(m0B, m0)
        self.assertEqual(m1B, m1)
        self.assertEqual(m2B, m2)

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
        self.assertEqual(m0._vec, m0._vector[m0._val_shift :])
        self.assertEqual(m1._vec, m1._vector[m1._val_shift :])
        self.assertEqual(m2._vec, m2._vector[m2._val_shift :])
        self.assertEqual(m3._vec, m3._vector[m3._val_shift :])

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

    def test_factorised(self):
        m0 = ji.Monzo([-1, 1], 1)
        m1 = ji.Monzo([-1, 1], 2)
        m2 = ji.Monzo([2], 2)
        m3 = ji.Monzo([2, -1], 2)
        self.assertEqual(m0.factorised, (2, 3))
        self.assertEqual(m1.factorised, (3, 5))
        self.assertEqual(m2.factorised, (2, 2, 2, 3, 3))
        self.assertEqual(m3.factorised, (3, 3, 5))

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
        self.assertEqual(ji.Monzo.adjust_ratio(Fraction(9, 1), 1), Fraction(9, 1))
        self.assertEqual(ji.Monzo.adjust_ratio(Fraction(9, 4), 2), Fraction(9, 8))
        self.assertEqual(ji.Monzo.adjust_ratio(Fraction(9, 16), 2), Fraction(9, 8))
        self.assertEqual(ji.Monzo.adjust_ratio(Fraction(15, 7), 2), Fraction(15, 14))
        self.assertEqual(ji.Monzo.adjust_ratio(Fraction(15, 7), 3), Fraction(15, 7))

    def test_adjust_monzo(self):
        vec0 = (1,)
        val0 = (3,)
        vec0res = (-1, 1)
        val0res = (2, 3)
        self.assertEqual(ji.Monzo.adjust_monzo(vec0, val0, 2), (vec0res, val0res))
        vec1 = (0, 1)
        val1 = (3, 5)
        vec1res = (-2, 0, 1)
        val1res = (2, 3, 5)
        self.assertEqual(ji.Monzo.adjust_monzo(vec1, val1, 2), (vec1res, val1res))
        vec2 = (-1, 1)
        val2 = (3, 5)
        vec2res = (0, -1, 1)
        val2res = (2, 3, 5)
        self.assertEqual(ji.Monzo.adjust_monzo(vec2, val2, 2), (vec2res, val2res))

    def test_ratio2monzo(self):
        self.assertEqual(ji.Monzo.ratio2monzo(Fraction(4, 3)), ji.Monzo((2, -1)))
        self.assertEqual(ji.Monzo.ratio2monzo(Fraction(9, 8)), ji.Monzo((-3, 2)))
        self.assertEqual(ji.Monzo.ratio2monzo(Fraction(9, 5)), ji.Monzo((0, 2, -1)))

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
        ls = [
            ji.Monzo([0, 0, 1]),
            ji.Monzo([0, 0, 0, 1]),
            ji.Monzo([0, 0, 0, 1]),
            ji.Monzo([0, 0, 0, 0, -1]),
            ji.Monzo([0, 0, 0, 0, -1]),
        ]
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

    def test_sibling(self):
        m0 = ji.Monzo([1], 1)
        m1 = ji.Monzo([-2, 1], 1)
        m2 = ji.Monzo([0, 1], 1)
        m3 = ji.Monzo([-1], 1)
        m4 = ji.Monzo([2], 1)
        m5 = ji.Monzo([10], 1)
        self.assertEqual(m0.is_sibling(m0), True)
        self.assertEqual(m0.is_sibling(m1), False)
        self.assertEqual(m0.is_sibling(m2), False)
        self.assertEqual(m0.is_sibling(m3), False)
        self.assertEqual(m0.is_sibling(m4), True)
        self.assertEqual(m0.is_sibling(m5), True)

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
        p0 = ji.r(2, 1)
        p1 = ji.r(1, 2)
        p2 = ji.r(7, 6)
        p3 = ji.r(6, 7)
        p4 = ji.r(10, 3)
        p5 = ji.r(3, 10)
        self.assertEqual(abs(p1), p0)
        self.assertEqual(abs(p3), p2)
        self.assertEqual(abs(p5), p4)

    def test_normalize(self):
        p0 = ji.JIPitch((0, 1), 1)
        p1 = ji.JIPitch((-1, 1), 1)
        self.assertEqual(p0.normalize(2), p1)

    def test_is_symmetric(self):
        p0 = ji.JIPitch((1, -1), 2)
        p1 = ji.JIPitch((2, -2, 0, -2), 2)
        p2 = ji.JIPitch((1, 3), 2)
        p3 = ji.JIPitch((2, -1), 2)
        p4 = ji.JIPitch((0, 2, 2), 2)
        self.assertEqual(p0.is_symmetric, True)
        self.assertEqual(p1.is_symmetric, True)
        self.assertEqual(p2.is_symmetric, False)
        self.assertEqual(p3.is_symmetric, False)
        self.assertEqual(p4.is_symmetric, True)

    def test_adjusted_register(self):
        m0 = ji.Monzo.from_ratio(9, 8, val_border=2)
        self.assertEqual(m0.adjust_register().ratio, Fraction(9, 4))

    def test_indigestibility(self):
        self.assertEqual(ji.Monzo.indigestibility(1), 0)
        self.assertEqual(ji.Monzo.indigestibility(2), 1)
        self.assertEqual(ji.Monzo.indigestibility(4), 2)
        self.assertEqual(ji.Monzo.indigestibility(5), 6.4)
        self.assertEqual(ji.Monzo.indigestibility(6), 3.6666666666666665)
        self.assertEqual(ji.Monzo.indigestibility(8), 3)

    def test_harmonicity_barlow(self):
        m0 = ji.Monzo((1,), val_border=2)
        m1 = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1), val_border=2)
        m3 = ji.Monzo((0, -1), val_border=2)
        self.assertEqual(m0.harmonicity_barlow, 0.27272727272727276)
        self.assertEqual(m1.harmonicity_barlow, float("inf"))
        self.assertEqual(m2.harmonicity_barlow, 0.11904761904761904)
        self.assertEqual(m3.harmonicity_barlow, -0.10638297872340426)

    def test_harmonicity_euler(self):
        m0 = ji.Monzo((1,), val_border=2)
        m1 = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1), val_border=2)
        m3 = ji.Monzo((0, -1), val_border=2)
        self.assertEqual(m0.harmonicity_euler, 4)
        self.assertEqual(m1.harmonicity_euler, 1)
        self.assertEqual(m2.harmonicity_euler, 7)
        self.assertEqual(m3.harmonicity_euler, 8)

    def test_harmonicity_tenney(self):
        m0 = ji.Monzo((1,), val_border=2)
        m1 = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1), val_border=2)
        m3 = ji.Monzo((0, -1), val_border=2)
        self.assertEqual(m0.harmonicity_tenney, 2.584962500721156)
        self.assertEqual(m1.harmonicity_tenney, 0)
        self.assertEqual(m2.harmonicity_tenney, 4.321928094887363)
        self.assertEqual(m3.harmonicity_tenney, 5.321928094887363)

    def test_harmonicity_vogel(self):
        m0 = ji.Monzo((1,), val_border=2)
        m1 = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1), val_border=2)
        m3 = ji.Monzo((0, -1), val_border=2)
        self.assertEqual(m0.harmonicity_vogel, 4)
        self.assertEqual(m1.harmonicity_vogel, 1)
        self.assertEqual(m2.harmonicity_vogel, 7)
        self.assertEqual(m3.harmonicity_vogel, 8)

    def test_harmonicity_wilson(self):
        m0 = ji.Monzo((1,), val_border=2)
        m1 = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1), val_border=2)
        m3 = ji.Monzo((0, -1), val_border=2)
        self.assertEqual(m0.harmonicity_wilson, 3)
        self.assertEqual(m1.harmonicity_wilson, 1)
        self.assertEqual(m2.harmonicity_wilson, 5)
        self.assertEqual(m3.harmonicity_wilson, 5)

    def test_sparsity(self):
        m0 = ji.Monzo((1,), val_border=2)
        m1 = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1), val_border=2)
        m3 = ji.Monzo((0, -1, 1, 2), val_border=2)
        self.assertEqual(m0.sparsity, 0)
        self.assertEqual(m1.sparsity, 0)
        self.assertEqual(m2.sparsity, 1 / 2)
        self.assertEqual(m3.sparsity, 1 / 4)

    def test_density(self):
        m0 = ji.Monzo((1, 2, 1), val_border=2)
        m1 = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1), val_border=2)
        m3 = ji.Monzo((0, -1, 1, 4), val_border=2)
        self.assertEqual(m0.density, 1)
        self.assertEqual(m1.density, 1)
        self.assertEqual(m2.density, 0.5)
        self.assertEqual(m3.density, 0.75)

    def test_mk_filter_vec(self):
        primes0 = (3, 5)
        expected0 = (1, 0, 0)
        self.assertEqual(ji.Monzo.mk_filter_vec(*primes0), expected0)
        primes1 = (3, 5, 11)
        expected1 = (1, 0, 0, 1, 0)
        self.assertEqual(ji.Monzo.mk_filter_vec(*primes1), expected1)
        primes2 = (2, 5, 11, 17)
        expected2 = (0, 1, 0, 1, 0, 1, 0)
        self.assertEqual(ji.Monzo.mk_filter_vec(*primes2), expected2)

    def test_filter(self):
        m0 = ji.Monzo((1, 2, 1), val_border=2)
        m0B = ji.Monzo((1, 0, 1), val_border=2)
        m1 = ji.Monzo([3], val_border=2)
        m1B = ji.Monzo([], val_border=2)
        m2 = ji.Monzo((0, 1, 1, 1), val_border=2)
        m2B = ji.Monzo((0, 0, 1), val_border=2)
        m3 = ji.Monzo((2, -1, 1, 4), val_border=1)
        m3B = ji.Monzo((0, -1, 1, 4), val_border=1)
        self.assertEqual(m0.filter(5), m0B)
        self.assertEqual(m1.filter(3), m1B)
        self.assertEqual(m2.filter(5, 11), m2B)
        self.assertEqual(m3.filter(2), m3B)


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

    def test_octave(self):
        m0 = ji.JIPitch([0])
        m1 = ji.JIPitch([1])
        m2 = ji.JIPitch([2])
        m3 = ji.JIPitch([3])
        m4 = ji.JIPitch([-1])
        m5 = ji.JIPitch([-2])
        m6 = ji.JIPitch([-3])
        m7 = ji.JIPitch([-1, 1])
        m8 = ji.JIPitch([0, 1])
        m9 = ji.JIPitch([1, -1])
        self.assertEqual(m0.octave, 0)
        self.assertEqual(m1.octave, 1)
        self.assertEqual(m2.octave, 2)
        self.assertEqual(m3.octave, 3)
        self.assertEqual(m4.octave, -1)
        self.assertEqual(m5.octave, -2)
        self.assertEqual(m6.octave, -3)
        self.assertEqual(m7.octave, 0)
        self.assertEqual(m8.octave, 1)
        self.assertEqual(m9.octave, -1)

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

    def test_repr(self):
        r0 = Fraction(3, 2)
        r1 = Fraction(7, 5)
        t0 = ji.r(r0.numerator, r0.denominator)
        t1 = ji.r(r1.numerator, r1.denominator)
        self.assertEqual(str(r0), repr(t0))
        self.assertEqual(str(r1), repr(t1))

    def test_differential_pitch(self):
        p0 = ji.r(7, 4)
        p1 = ji.r(4, 3)
        p2 = ji.r(5, 12)
        self.assertEqual(p0.differential(p1), p2)
        self.assertEqual(p1.differential(p0), p2)
        p0 += ji.r(2, 1)
        p3 = ji.r(13, 6)
        self.assertEqual(p0.differential(p1), p3)


class JIScaleTest(unittest.TestCase):
    def test_add(self):
        scale0 = ji.JIScale(
            [ji.r(1, 1), ji.r(9, 8), ji.r(4, 3), ji.r(3, 2), ji.r(7, 4)], ji.r(2, 1)
        )
        scale1 = ji.JIScale(
            [ji.r(1, 1), ji.r(16, 15), ji.r(4, 3), ji.r(3, 2), ji.r(7, 4)], ji.r(2, 1)
        )
        scale2 = ji.JIScale(
            [ji.r(1, 1), ji.r(16, 15), ji.r(9, 8), ji.r(4, 3), ji.r(3, 2), ji.r(7, 4)],
            ji.r(2, 1),
        )
        self.assertEqual(scale0 + scale1, scale2)

    """
    def test_intervals(self):
        scale = ji.JIScale(
            [ji.r(1, 1), ji.r(9, 8), ji.r(4, 3), ji.r(3, 2), ji.r(7, 4)], ji.r(2, 1)
        )
        intervals = ji.JIMel(
            [ji.r(9, 8), ji.r(32, 27), ji.r(9, 8), ji.r(7, 6), ji.r(8, 7)]
        )
        self.assertEqual(scale.intervals, intervals)
    """


class JIMelTest(unittest.TestCase):
    def test_from_str(self):
        m0 = ji.JIMel([ji.r(1, 1), ji.r(4, 3), ji.r(5, 4), ji.r(9, 8)])
        m0B = ji.JIMel.from_str("1/1, 4/3, 5/4, 9/8")
        self.assertEqual(m0, m0B)

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
        correct = (float(m_fac * 2 * Fraction(3, 2)), float(m_fac * Fraction(5, 4)))
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
            (ji.JIPitch((0, 1, -1)), ji.JIPitch((0, 2, -2)), ji.JIPitch((0, 3, -3)))
        )
        self.assertEqual(test_mel0, test_mel1)

    def test_mk_line_and_inverse(self):
        test_mel0 = ji.JIMel.mk_line_and_inverse(ji.JIPitch((0, 1, -1)), 3)
        test_mel1 = ji.JIMel.mk_line(ji.JIPitch((0, 1, -1)), 3)
        test_mel1 = test_mel1 + test_mel1.inverse()
        self.assertEqual(test_mel0, test_mel1)

    def test_intervals(self):
        test_mel0 = ji.JIMel(
            (ji.JIPitch((0, 1, -1)), ji.JIPitch((0, 2, -2)), ji.JIPitch((0, 3, -3)))
        )
        test_mel1 = ji.JIMel((ji.JIPitch((0, 1, -1)), ji.JIPitch((0, 1, -1))))
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
        self.assertEqual(test_mel0.pitch_rate, ((t0, 1), (t1, 1), (t2, 1), (t3, 1)))
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
        self.assertEqual(
            test_mel0.pitch_rate_sorted, ((t0, 1), (t1, 1), (t2, 1), (t3, 1))
        )
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
        test_mel1 = ji.JIMel(
            (
                ji.JIPitch.from_ratio(3, 5),
                ji.JIPitch.from_ratio(9, 5),
                ji.JIPitch.from_ratio(9, 25),
            )
        )
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
        self.assertEqual(test_mel0.dominant_prime, (3,))

    def test_remove(self):
        p0 = ji.JIPitch((0, 1, -1), 2)
        p1 = ji.JIPitch((0, 1), 2)
        p2 = ji.JIPitch((0, 0, 1), 2)
        p3 = ji.JIPitch((0, 2), 2)
        p4 = ji.JIPitch((0, -2), 2)
        test_mel0 = ji.JIMel([p4, p0, p4, p1, p2, p3, p4])
        test_mel1 = ji.JIMel([p0, p1, p2, p3])
        test_mel2 = ji.JIMel([p4, p4, p1, p2, p3, p4])
        self.assertEqual(test_mel0.remove(p4), test_mel1)
        self.assertEqual(test_mel0.remove(p0), test_mel2)
        test_mel0.val_border = 3
        test_mel1.val_border = 3
        p4.val_border = 3
        self.assertEqual(test_mel0.remove(p4), test_mel1)

    def test_find_by(self):
        def summed_minus(p0, p1):
            return (p0 - p1).summed()

        p0 = ji.JIPitch((1, -1), 2)
        p1 = ji.JIPitch((1,), 2)
        p2 = ji.JIPitch((0, 1), 2)
        p3 = ji.JIPitch((2,), 2)
        p4 = ji.JIPitch((-2,), 2)
        p5 = ji.JIPitch((-1,), 2)
        test_mel0 = ji.JIMel([p0, p1, p2, p3, p4])
        self.assertEqual(test_mel0.find_by(p0, summed_minus), p0)
        self.assertEqual(test_mel0.find_by(p5, summed_minus), p4)

    def test_find_by_walk(self):
        def summed_minus(p0, p1):
            return (p0 - p1).summed()

        p0 = ji.JIPitch((1, -1), 2)
        p1 = ji.JIPitch((1,), 2)
        p2 = ji.JIPitch((0, 1), 2)
        p3 = ji.JIPitch((2,), 2)
        p4 = ji.JIPitch((-2,), 2)
        test_mel0 = ji.JIMel([p0, p1, p2, p3, p4])
        test_result = ji.JIMel((p0, p0, p1, p3, p2, p4))
        self.assertEqual(test_mel0.find_by_walk(p0, summed_minus), test_result)

    def test_uniqify(self):
        p0 = ji.JIPitch((1, -1), 2)
        p1 = ji.JIPitch((1,), 2)
        p2 = ji.JIPitch((0, 1), 2)
        p3 = ji.JIPitch((2,), 2)
        p4 = ji.JIPitch((-2,), 2)
        test_mel0 = ji.JIMel((p0, p1, p0, p2, p3, p2, p4, p0))
        test_mel1 = ji.JIMel((p0, p1, p2, p3, p4))
        self.assertEqual(test_mel0.uniqify(), test_mel1)


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
        self.assertEqual(h5.root, (n3, n1))
        self.assertEqual(h6.root, tuple(reversed((n1, n1.inverse(), n3.inverse(), n3))))

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
        harmonic_series = ji.JIHarmony(
            [ji.r(1, 7), ji.r(2, 7), ji.r(3, 7), ji.r(4, 7), ji.r(5, 7)]
        )
        self.assertEqual(
            ji.JIHarmony.mk_harmonic_series(test_pitch, 6), harmonic_series
        )

    def test_intervals(self):
        h0 = ji.JIHarmony([ji.r(1, 1), ji.r(7, 4), ji.r(3, 2)])
        h0_intervals = ji.JIHarmony(
            [
                ji.r(4, 7),
                ji.r(7, 4),
                ji.r(7, 4) - ji.r(3, 2),
                ji.r(2, 3),
                ji.r(6, 7),
                ji.r(3, 2),
            ]
        )
        h1 = ji.JIHarmony([ji.r(1, 1), ji.r(5, 4), ji.r(2, 1), ji.r(8, 5)])
        h1_intervals = ji.JIHarmony(
            [
                ji.r(1, 2),
                ji.r(2, 1),
                ji.r(4, 5),
                ji.r(25, 32),
                ji.r(5, 4),
                ji.r(2, 1),
                ji.r(8, 5),
                ji.r(5, 8),
                ji.r(32, 25),
            ]
        )
        self.assertEqual(h0.intervals, h0_intervals)
        self.assertEqual(h1.intervals, h1_intervals)


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
        self.assertEqual(cadence0.identity, (h0.identity, h1.identity, h2.identity))
        self.assertEqual(
            cadence1.identity, (h3.identity, h4.identity, h5.identity, h6.identity)
        )

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

    def test_chord_rate(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = ji.JIHarmony([n0, n1])
        cadence0 = ji.JICadence((h0, h0, h0))
        cadence1 = ji.JICadence((h0, h0, h1, h2))
        self.assertEqual(cadence0.chord_rate, ((h0, 3),))
        self.assertEqual(cadence1.chord_rate, ((h0, 2), (h1, 1), (h2, 1)))

    def test_chord_rate_sorted(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = ji.JIHarmony([n0, n1])
        cadence0 = ji.JICadence((h1, h0, h0, h0))
        cadence1 = ji.JICadence((h0, h0, h1, h2))
        self.assertEqual(cadence0.chord_rate_sorted, ((h1, 1), (h0, 3)))
        self.assertEqual(cadence1.chord_rate_sorted, ((h1, 1), (h2, 1), (h0, 2)))

    def test_different_chords(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = ji.JIHarmony([n0, n1])
        cadence0 = ji.JICadence((h1, h0, h0))
        cadence1 = ji.JICadence((h0, h0, h1, h2))
        cadence2 = ji.JICadence((h2, h2, h2))
        cadence3 = ji.JICadence((h0,))
        self.assertEqual(cadence0.different_chords, (h1, h0))
        self.assertEqual(cadence1.different_chords, (h0, h1, h2))
        self.assertEqual(cadence2.different_chords, (h2,))
        self.assertEqual(cadence3.different_chords, (h0,))

    def test_count_pitch(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = ji.JIHarmony([n3, n1])
        cadence0 = ji.JICadence((h1, h0, h0))
        cadence1 = ji.JICadence((h0, h0, h1, h2))
        self.assertEqual(cadence0.count_pitch(n0), 3)
        self.assertEqual(cadence1.count_pitch(n0), 3)

    def test_count_different_pitches(self):
        n0 = ji.JIPitch([], val_border=2)
        n1 = ji.JIPitch([1], val_border=2)
        n2 = ji.JIPitch([1, 1], val_border=2)
        n3 = ji.JIPitch([0, 1], val_border=2)
        h0 = ji.JIHarmony([n0, n1, n3])
        h1 = ji.JIHarmony([n0, n1, n2])
        h2 = ji.JIHarmony([n3, n1])
        cadence0 = ji.JICadence((h1, h0, h0))
        cadence1 = ji.JICadence((h0, h0, h2))
        self.assertEqual(cadence0.count_different_pitches(), 3)
        self.assertEqual(cadence1.count_different_pitches(), 3)


class JIStencilTest(unittest.TestCase):
    def test_add_zero(self):
        m0 = ji.Monzo([0, 1], 2)
        tuple0 = (m0, 2)
        tuple1 = (m0, 0, 2)
        stencil = ji.JIStencil(tuple0, tuple1)
        self.assertEqual(stencil._vector[0], stencil._vector[1])

    def test_convert2harmony(self):
        p0 = ji.JIPitch((1,), 2)
        p1 = ji.JIPitch((0, 1), 2)
        teststencil = ji.JIStencil((p0, 0, 2), (p1, 1, 3))
        testharmony = ji.JIHarmony((p0.scalar(0), p0, p1, p1 + p1))
        self.assertEqual(teststencil.convert2harmony(), testharmony)

    def test_primes(self):
        p0 = ji.JIPitch((1,), 2)
        p1 = ji.JIPitch((0, 1), 2)
        teststencil = ji.JIStencil((p0, 0, 2), (p1, 1, 3))
        self.assertEqual(teststencil.primes, (3, 5))

    def test_identity(self):
        p0 = ji.JIPitch((1,), 2)
        p1 = ji.JIPitch((0, 1), 2)
        teststencil = ji.JIStencil((p0, 0, 2), (p1, 1, 3))
        self.assertEqual(teststencil.identity, (p1, p0))


class BlueprintPitchTest(unittest.TestCase):
    def test_init(self):
        self.assertRaises(AssertionError, ji.BlueprintPitch, [0], [1])
        self.assertRaises(AssertionError, ji.BlueprintPitch, [-1], [1])

        # that's the correct form
        self.assertTrue(ji.BlueprintPitch([1, 2], []))
        self.assertTrue(ji.BlueprintPitch())
        self.assertTrue(ji.BlueprintPitch([2], [1]))

    def test_size(self):
        bp0 = ji.BlueprintPitch((2,), [])
        bp1 = ji.BlueprintPitch((2,), (2,))
        bp2 = ji.BlueprintPitch([], [])

        self.assertEqual(bp0.size, 2)
        self.assertEqual(bp1.size, 4)
        self.assertEqual(bp2.size, 0)

    def test_is_instance(self):
        bp = ji.BlueprintPitch((2,), [])
        p0 = ji.r(15, 8)
        p1 = ji.r(3, 2)
        p2 = ji.r(3, 5)
        p3 = ji.r(32, 21)
        p4 = ji.r(33, 21)

        self.assertTrue(bp.is_instance(p0))
        self.assertFalse(bp.is_instance(p1))
        self.assertFalse(bp.is_instance(p2))
        self.assertFalse(bp.is_instance(p3))
        self.assertFalse(bp.inverse().is_instance(p0))
        self.assertFalse(bp.is_instance(p4))

    def test_call(self):
        bp0 = ji.BlueprintPitch((2,), [])
        bp1 = ji.BlueprintPitch((1,), (1,))

        p0 = ji.r(15, 1)
        p1 = ji.r(7, 3)

        self.assertEqual(bp0(3, 5), p0)
        self.assertEqual(bp0(5, 3), p0)
        self.assertNotEqual(bp0(7, 3), p1)
        self.assertEqual(bp1(7, 3), p1)
        self.assertNotEqual(bp1(3, 7), p1)

        # 10 isn't a prime number
        self.assertRaises(ValueError, bp0, 7, 10)

        # 4 isn't a prime number
        self.assertRaises(ValueError, bp0, 7, 4)

        # bp0 only has 2 arguments
        self.assertRaises(ValueError, bp0, 7, 5, 11)


class BlueprintHarmonyTest(unittest.TestCase):
    bp0 = ji.BlueprintPitch((1,))
    bp1 = ji.BlueprintPitch((1,), (1,))
    bp2 = ji.BlueprintPitch([], (2,))
    bp3 = ji.BlueprintPitch([0, 1], [])

    def test_init(self):
        self.assertTrue(
            ji.BlueprintHarmony(
                (BlueprintHarmonyTest.bp0, (0,)),
                (BlueprintHarmonyTest.bp0, (1,)),
                (BlueprintHarmonyTest.bp3, (0,)),
            )
        )

        # indices don't have to be in ascending order
        self.assertTrue(
            ji.BlueprintHarmony(
                (BlueprintHarmonyTest.bp0, (0,)),
                (BlueprintHarmonyTest.bp0, (2,)),
                (BlueprintHarmonyTest.bp3, (0,)),
            )
        )
        self.assertTrue(
            ji.BlueprintHarmony(
                (BlueprintHarmonyTest.bp0, (1,)),
                (BlueprintHarmonyTest.bp0, (4,)),
                (BlueprintHarmonyTest.bp3, (5,)),
            )
        )

    def test_call(self):
        bph0 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )

        bph1 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp1, (0, 1)),
            (BlueprintHarmonyTest.bp1, (1, 0)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )

        self.assertEqual(bph0(3, 5), (ji.r(3, 1), ji.r(5, 1), ji.r(9, 1)))
        self.assertEqual(bph0(7, 5), (ji.r(7, 1), ji.r(5, 1), ji.r(49, 1)))

        self.assertEqual(bph1(3, 5), (ji.r(3, 1), ji.r(3, 5), ji.r(5, 3), ji.r(9, 1)))
        self.assertEqual(bph1(7, 5), (ji.r(7, 1), ji.r(7, 5), ji.r(5, 7), ji.r(49, 1)))

    def test_identity(self):
        bph0 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )

        identity = {
            ((hash(BlueprintHarmonyTest.bp0), 0), (hash(BlueprintHarmonyTest.bp3), 0)),
            ((hash(BlueprintHarmonyTest.bp0), 0),),
        }

        self.assertEqual(identity, bph0.identity)

    def test_equal(self):
        bph0 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )
        bph1 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )
        bph2 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp3, (1,)),
        )
        bph3 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (3,)),
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp3, (3,)),
        )
        bph4 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp2, (0, 1)),
            (BlueprintHarmonyTest.bp3, (1,)),
        )
        bph5 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp2, (0, 1)),
            (BlueprintHarmonyTest.bp3, (1,)),
        )
        bph6 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp2, (0, 1)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )
        bph7 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (2,)),
            (BlueprintHarmonyTest.bp2, (0, 2)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )
        bph8 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp2, (1, 0)),
            (BlueprintHarmonyTest.bp3, (1,)),
        )
        bph9 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp1, (0, 1)),
            (BlueprintHarmonyTest.bp3, (1,)),
        )
        bph10 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp1, (1, 0)),
            (BlueprintHarmonyTest.bp3, (1,)),
        )
        bph11 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)), (BlueprintHarmonyTest.bp0, (0,))
        )
        bph12 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (1,)), (BlueprintHarmonyTest.bp0, (1,))
        )

        self.assertEqual(bph0, bph1)
        self.assertEqual(bph0, bph2)
        self.assertEqual(bph1, bph2)
        self.assertEqual(bph1, bph3)
        self.assertEqual(bph4, bph5)
        self.assertEqual(bph4, bph6)
        self.assertEqual(bph4, bph7)
        self.assertEqual(bph4, bph8)
        self.assertNotEqual(bph9, bph10)
        self.assertEqual(bph11, bph12)

    def test_is_instance(self):
        bph0 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )
        bph1 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp3, (1,)),
        )

        self.assertTrue(bph0.is_instance((ji.r(3, 1), ji.r(5, 1), ji.r(9, 1))))
        self.assertTrue(bph1.is_instance((ji.r(3, 1), ji.r(5, 1), ji.r(9, 1))))
        self.assertTrue(
            bph1.inverse().is_instance((ji.r(1, 3), ji.r(1, 5), ji.r(1, 9)))
        )

        self.assertFalse(bph0.is_instance((ji.r(7, 1), ji.r(5, 1), ji.r(9, 1))))
        self.assertFalse(bph0.is_instance((ji.r(5, 1), ji.r(5, 1), ji.r(7, 1))))
        self.assertFalse(
            bph0.inverse().is_instance((ji.r(3, 1), ji.r(5, 1), ji.r(9, 1)))
        )

    def test_n_pitch_repetitions(self):
        bph0 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (1,)), (BlueprintHarmonyTest.bp0, (1,))
        )
        bph1 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp0, (1,)),
        )
        bph2 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp1, (0, 1)), (BlueprintHarmonyTest.bp1, (1, 0))
        )
        bph3 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp1, (0, 1)), (BlueprintHarmonyTest.bp1, (0, 1))
        )
        bph4 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp2, (0, 1)), (BlueprintHarmonyTest.bp2, (0, 1))
        )
        bph5 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp2, (0, 1)), (BlueprintHarmonyTest.bp2, (1, 0))
        )

        self.assertEqual(bph0.n_pitch_repetitions, 1)
        self.assertEqual(bph1.n_pitch_repetitions, 2)
        self.assertEqual(bph2.n_pitch_repetitions, 0)
        self.assertEqual(bph3.n_pitch_repetitions, 1)
        self.assertEqual(bph4.n_pitch_repetitions, 1)
        self.assertEqual(bph5.n_pitch_repetitions, 1)

    def test_n_common_pitches(self):
        bph0 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp0, (0,)),
            (BlueprintHarmonyTest.bp0, (1,)),
            (BlueprintHarmonyTest.bp3, (0,)),
        )
        bph1 = ji.BlueprintHarmony(
            (BlueprintHarmonyTest.bp3, (1,)),
            (BlueprintHarmonyTest.bp1, (0, 1)),
            (BlueprintHarmonyTest.bp0, (0,)),
        )

        self.assertEqual(bph0.n_common_pitches(bph1), 1)


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
