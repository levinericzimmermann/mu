import unittest
from mu.mel import abstract


class AbstractTest(unittest.TestCase):
    def test_abstract_error(self):
        self.assertRaises(TypeError, type("test_cls", (abstract.AbstractPitch,), {}))

    def test_private_test(self):
        self.assertEqual(abstract.is_private("__repr"), True)
        self.assertEqual(abstract.is_private("_repr"), True)
        self.assertEqual(abstract.is_private("repr"), False)


class InheritanceTest(unittest.TestCase):
    class PitchTest(abstract.AbstractPitch):
        def __init__(self, freq):
            self._freq = freq

        def __repr__(self):
            return str(self._freq)

        def calc(self):
            return self._freq

        def multiply(self, arg):
            self._freq *= arg

        def multiplied(self, arg):
            return type(self)(self.freq * arg)

    def test_Pitch_functionality(self):
        self.PitchTest(200)

    def test_Pitch_methods_functionality(self):
        f = 200
        n0 = self.PitchTest(f)
        self.assertEqual(n0.freq, f)
        self.assertEqual(n0.calc(), f)

    def test_iterable_functionality(self):
        test_class0 = self.PitchTest.mk_iterable(list)
        test_class0([])

    def test_iterable_inheritance(self):
        self.assertIsInstance(self.PitchTest.mk_iterable(list)([]), list)
        self.assertIsInstance(self.PitchTest.mk_iterable(set)([]), set)
        self.assertIsInstance(self.PitchTest.mk_iterable(tuple)([]), tuple)

    def test_iterable_method_takeover(self):
        test_class0 = self.PitchTest.mk_iterable(list)
        self.assertIn("calc", dir(test_class0))
        self.assertIn("multiply", dir(test_class0))

    def test_iterable_method_functionality0(self):
        """test whether funtions which might return floats
        returns a list of floats in the iterable-version.
        test whether functions which returns None
        returns a list of None in the iterable-version"""
        fac = 2
        f0, f1 = 200, 260
        n0, n1 = self.PitchTest(f0), self.PitchTest(f1)
        test_class = self.PitchTest.mk_iterable(list)
        test_obj0 = test_class([n0, n1])
        self.assertEqual(test_obj0.calc(), (f0, f1))
        self.assertEqual(test_obj0.multiply(fac), None)
        correct = (f0 * fac, f1 * fac)
        self.assertEqual(test_obj0.calc(), correct)
        self.assertEqual(test_obj0.freq, correct)

    def test_iterable_method_functionality1(self):
        """test whether funtions which might return a type(self)-object
        returns a specific_iterable_type([type(self), ..])-object
        in the iterable-version."""
        fac = 2
        f0, f1 = 200, 260
        n0, n1 = self.PitchTest(f0), self.PitchTest(f1)
        test_class = self.PitchTest.mk_iterable(list)
        test_obj0 = test_class([n0, n1])
        compare_obj = test_class([n0.multiplied(fac), n1.multiplied(2)])
        self.assertEqual(test_obj0.multiplied(fac), compare_obj)

    def test_midi_conversion(self):
        """test whether the midi conversion function works properly"""
        f0 = 300
        n0 = self.PitchTest(f0)
        hex_number = n0.convert2midi_tuning()
        closest_pitch = 293.6647679174075
        cent_difference = abstract.AbstractPitch.hz2ct(closest_pitch, f0)
        steps0 = int(cent_difference // 0.78125)
        steps1 = int((cent_difference - (steps0 * 0.78125)) // 0.0061)
        expected_hex = 50, steps0, steps1
        self.assertEqual(hex_number, expected_hex)


if __name__ == "__main__":
    unittest.main()
