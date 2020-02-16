try:
    import quicktions as fractions
except ImportError:
    import fractions

import unittest

from mu.rhy import binr


class CompoundTest(unittest.TestCase):
    def test_is_unvalid_essence(self) -> None:
        self.assertFalse(binr.Compound.is_valid_essence(-2))
        self.assertFalse(binr.Compound.is_valid_essence(1.4))
        self.assertFalse(binr.Compound.is_valid_essence("hi"))

    def test_is_unvalid_multiply(self) -> None:
        self.assertFalse(binr.Compound.is_valid_multiply(-2))
        self.assertFalse(binr.Compound.is_valid_multiply("hi"))
        self.assertFalse(binr.Compound.is_valid_multiply(True))

    def test_conversion_functions(self) -> None:
        self.assertEqual(binr.Compound.convert_int2binary_rhythm(10), (1, 0, 1, 0))
        self.assertEqual(binr.Compound.convert_int2binary_rhythm(11), (1, 0, 1, 1))
        self.assertEqual(binr.Compound.convert_int2binary_rhythm(12), (1, 1, 0, 0))
        self.assertEqual(binr.Compound.convert_int2binary_rhythm(20), (1, 0, 1, 0, 0))

        self.assertEqual(
            binr.Compound.convert_binary_rhythm2rhythm((1, 0, 1, 0, 0)), (2, 3)
        )
        self.assertEqual(
            binr.Compound.convert_binary2binary_rhythm(bin(28)), (1, 1, 1, 0, 0)
        )
        self.assertEqual(
            binr.Compound.convert_binary2binary_rhythm(bin(113)), (1, 1, 1, 0, 0, 0, 1)
        )
        self.assertEqual(binr.Compound.convert_binary2rhythm(bin(28)), (1, 1, 3))
        self.assertEqual(binr.Compound.convert_binary2rhythm(bin(113)), (1, 1, 4, 1))
        self.assertEqual(
            binr.Compound.convert_binary_rhythm2rhythm((1, 1, 1, 0, 0)), (1, 1, 3)
        )
        self.assertEqual(
            binr.Compound.convert_binary_rhythm2rhythm((1, 1, 0, 0, 1)), (1, 3, 1)
        )
        self.assertEqual(binr.Compound.convert_int2rhythm(20), (2, 3))
        self.assertEqual(
            binr.Compound.convert_binary_rhythm2binary((1, 0, 1, 0, 0)), bin(20)
        )
        self.assertEqual(
            binr.Compound.convert_binary_rhythm2binary((1, 1, 0, 1, 1, 0)), bin(54)
        )
        self.assertEqual(
            binr.Compound.convert_int_rhythm2binary_rhythm((1, 2, 1, 2)),
            (1, 1, 0, 1, 1, 0),
        )
        self.assertEqual(
            binr.Compound.convert_int_rhythm2binary_rhythm((4, 2, 3, 2)),
            (1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0),
        )
        self.assertEqual(
            binr.Compound.convert_rhythm2essence_and_multiply((1, 0.5, 2, 1)),
            (354, fractions.Fraction(1, 2)),
        )
        multiplied = fractions.Fraction(1, 3)
        self.assertEqual(
            binr.Compound.convert_rhythm2essence_and_multiply(
                tuple(r * multiplied for r in (4, 1, 1, 2))
            ),
            (142, multiplied),
        )
        self.assertEqual(
            binr.Compound.convert_essence_and_multiply2rhythm(
                354, fractions.Fraction(1, 2)
            ),
            list(fractions.Fraction(n) for n in (1, 0.5, 2, 1)),
        )

    def test_init_functions(self) -> None:
        basic = [8, 4, 7, 12, 8]
        factor = fractions.Fraction(1, 4)
        rhythm = [i * factor for i in basic]
        c0 = binr.Compound(rhythm)
        self.assertEqual(list(c0), rhythm)

        c1 = binr.Compound.from_binary(bin(276019282048), multiply=factor)
        self.assertEqual(list(c0), list(c1))
        self.assertEqual(c0.multiply, c1.multiply)

        c2 = binr.Compound.from_int(276019282048, factor)
        self.assertEqual(list(c0), list(c2))
        self.assertEqual(c0.multiply, c2.multiply)

        c3 = binr.Compound.from_binary_rhythm(
            (
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            factor,
        )
        self.assertEqual(list(c0), list(c3))
        self.assertEqual(c0.multiply, c3.multiply)

        # equality method tester
        self.assertEqual(c0, c1)
        self.assertEqual(c0, c2)
        self.assertEqual(c0, c3)
        self.assertEqual(c0, list(c1))

        c4 = binr.Compound((1, 1, 1, 1))
        self.assertNotEqual(c0, c4)

        c5 = binr.Compound(tuple(c0.intr))
        self.assertNotEqual(c0, c5)

    def test_get_item(self) -> None:
        basic = [2, 1, 1, 3, 2]
        c = binr.Compound(basic)
        for idx, n in enumerate(basic):
            self.assertEqual(c[idx], n)

    def test_set_item(self) -> None:
        basic = [2, 1, 1, 3, 2]
        c = binr.Compound(basic)
        new_value = 0.5
        c[1] = new_value
        self.assertEqual(c.multiply, new_value)
        self.assertEqual(c[1], new_value)
