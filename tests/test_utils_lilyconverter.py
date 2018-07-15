import unittest
from mu.utils import lilyconverter


class PeriodTest(unittest.TestCase):
    def test_current_rhythm_added(self):
        p = lilyconverter.Period(1, 1, 1, 1)
        self.assertEqual(p.current_rhythm, 0)
        add_v = 1
        p.current_rhythm_add(add_v)
        self.assertEqual(p.current_rhythm, add_v)
        p = lilyconverter.Period(1, 1, 1, 1)
        add_v = 5
        p.current_rhythm_add(add_v)
        self.assertEqual(p.current_rhythm, 1)

    def test_apply_rhythm(self):
        p = lilyconverter.Period(1, 1, 1, 1)
        dur = 2.5
        expected_distr = [2, 0.5]
        self.assertEqual(p.apply_rhythm(dur),
                         expected_distr)
        dur = 1.5
        expected_distr = [0.5, 1]
        self.assertEqual(p.apply_rhythm(dur),
                         expected_distr)
        p = lilyconverter.Period(1.5, 1.5, 1.5)
        dur = 4.5
        expected_distr = [3, 1.5]
        self.assertEqual(p.apply_rhythm(dur),
                         expected_distr)

    def test_is_dotted(self):
        self.assertEqual(
            lilyconverter.Period.is_dotted(3),
            True)
        self.assertEqual(
            lilyconverter.Period.is_dotted(4.5),
            False)
        self.assertEqual(
            lilyconverter.Period.is_dotted(0.75),
            True)
        self.assertEqual(
            lilyconverter.Period.is_dotted(5.5),
            False)

