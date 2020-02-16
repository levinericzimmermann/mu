import unittest

from mu.rhy import rhy
from mu.time import time


class RhyTest(unittest.TestCase):
    def test_delay(self):
        rh0 = rhy.Unit(4)
        rhy_comp0 = rhy.Compound([rhy.Unit(2), rhy.Unit(4), rhy.Unit(1)])
        rhy_comp1 = rhy.Compound([rhy.Unit(4), rhy.Unit(8), rhy.Unit(2)])
        self.assertEqual(rh0.delay, time.Time(4))
        self.assertEqual(rhy_comp0.delay, time.Time(7))
        self.assertEqual(rhy_comp1.delay, time.Time(14))

    def test_stretch(self):
        rh0 = rhy.Unit(2)
        rh1 = rhy.Unit(4)
        rh2 = rhy.Unit(1)
        self.assertEqual(rh0.stretch(2), rh1)
        self.assertEqual(rh0.stretch(0.5), rh2)
        rhy_comp0 = rhy.Compound([rhy.Unit(2), rhy.Unit(4), rhy.Unit(1)])
        rhy_comp1 = rhy.Compound([rhy.Unit(4), rhy.Unit(8), rhy.Unit(2)])
        rhy_comp2 = rhy.Compound([rhy.Unit(1), rhy.Unit(2), rhy.Unit(0.5)])
        self.assertEqual(rhy_comp0, rhy_comp1.stretch(0.5))
        self.assertEqual(rhy_comp0.stretch(2), rhy_comp1)
        self.assertEqual(rhy_comp0.stretch(0.5), rhy_comp2)

    def test_convert2absolute(self):
        r0 = rhy.Compound((2, 2, 3, 1))
        r1 = rhy.Compound((0, 2, 4, 7, 8))
        self.assertEqual(r0.convert2absolute(), r1[:-1])
        self.assertEqual(r0.convert2absolute(skiplast=False), r1)

    def test_convert2relative(self):
        r0 = rhy.Compound((0, 2, 4, 7, 8))
        r1 = rhy.Compound((2, 2, 3, 1))
        self.assertEqual(r0.convert2relative(), r1)


class PulseChromaTest(unittest.TestCase):
    def test_subpulse(self):
        chrome0 = rhy.PulseChroma(12)
        chrome1 = rhy.PulseChroma(6)
        chrome2 = rhy.PulseChroma(3)
        chrome3 = rhy.PulseChroma(2)
        self.assertEqual(chrome0.subpulse, (chrome1,))
        self.assertEqual(set(chrome1.subpulse), set((chrome2, chrome3)))
        chrome0 = rhy.PulseChroma(90)
        chrome1 = rhy.PulseChroma(30)
        chrome2 = rhy.PulseChroma(15)
        chrome3 = rhy.PulseChroma(10)
        chrome4 = rhy.PulseChroma(6)
        chrome5 = rhy.PulseChroma(2)
        chrome6 = rhy.PulseChroma(3)
        chrome7 = rhy.PulseChroma(5)
        chrome8 = rhy.PulseChroma(1)
        self.assertEqual(chrome0.subpulse, (chrome1,))
        self.assertEqual(set(chrome1.subpulse), set((chrome2, chrome3, chrome4)))
        self.assertEqual(set(chrome2.subpulse), set((chrome6, chrome7)))
        self.assertEqual(set(chrome3.subpulse), set((chrome5, chrome7)))
        self.assertEqual(set(chrome4.subpulse), set((chrome5, chrome6)))
        self.assertEqual(chrome7.subpulse, (chrome8,))

    def test_count_subpulse(self):
        chrome0 = rhy.PulseChroma(90)
        chrome1 = rhy.PulseChroma(30)
        chrome2 = rhy.PulseChroma(15)
        chrome3 = rhy.PulseChroma(10)
        chrome4 = rhy.PulseChroma(6)
        chrome5 = rhy.PulseChroma(2)
        chrome6 = rhy.PulseChroma(1)
        self.assertEqual(chrome0.count_subpulse(), (3,))
        self.assertEqual(set(chrome1.count_subpulse()), set((2, 3, 5)))
        self.assertEqual(set(chrome2.count_subpulse()), set((3, 5)))
        self.assertEqual(set(chrome3.count_subpulse()), set((2, 5)))
        self.assertEqual(set(chrome4.count_subpulse()), set((2, 3)))
        self.assertEqual(chrome5.count_subpulse(), (2,))
        self.assertEqual(chrome6.count_subpulse(), (0,))

    def test_specify(self):
        chrome0 = rhy.PulseChroma(10)
        specified000 = rhy.SpecifiedPulseChroma(1, 0)
        specified00 = rhy.SpecifiedPulseChroma(2, specified000)
        specified01 = rhy.SpecifiedPulseChroma(5, specified000)
        specified0 = rhy.SpecifiedPulseChroma(10, specified00)
        specified1 = rhy.SpecifiedPulseChroma(10, specified01)
        self.assertEqual(chrome0.specify(), (specified0, specified1))
